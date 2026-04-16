/**
 * @file fused3s_temp.cu
 * @brief Fused3S v2: Online Softmax + 2D Thread Block Layout
 *
 * Improvements over fused_3s.cu:
 *   1. Online (streaming) softmax — correct across ALL active column blocks per row
 *   2. 2D thread block layout — dim3(32, WARPS_PER_BLOCK) with threadIdx.y as warp ID
 *
 * The online softmax maintains running max and sum across column blocks, rescaling
 * the output accumulator when the max changes, so the final result equals a global
 * softmax computed across all column blocks simultaneously.
 *
 * @author Mayank Choudhary
 * @date 2026-04-4
 * @version 2.0 (Online Softmax + 2D Layout)
 */

#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

using namespace nvcuda;

// =====================================================================
// CONFIGURATION
// =====================================================================

/** Tensor Core dimensions */
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

/** Kernel configuration */
const int NUM_FEAT_CHUNKS = 4;   // feat_dim(64) / WMMA_K(16)
const int WARPS_PER_BLOCK = 4;   // threadIdx.y range
const int FEAT_DIM_FIXED  = 64;

// =====================================================================
// KERNEL: Online Softmax Fused3S with 2D Thread Blocks
// =====================================================================
/**
 * @brief Fused SDDMM + Online-Softmax + SpMM kernel.
 *
 * Each thread block processes one Row Window (16 rows). Column blocks are
 * iterated sequentially so that the online softmax running state (max, sum)
 * is coherent across all active column blocks for each row.
 *
 * Launch config: <<<num_row_windows, dim3(32, WARPS_PER_BLOCK)>>>
 */
__global__ void f3s_1tb1rw_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ O_out,
    const int* __restrict__ window_offsets,
    const int* __restrict__ block_col_indices,
    const int* __restrict__ window_order,
    int num_row_windows,
    int feat_dim)
{
    // =================================================================
    // THREAD IDENTIFICATION (2D Layout)
    // =================================================================

    int logical_window_idx = blockIdx.x;
    if (logical_window_idx >= num_row_windows) return;
    int window_idx = window_order[logical_window_idx];

    int warp_id = threadIdx.y;              // 0..WARPS_PER_BLOCK-1
    int lane_id = threadIdx.x;              // 0..31
    int tid = threadIdx.x + threadIdx.y * blockDim.x;  // flat thread id

    int block_start = window_offsets[window_idx];
    int block_end   = window_offsets[window_idx + 1];

    // =================================================================
    // SHARED MEMORY
    // =================================================================

    // Q tile cache: 16 × 64 half
    __shared__ half smem_Q[WMMA_M][FEAT_DIM_FIXED];

    // Scratch for SDDMM result (one 16×16 block at a time, single-writer)
    __shared__ float smem_S[16][16];

    // Softmax half-precision weights (after exp & before SpMM)
    __shared__ half smem_P[16][16];

    // Online softmax running state per row
    __shared__ float row_max[WMMA_M];   // running max
    __shared__ float row_sum[WMMA_M];   // running sum of exp

    // Output accumulator in shared memory (needed for rescaling by all warps)
    __shared__ float smem_O[WMMA_M][FEAT_DIM_FIXED];

    // =================================================================
    // INITIALIZATION
    // =================================================================

    // Cooperatively load Q[16×64] into shared memory
    int total_elements_Q = WMMA_M * feat_dim;
    for (int i = tid; i < total_elements_Q; i += blockDim.x * blockDim.y) {
        int r = i / feat_dim;
        int c = i % feat_dim;
        smem_Q[r][c] = Q[(window_idx * WMMA_M * feat_dim) + i];
    }

    // Initialize O accumulator to zero
    int total_elements_O = WMMA_M * FEAT_DIM_FIXED;
    for (int i = tid; i < total_elements_O; i += blockDim.x * blockDim.y) {
        int r = i / FEAT_DIM_FIXED;
        int c = i % FEAT_DIM_FIXED;
        smem_O[r][c] = 0.0f;
    }

    // Initialize online softmax state
    if (tid < WMMA_M) {
        row_max[tid] = -1e30f;
        row_sum[tid] = 0.0f;
    }

    __syncthreads();

    // =================================================================
    // MAIN LOOP: Process column blocks SEQUENTIALLY for correct softmax
    // =================================================================

    for (int b = block_start; b < block_end; ++b) {
        int block_col = block_col_indices[b];

        // =============================================================
        // PHASE 1: SDDMM (Q @ K^T) — Warp 0 computes the 16×16 scores
        // =============================================================

        if (warp_id == 0) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_Q;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_K;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_S;

            wmma::fill_fragment(frag_S, 0.0f);

            for (int k_step = 0; k_step < feat_dim; k_step += WMMA_K) {
                const half* smem_Q_ptr = &smem_Q[0][k_step];
                const half* K_ptr = K + (block_col * WMMA_N * feat_dim) + k_step;

                wmma::load_matrix_sync(frag_Q, smem_Q_ptr, feat_dim);
                wmma::load_matrix_sync(frag_K, K_ptr, feat_dim);
                wmma::mma_sync(frag_S, frag_Q, frag_K, frag_S);
            }

            // Store S to shared memory
            wmma::store_matrix_sync(&smem_S[0][0], frag_S, 16, wmma::mem_row_major);
        }

        // All warps must see the SDDMM result
        __syncthreads();

        // =============================================================
        // PHASE 2: Online Softmax Update
        //   - Compute new max, correction factor, rescale O & sum
        //   - Exponentiate current block scores
        //   - Convert to half for SpMM
        // =============================================================

        // Step 2a: Each of the first 16 threads handles one row
        if (warp_id == 0 && lane_id < 16) {
            int row = lane_id;

            // Find max of current block's row
            float block_max = -1e30f;
            for (int c = 0; c < 16; ++c) {
                block_max = fmaxf(block_max, smem_S[row][c]);
            }

            // New global max
            float old_max = row_max[row];
            float new_max = fmaxf(old_max, block_max);

            // Correction factor for previously accumulated values
            float correction = expf(old_max - new_max);

            // Rescale the running sum
            row_sum[row] = row_sum[row] * correction;

            // Rescale the output accumulator
            for (int f = 0; f < FEAT_DIM_FIXED; ++f) {
                smem_O[row][f] *= correction;
            }

            // Exponentiate current block scores and accumulate sum
            float block_sum = 0.0f;
            for (int c = 0; c < 16; ++c) {
                float exp_val = expf(smem_S[row][c] - new_max);
                smem_S[row][c] = exp_val;
                block_sum += exp_val;
            }

            row_sum[row] += block_sum;
            row_max[row] = new_max;

            // Convert to half for SpMM
            for (int c = 0; c < 16; ++c) {
                smem_P[row][c] = __float2half(smem_S[row][c]);
            }
        }

        // All warps must see the updated smem_P and smem_O
        __syncthreads();

        // =============================================================
        // PHASE 3: SpMM (P @ V) — Warps split across feature chunks
        // =============================================================

        for (int chunk = warp_id; chunk < NUM_FEAT_CHUNKS; chunk += WARPS_PER_BLOCK) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_P;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_V;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_O;

            // Load P from shared memory
            wmma::load_matrix_sync(frag_P, &smem_P[0][0], 16);

            // Load V chunk from global memory
            const half* V_ptr = V + (block_col * WMMA_N * feat_dim) + (chunk * WMMA_K);
            wmma::load_matrix_sync(frag_V, V_ptr, feat_dim);

            // Load current O accumulator from shared memory
            wmma::load_matrix_sync(frag_O, &smem_O[0][chunk * WMMA_K], FEAT_DIM_FIXED, wmma::mem_row_major);

            // Accumulate: O += P @ V
            wmma::mma_sync(frag_O, frag_P, frag_V, frag_O);

            // Store back to shared memory
            wmma::store_matrix_sync(&smem_O[0][chunk * WMMA_K], frag_O, FEAT_DIM_FIXED, wmma::mem_row_major);
        }

        // Ensure all warps finish SpMM before next iteration's rescaling
        __syncthreads();
    }

    // =================================================================
    // FINAL NORMALIZATION: O /= row_sum
    // =================================================================

    for (int i = tid; i < WMMA_M * FEAT_DIM_FIXED; i += blockDim.x * blockDim.y) {
        int r = i / FEAT_DIM_FIXED;
        int c = i % FEAT_DIM_FIXED;
        float sum = row_sum[r];
        if (sum > 0.0f) {
            smem_O[r][c] /= sum;
        }
    }

    __syncthreads();

    // =================================================================
    // WRITE OUTPUT TO GLOBAL MEMORY
    // =================================================================

    float* O_base = O_out + (window_idx * WMMA_M * feat_dim);
    for (int i = tid; i < WMMA_M * feat_dim; i += blockDim.x * blockDim.y) {
        int r = i / feat_dim;
        int c = i % feat_dim;
        O_base[r * feat_dim + c] = smem_O[r][c];
    }
}
