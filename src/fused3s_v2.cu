/**
 * @file fused3s_temp2.cu
 * @brief Fused3S v3: Optimized Online Softmax with O in Registers
 *
 * Key optimizations over fused3s_temp.cu:
 *   1. Output accumulator (frag_O) stays in WMMA register fragments — eliminates
 *      the costly smem_O[16][64] shared memory roundtrip every iteration
 *   2. In-register rescaling using WMMA fragment layout mapping (sm_80+):
 *      directly multiply frag_O.x[] elements by the per-row correction factor
 *   3. Removed one __syncthreads() per iteration (after SpMM) since SpMM only
 *      writes to registers, not shared memory
 *   4. Each warp permanently owns its output feature chunk (warp i → chunk i),
 *      so SpMM accumulation is fully register-resident
 *
 * WMMA Fragment Layout (sm_80+, m16n16k16, float accumulator):
 *   groupID = lane_id / 4
 *   Elements 0,1,4,5 → row = groupID
 *   Elements 2,3,6,7 → row = groupID + 8
 *
 * @author Mayank Choudhary
 * @date 2026-04-15
 * @version 3.0 (Register-Resident O + Fragment Rescaling)
 * @note Requires sm_80+ for correct WMMA fragment layout assumptions
 */

#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

using namespace nvcuda;

// =====================================================================
// CONFIGURATION
// =====================================================================

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
const int NUM_FEAT_CHUNKS = 4;   // feat_dim(64) / WMMA_K(16)
const int WARPS_PER_BLOCK = 4;   // 1:1 mapping with NUM_FEAT_CHUNKS
const int FEAT_DIM_FIXED  = 64;

// =====================================================================
// HELPER: Apply per-row scaling to a WMMA accumulator fragment
// =====================================================================
/**
 * @brief Multiply each element of a 16x16 float accumulator fragment by a
 *        per-row scale factor, using the known sm_80+ fragment layout.
 *
 * @param frag   The accumulator fragment to scale in-place
 * @param scale  Array of 16 scale factors (one per row), in shared memory
 * @param lane_id  The lane index within the warp (threadIdx.x)
 */
__device__ __forceinline__ void scale_fragment_by_row(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& frag,
    const float* __restrict__ scale,
    int lane_id)
{
    int groupID = lane_id / 4;
    float s0 = scale[groupID];       // rows 0-7
    float s1 = scale[groupID + 8];   // rows 8-15

    frag.x[0] *= s0;
    frag.x[1] *= s0;
    frag.x[2] *= s1;
    frag.x[3] *= s1;
    frag.x[4] *= s0;
    frag.x[5] *= s0;
    frag.x[6] *= s1;
    frag.x[7] *= s1;
}

// =====================================================================
// KERNEL
// =====================================================================
/**
 * @brief Fused SDDMM + Online-Softmax + SpMM with register-resident output.
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
    // THREAD IDENTIFICATION (2D layout)
    // =================================================================

    int logical_window_idx = blockIdx.x;
    if (logical_window_idx >= num_row_windows) return;
    int window_idx = window_order[logical_window_idx];

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int tid     = threadIdx.x + threadIdx.y * blockDim.x;

    int block_start = window_offsets[window_idx];
    int block_end   = window_offsets[window_idx + 1];

    // =================================================================
    // SHARED MEMORY (compact — no smem_O!)
    // =================================================================

    __shared__ half  smem_Q[WMMA_M][FEAT_DIM_FIXED];  // Q cache: 2 KB
    __shared__ float smem_S[16][16];                    // SDDMM scores: 1 KB
    __shared__ half  smem_P[16][16];                    // softmax weights: 0.5 KB
    __shared__ float row_max[WMMA_M];                   // running max
    __shared__ float row_sum[WMMA_M];                   // running exp-sum
    __shared__ float correction[WMMA_M];                // rescale factor for O

    // =================================================================
    // INITIALIZATION
    // =================================================================

    // Cooperative Q load
    int total_Q = WMMA_M * feat_dim;
    for (int i = tid; i < total_Q; i += blockDim.x * blockDim.y) {
        int r = i / feat_dim;
        int c = i % feat_dim;
        smem_Q[r][c] = Q[(window_idx * WMMA_M * feat_dim) + i];
    }
    if (tid < WMMA_M) {
        row_max[tid] = -1e30f;
        row_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Each warp owns ONE output chunk (warp i → chunk i)
    // frag_O lives in registers for the entire kernel lifetime
    int my_chunk = warp_id;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_O;
    wmma::fill_fragment(frag_O, 0.0f);

    // =================================================================
    // MAIN LOOP — sequential over column blocks for online softmax
    // =================================================================

    for (int b = block_start; b < block_end; ++b) {
        int block_col = block_col_indices[b];

        // =============================================================
        // PHASE 1: SDDMM (warp 0 — 4 MMA ops for 64-dim)
        // =============================================================

        if (warp_id == 0) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_Q;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_K;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_S;
            wmma::fill_fragment(frag_S, 0.0f);

            #pragma unroll
            for (int k = 0; k < feat_dim; k += WMMA_K) {
                wmma::load_matrix_sync(frag_Q, &smem_Q[0][k], feat_dim);
                wmma::load_matrix_sync(frag_K, K + block_col * WMMA_N * feat_dim + k, feat_dim);
                wmma::mma_sync(frag_S, frag_Q, frag_K, frag_S);
            }
            wmma::store_matrix_sync(&smem_S[0][0], frag_S, 16, wmma::mem_row_major);
        }
        __syncthreads();  // all warps need smem_S

        // =============================================================
        // PHASE 2: Online Softmax — warp 0, lanes 0-15
        //   Updates: correction[], smem_P[], row_max[], row_sum[]
        // =============================================================

        if (warp_id == 0 && lane_id < 16) {
            int row = lane_id;

            // Block-local max
            float bmax = -1e30f;
            #pragma unroll
            for (int c = 0; c < 16; ++c)
                bmax = fmaxf(bmax, smem_S[row][c]);

            float old_max = row_max[row];
            float new_max = fmaxf(old_max, bmax);
            float corr = expf(old_max - new_max);

            // Publish correction for all warps to rescale their frag_O
            correction[row] = corr;

            // Rescale running sum
            row_sum[row] *= corr;

            // Exponentiate scores and accumulate block sum
            float bsum = 0.0f;
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float e = expf(smem_S[row][c] - new_max);
                smem_P[row][c] = __float2half(e);
                bsum += e;
            }

            row_sum[row] += bsum;
            row_max[row] = new_max;
        }
        __syncthreads();  // all warps need correction[] and smem_P[]

        // =============================================================
        // PHASE 2b: All warps rescale their frag_O IN-REGISTER
        // =============================================================

        scale_fragment_by_row(frag_O, correction, lane_id);

        // =============================================================
        // PHASE 3: SpMM — each warp does P @ V_chunk for its own chunk
        // =============================================================

        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_P;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_V;

            wmma::load_matrix_sync(frag_P, &smem_P[0][0], 16);
            wmma::load_matrix_sync(frag_V,
                V + block_col * WMMA_N * feat_dim + my_chunk * WMMA_K,
                feat_dim);
            wmma::mma_sync(frag_O, frag_P, frag_V, frag_O);
        }
        // No __syncthreads() needed — SpMM only writes to registers,
        // and next iteration's Phase 1 writes smem_S (disjoint from smem_P)
    }

    // =================================================================
    // FINAL: Normalize frag_O by row_sum, then write to global memory
    // =================================================================

    // Build inverse-sum scale array in shared memory (reuse correction[])
    if (warp_id == 0 && lane_id < 16) {
        float s = row_sum[lane_id];
        correction[lane_id] = (s > 0.0f) ? (1.0f / s) : 0.0f;
    }
    __syncthreads();

    scale_fragment_by_row(frag_O, correction, lane_id);

    // Each warp writes its chunk to global memory
    float* O_ptr = O_out + window_idx * WMMA_M * feat_dim + my_chunk * WMMA_K;
    wmma::store_matrix_sync(O_ptr, frag_O, feat_dim, wmma::mem_row_major);
}
