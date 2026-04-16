/**
 * @file fused_3s.cu
 * @brief Fused3S: Efficient Sparse-Dense Graph Attention Kernel
 * * This file implements the fused 3-stage (SDDMM + Softmax + SpMM) kernel for sparse-dense
 * graph attention computation. The implementation leverages NVIDIA Tensor Cores (WMMA) for
 * efficient matrix multiplication and uses shared memory for numerical stability in softmax
 * and caching of the Query matrix to eliminate memory bandwidth bottlenecks.
 * * @author Mayank Choudhary
 * @date 2026-03-24
 * @version 1.0
 */

#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

using namespace nvcuda;

/** @defgroup WMMA_DIMS WMMA Tensor Core Dimensions
 * @brief Dimensions for NVIDIA Tensor Core (WMMA) operations
 * @{
 */
/** Tensor Core M dimension (row tiles) */
const int WMMA_M = 16;
/** Tensor Core N dimension (column tiles) */
const int WMMA_N = 16;
/** Tensor Core K dimension (reduction dimension) */
const int WMMA_K = 16;
/** @} */

/** @defgroup KERNEL_CONFIG Kernel Configuration Constants
 * @brief Configuration parameters for fused3S kernel execution
 * @{
 */
/** Number of 16-element chunks needed to cover the 64-dimensional feature space.
 * Calculated as: feature_dim (64) / WMMA_K (16) = 4 chunks for SpMM output dimension */
const int NUM_FEAT_CHUNKS = 4; 
/** Number of warps per thread block. Each warp processes one column block.
 * 4 warps × 32 threads/warp = 128 threads per block */
const int WARPS_PER_BLOCK = 4;
/** Fixed feature dimension assumed for static shared memory allocation */
const int FEAT_DIM_FIXED = 64;
/** @} */

/** * @brief Fused3S Kernel: 1 Tensor Block per Row Window
 * * This kernel processes one Row Window (16 rows) of the output at a time. Each block is 
 * responsible for one Row Window, and the warps within the block collaboratively compute 
 * the SDDMM and SpMM for that window. The softmax is computed in shared memory for 
 * numerical stability. Each block processes a 16x64 output tile corresponding to one Row Window, 
 * iterating over the active column blocks and performing load-balanced computations.
 * * @param Q Input query feature matrix in FP16 format
 * @param K Input key feature matrix in FP16 format
 * @param V Input value feature matrix in FP16 format
 * @param O_out Output feature matrix in FP32 format (16x64 tile per Row Window)
 * @param window_offsets Array indicating the start of each Row Window in block_col_indices
 * @param block_col_indices Column indices of the active blocks for each Row Window
 * @param window_order Array mapping logical window indices to physical window indices, 
 * sorted by workload weight for load balancing
 * @param num_row_windows Total number of Row Windows in the graph
 * @param feat_dim Feature dimension (assumed to be 64 for this implementation)
 */
__global__ void f3s_1tb1rw_kernel(
    const half* __restrict__ Q, 
    const half* __restrict__ K, 
    const half* __restrict__ V, 
    float* __restrict__ O_out, 
    const int* __restrict__ window_offsets,
    const int* __restrict__ block_col_indices,
    const int* __restrict__ window_order, // INJECTED FOR LOAD BALANCING
    int num_row_windows, 
    int feat_dim)
{
    // ===================================================================
    // THREAD AND BLOCK INITIALIZATION
    // ===================================================================
    
    int logical_window_idx = blockIdx.x;
    if (logical_window_idx >= num_row_windows) return;
    int window_idx = window_order[logical_window_idx];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int tid = threadIdx.x;

    int block_start = window_offsets[window_idx];
    int block_end = window_offsets[window_idx + 1];

    // ===================================================================
    // SHARED MEMORY ALLOCATION
    // ===================================================================
    
    /** @brief Shared memory to cache the Query matrix tile for this Row Window.
     * Eliminates redundant global memory loads during the SDDMM phase. */
    __shared__ half smem_Q[WMMA_M][FEAT_DIM_FIXED];

    __shared__ float smem_S[WARPS_PER_BLOCK][16][16];
    __shared__ half  smem_P[WARPS_PER_BLOCK][16][16];

    // ===================================================================
    // COLLABORATIVE QUERY (Q) CACHING
    // ===================================================================
    
    /** @brief All threads in the block cooperatively load the 16x64 tile of Q
     * from global memory into shared memory. (1024 elements / 128 threads = 8 elements/thread) */
    int total_elements_Q = WMMA_M * feat_dim;
    for (int i = tid; i < total_elements_Q; i += blockDim.x) {
        int r = i / feat_dim;
        int c = i % feat_dim;
        smem_Q[r][c] = Q[(window_idx * WMMA_M * feat_dim) + i];
    }
    
    // Ensure the entire Q tile is loaded before any warp begins SDDMM
    __syncthreads();

    // ===================================================================
    // INITIALIZE OUTPUT ACCUMULATORS
    // ===================================================================
    
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_O[NUM_FEAT_CHUNKS];
    for (int c = 0; c < NUM_FEAT_CHUNKS; ++c) {
        wmma::fill_fragment(frag_O[c], 0.0f);
    }

    // ===================================================================
    // PROCESS ACTIVE COLUMN BLOCKS (THREE-STAGE FUSED COMPUTATION)
    // ===================================================================
    
    for (int b = block_start + warp_id; b < block_end; b += WARPS_PER_BLOCK) {
        int block_col = block_col_indices[b];

        // ===================================================================
        // PHASE 1: SDDMM - Attention Score Computation (Q @ K^T)
        // ===================================================================
        
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_Q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_K;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_S;

        wmma::fill_fragment(frag_S, 0.0f);

        for (int k_step = 0; k_step < feat_dim; k_step += WMMA_K) {
            // LOAD Q FROM SHARED MEMORY (Cached)
            const half* smem_Q_ptr = &smem_Q[0][k_step];
            // LOAD K FROM GLOBAL MEMORY (Dependent on block_col)
            const half* K_ptr = K + (block_col * WMMA_N * feat_dim) + k_step; 

            // Stride for shared memory Q is feat_dim
            wmma::load_matrix_sync(frag_Q, smem_Q_ptr, feat_dim);
            wmma::load_matrix_sync(frag_K, K_ptr, feat_dim);
            wmma::mma_sync(frag_S, frag_Q, frag_K, frag_S);
        }

        // ===================================================================
        // PHASE 2: Softmax - Attention Weight Normalization
        // ===================================================================
        
        wmma::store_matrix_sync(&smem_S[warp_id][0][0], frag_S, 16, wmma::mem_row_major);
        __syncwarp();

        if (lane_id < 16) {
            int row = lane_id;
            
            float max_val = -1e9f;
            for (int c = 0; c < 16; ++c) {
                max_val = fmaxf(max_val, smem_S[warp_id][row][c]);
            }
            
            float sum_val = 0.0f;
            for (int c = 0; c < 16; ++c) {
                float exp_val = expf(smem_S[warp_id][row][c] - max_val);
                smem_S[warp_id][row][c] = exp_val;
                sum_val += exp_val;
            }
            
            for (int c = 0; c < 16; ++c) {
                smem_S[warp_id][row][c] /= sum_val;
            }
        }
        __syncwarp();

        for (int i = 0; i < 8; ++i) {
            int idx = lane_id * 8 + i;
            int r = idx / 16;
            int c = idx % 16;
            smem_P[warp_id][r][c] = __float2half(smem_S[warp_id][r][c]);
        }
        __syncwarp();

        // ===================================================================
        // PHASE 3: SpMM - Output Computation (P @ V)
        // ===================================================================
        
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_P;
        wmma::load_matrix_sync(frag_P, &smem_P[warp_id][0][0], 16);

        for (int c = 0; c < NUM_FEAT_CHUNKS; ++c) {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_V; 
            const half* V_ptr = V + (block_col * WMMA_N * feat_dim) + (c * WMMA_K);
            
            wmma::load_matrix_sync(frag_V, V_ptr, feat_dim); 
            wmma::mma_sync(frag_O[c], frag_P, frag_V, frag_O[c]);
        }
    }

    // ===================================================================
    // FINAL OUTPUT WRITE
    // ===================================================================
    
    for (int c = 0; c < NUM_FEAT_CHUNKS; ++c) {
        float* O_ptr = O_out + (window_idx * WMMA_M * feat_dim) + (c * WMMA_K);
        wmma::store_matrix_sync(O_ptr, frag_O[c], feat_dim, wmma::mem_row_major);
    }
}