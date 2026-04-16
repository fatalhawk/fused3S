/**
 * @brief Naive Baseline Kernel: SDDMM + Softmax + SpMM (Non-Fused)
 * 
 * This kernel implements all three stages of graph attention computation in a naive,
 * non-optimized manner for performance baseline comparison. Each thread processes one
 * row, computing attention scores, applying softmax normalization, then aggregating
 * neighbor features.
 * 
 * Computation stages:
 * 1. SDDMM: Compute attention scores S[i][j] = Q[i] · K[j]^T for each neighbor j
 * 2. Softmax: Normalize scores row-wise with numerically stable softmax
 * 3. SpMM: Compute output O[i] = Σ_j softmax(S[i][j]) * V[j]
 * 
 * Parameters:
 * - Q: Query matrix (dense, row-major, FP32), shape: [num_nodes, feat_dim]
 * - K: Key matrix (dense, row-major, FP32), shape: [num_nodes, feat_dim]
 * - V: Value matrix (dense, row-major, FP32), shape: [num_nodes, feat_dim]
 * - O_out: Output matrix (FP32), shape: [num_nodes, feat_dim]
 * - row_ptr: CSR row pointer array for sparse adjacency
 * - col_idx: CSR column indices for sparse adjacency
 * - num_nodes: Total number of nodes
 * - feat_dim: Feature dimension
 * 
 * @note This is a naive baseline without any optimization (no Tensor Cores, no
 *       shared memory optimization). It processes one node per thread.
 * @note Shared memory allocation must be dynamic for storing attention scores.
 */
__global__ void f3s_naive_baseline_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O_out,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int num_nodes,
    int feat_dim)
{
    int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= num_nodes) return;

    int neighbor_start = row_ptr[node_id];
    int neighbor_end = row_ptr[node_id + 1];
    int num_neighbors = neighbor_end - neighbor_start;

    if (num_neighbors == 0) {
        // No neighbors: output is zero
        for (int f = 0; f < feat_dim; ++f) {
            O_out[node_id * feat_dim + f] = 0.0f;
        }
        return;
    }

    // ===================================================================
    // PHASE 1: SDDMM - Compute attention scores Q[i] @ K[neighbors]^T
    // ===================================================================
    
    // Allocate space on stack for attention scores (per-neighbor)
    float* attention_scores = (float*)malloc(num_neighbors * sizeof(float));
    
    // Compute dot product: S[j] = Q[i] · K[neighbor_j]
    for (int idx = 0; idx < num_neighbors; ++idx) {
        int neighbor_col = col_idx[neighbor_start + idx];
        
        float dot_product = 0.0f;
        for (int f = 0; f < feat_dim; ++f) {
            float q_val = Q[node_id * feat_dim + f];
            float k_val = K[neighbor_col * feat_dim + f];
            dot_product += q_val * k_val;
        }
        
        attention_scores[idx] = dot_product;
    }

    // ===================================================================
    // PHASE 2: Softmax - Numerically stable normalization
    // ===================================================================
    
    // Find max for numerical stability
    float max_score = -1e9f;
    for (int idx = 0; idx < num_neighbors; ++idx) {
        max_score = fmaxf(max_score, attention_scores[idx]);
    }

    // Compute exp(x - max) and accumulate sum
    float sum_exp = 0.0f;
    for (int idx = 0; idx < num_neighbors; ++idx) {
        attention_scores[idx] = expf(attention_scores[idx] - max_score);
        sum_exp += attention_scores[idx];
    }

    // Normalize to get probabilities
    for (int idx = 0; idx < num_neighbors; ++idx) {
        attention_scores[idx] /= sum_exp;
    }

    // ===================================================================
    // PHASE 3: SpMM - Weighted aggregation O[i] = P[neighbors] @ V[neighbors]
    // ===================================================================
    
    // Initialize output to zero
    for (int f = 0; f < feat_dim; ++f) {
        O_out[node_id * feat_dim + f] = 0.0f;
    }

    // Accumulate weighted sum of neighbor features
    for (int idx = 0; idx < num_neighbors; ++idx) {
        int neighbor_col = col_idx[neighbor_start + idx];
        float weight = attention_scores[idx];
        
        for (int f = 0; f < feat_dim; ++f) {
            O_out[node_id * feat_dim + f] += weight * V[neighbor_col * feat_dim + f];
        }
    }

    // Cleanup
    free(attention_scores);
}