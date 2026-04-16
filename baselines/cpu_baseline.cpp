#include <cmath>
#include <algorithm>
#include <vector>
/**
 * Naive CPU implementation of sparse attention: SDDMM + Softmax + SpMM.
 * Computes O = softmax(Q[i] · K[neighbors]^T) @ V[neighbors] for each node i
 * and its neighbors defined by the sparse adjacency structure.
 * 
 * Parameters:
 * - num_nodes: number of nodes
 * - feat_dim: feature dimension
 * - Q, K, V: query, key, value matrices (num_nodes x feat_dim each)
 * - row_ptr, col_idx: CSR format sparse adjacency matrix
 * - out_mat: output matrix to store results (num_nodes x feat_dim)
 */
void cpu_sparse_attention_csr(int num_nodes, int feat_dim, 
                              const float* Q, const float* K, const float* V,
                              const int* row_ptr, const int* col_idx, 
                              float* out_mat) {
    // Process each node
    for (int i = 0; i < num_nodes; ++i) {
        int neighbor_start = row_ptr[i];
        int neighbor_end = row_ptr[i + 1];
        int num_neighbors = neighbor_end - neighbor_start;
        
        if (num_neighbors == 0) {
            // No neighbors: output is zero
            for (int f = 0; f < feat_dim; ++f) {
                out_mat[i * feat_dim + f] = 0.0f;
            }
            continue;
        }
        
        // ===================================================================
        // PHASE 1: SDDMM - Compute attention scores Q[i] @ K[neighbors]^T
        // ===================================================================
        std::vector<float> attention_scores(num_neighbors, 0.0f);
        
        for (int idx = 0; idx < num_neighbors; ++idx) {
            int neighbor_col = col_idx[neighbor_start + idx];
            
            // Compute dot product: S[j] = Q[i] · K[neighbor_j]
            float dot_product = 0.0f;
            for (int f = 0; f < feat_dim; ++f) {
                dot_product += Q[i * feat_dim + f] * K[neighbor_col * feat_dim + f];
            }
            
            attention_scores[idx] = dot_product;
        }
        
        // ===================================================================
        // PHASE 2: Softmax - Numerically stable normalization
        // ===================================================================
        
        // Find max for numerical stability
        float max_score = -1e9f;
        for (int idx = 0; idx < num_neighbors; ++idx) {
            max_score = std::max(max_score, attention_scores[idx]);
        }
        
        // Compute exp(x - max) and accumulate sum
        float sum_exp = 0.0f;
        for (int idx = 0; idx < num_neighbors; ++idx) {
            attention_scores[idx] = std::exp(attention_scores[idx] - max_score);
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
            out_mat[i * feat_dim + f] = 0.0f;
        }
        
        // Accumulate weighted sum of neighbor features
        for (int idx = 0; idx < num_neighbors; ++idx) {
            int neighbor_col = col_idx[neighbor_start + idx];
            float weight = attention_scores[idx];
            
            for (int f = 0; f < feat_dim; ++f) {
                out_mat[i * feat_dim + f] += weight * V[neighbor_col * feat_dim + f];
            }
        }
    }
}