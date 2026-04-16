#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cuda_fp16.h>
#include "../include/utils.cuh"


blockSparseFormat convert_csr_to_block_sparse(int num_nodes, const std::vector<int>& row_ptr, const std::vector<int>& col_ind);

// Forward-declare the kernel with exact signature from fused3s_temp.cu
__global__ void f3s_1tb1rw_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, float* __restrict__ O_out, 
    const int* __restrict__ window_offsets, const int* __restrict__ block_col_indices, const int* __restrict__ window_order, 
    int num_row_windows, int feat_dim);

// --- Exact Block-Sparse CPU Ground Truth (global softmax across all column blocks) ---
void cpu_block_sparse_attention(int num_nodes, int feat_dim, const float* Q, const float* K, const float* V, float* O, const blockSparseFormat& format) {
    int num_row_windows = format.num_row_windows;
    
    for (int w = 0; w < num_row_windows; ++w) {
        int block_start = format.window_offsets[w];
        int block_end = format.window_offsets[w + 1];

        for (int r = 0; r < 16; ++r) {
            int global_row = w * 16 + r;
            if (global_row >= num_nodes) continue;

            std::vector<float> S;
            std::vector<int> col_indices;
            
            // 1. SDDMM
            for (int b = block_start; b < block_end; ++b) {
                int block_col = format.block_col_indices[b];
                for (int c = 0; c < 16; ++c) {
                    int global_col = block_col * 16 + c;
                    float sum = 0.0f;
                    if (global_col < num_nodes) {
                        for (int f = 0; f < feat_dim; ++f) {
                            sum += Q[global_row * feat_dim + f] * K[global_col * feat_dim + f];
                        }
                    } else {
                        sum = -1e9f;
                    }
                    S.push_back(sum);
                    col_indices.push_back(global_col);
                }
            }

            // 2. Softmax (global across ALL column blocks)
            if (S.empty()) continue;
            float max_val = -1e9f;
            for (float val : S) max_val = std::max(max_val, val);

            float exp_sum = 0.0f;
            std::vector<float> P(S.size());
            for (size_t i = 0; i < S.size(); ++i) {
                P[i] = std::exp(S[i] - max_val);
                exp_sum += P[i];
            }
            for (size_t i = 0; i < P.size(); ++i) P[i] /= exp_sum;

            // 3. SpMM
            for (int f = 0; f < feat_dim; ++f) {
                float out_val = 0.0f;
                for (size_t i = 0; i < P.size(); ++i) {
                    int global_col = col_indices[i];
                    if (global_col < num_nodes) {
                        out_val += P[i] * V[global_col * feat_dim + f];
                    }
                }
                O[global_row * feat_dim + f] = out_val;
            }
        }
    }
}

// Utility Kernel: Cast FP32 to FP16
__global__ void float2half_kernel(const float* in, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = __float2half(in[idx]);
}

int main(int argc, char** argv) {
    std::string dataset_path = "data/cora"; 
    if (argc > 1) dataset_path = argv[1];
    int feat_dim = 64; 
    const int WARPS_PER_BLOCK = 4;

    std::cout << "Loading graph from: " << dataset_path << std::endl;
    GraphData graph = load_csr_graph(dataset_path);
    
    blockSparseFormat format = convert_csr_to_block_sparse(graph.num_nodes, graph.row_ptr, graph.col_ind);
    std::vector<int> window_order = generate_balanced_window_order(format);

    int padded_nodes = format.num_row_windows * 16;
    size_t feat_size = padded_nodes * feat_dim;

    std::vector<float> h_Q(feat_size, 0.0f), h_K(feat_size, 0.0f), h_V(feat_size, 0.0f);
    std::vector<float> h_O_cpu(feat_size, 0.0f);
    std::vector<float> h_O_gpu(feat_size, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < graph.num_nodes * feat_dim; ++i) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }

    std::cout << "Computing CPU Block-Sparse Ground Truth..." << std::endl;
    cpu_block_sparse_attention(graph.num_nodes, feat_dim, h_Q.data(), h_K.data(), h_V.data(), h_O_cpu.data(), format);

    // Allocate Device Memory
    float *d_Q_f32, *d_K_f32, *d_V_f32, *d_O_out;
    half *d_Q, *d_K, *d_V;
    int *d_window_offsets, *d_block_col_indices, *d_window_order;

    CUDA_CHECK(cudaMalloc(&d_Q_f32, feat_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_f32, feat_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_f32, feat_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q, feat_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, feat_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, feat_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O_out, feat_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_O_out, 0, feat_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_window_offsets, format.window_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_col_indices, format.block_col_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_window_order, window_order.size() * sizeof(int)));

    // Copy to Device & Cast
    CUDA_CHECK(cudaMemcpy(d_Q_f32, h_Q.data(), feat_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_f32, h_K.data(), feat_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_f32, h_V.data(), feat_size * sizeof(float), cudaMemcpyHostToDevice));

    int cast_threads = 256;
    int cast_blocks = (feat_size + cast_threads - 1) / cast_threads;
    float2half_kernel<<<cast_blocks, cast_threads>>>(d_Q_f32, d_Q, feat_size);
    float2half_kernel<<<cast_blocks, cast_threads>>>(d_K_f32, d_K, feat_size);
    float2half_kernel<<<cast_blocks, cast_threads>>>(d_V_f32, d_V, feat_size);

    CUDA_CHECK(cudaMemcpy(d_window_offsets, format.window_offsets.data(), format.window_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_col_indices, format.block_col_indices.data(), format.block_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_window_order, window_order.data(), window_order.size() * sizeof(int), cudaMemcpyHostToDevice));

    // =====================================================================
    // Launch with 2D Thread Block: dim3(32, WARPS_PER_BLOCK)
    // =====================================================================
    std::cout << "Launching Fused3S v2 (Online Softmax, 2D Blocks)..." << std::endl;
    int num_row_windows = format.num_row_windows;
    dim3 block_dim(32, WARPS_PER_BLOCK);  // 2D: 32 lanes × 4 warps = 128 threads
    dim3 grid_dim(num_row_windows);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    f3s_1tb1rw_kernel<<<grid_dim, block_dim>>>(
        d_Q, d_K, d_V, d_O_out, d_window_offsets, d_block_col_indices, d_window_order, num_row_windows, feat_dim
    );
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel Execution Time: " << ms << " ms\n" << std::endl;

    // Verify Results
    CUDA_CHECK(cudaMemcpy(h_O_gpu.data(), d_O_out, feat_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool passed = true;
    float tolerance = 0.05f;
    int mismatch_count = 0;

    for (int i = 0; i < graph.num_nodes; ++i) {
        for (int j = 0; j < feat_dim; ++j) {
            float cpu_val = h_O_cpu[i * feat_dim + j];
            float gpu_val = h_O_gpu[i * feat_dim + j];

            if (std::fabs(cpu_val - gpu_val) > tolerance) {
                passed = false;
                mismatch_count++;
                if (mismatch_count <= 5) {
                    std::cerr << "Mismatch at node " << i << ", feat " << j << ": CPU=" << cpu_val << ", GPU=" << gpu_val 
                              << " (diff=" << std::fabs(cpu_val - gpu_val) << ")" << std::endl;
                }
            }
        }
    }
    
    if (!passed) {
        std::cerr << "Total mismatches: " << mismatch_count << " / " << graph.num_nodes * feat_dim << std::endl;
    }

    if (passed) {
        std::cout << "SUCCESS: Fused3S v2 (Online Softmax) matches exact CPU baseline!" << std::endl;
    } else {
        std::cout << "FAILURE: Results diverge beyond tolerance limits." << std::endl;
    }

    // Cleanup
    cudaFree(d_Q_f32); cudaFree(d_K_f32); cudaFree(d_V_f32); 
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O_out);
    cudaFree(d_window_offsets); cudaFree(d_block_col_indices); cudaFree(d_window_order);
    return 0;
}
