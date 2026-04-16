#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include "../include/utils.cuh"

// Forward declarations
void cpu_sparse_attention_csr(int num_nodes, int feat_dim, 
                              const float* Q, const float* K, const float* V,
                              const int* row_ptr, const int* col_idx, 
                              float* out_mat);
__global__ void f3s_naive_baseline_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O_out,
                                          const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                                          int num_nodes, int feat_dim);

// Utility to initialize random feature matrices
void init_random_matrix(std::vector<float>& mat, int size) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        mat[i] = dist(gen);
    }
}

// Utility to verify GPU results against CPU ground truth
bool verify_results(const std::vector<float>& cpu_out, const std::vector<float>& gpu_out, float tolerance = 1e-4f) {
    for (size_t i = 0; i < cpu_out.size(); ++i) {
        if (std::fabs(cpu_out[i] - gpu_out[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << cpu_out[i] << ", GPU=" << gpu_out[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // 1. Setup and Configuration
    std::string dataset_path = "data/citeseer";
    if (argc > 1) dataset_path = argv[1];
    
    int feat_dim = 64; // Standard hidden dimension size for GNNs

    std::cout << "Loading graph from: " << dataset_path << std::endl;
    GraphData graph = load_csr_graph(dataset_path);
    std::cout << "Nodes: " << graph.num_nodes << ", Edges: " << graph.num_edges << std::endl;

    // 2. Allocate Host Memory for Q, K, V
    size_t feat_size = graph.num_nodes * feat_dim;
    std::vector<float> h_Q(feat_size);
    std::vector<float> h_K(feat_size);
    std::vector<float> h_V(feat_size);
    std::vector<float> h_out_cpu(feat_size, 0.0f);
    std::vector<float> h_out_gpu(feat_size, 0.0f);

    init_random_matrix(h_Q, feat_size);
    init_random_matrix(h_K, feat_size);
    init_random_matrix(h_V, feat_size);

    // 3. Allocate Device Memory
    int *d_row_ptr, *d_col_idx;
    float *d_Q, *d_K, *d_V, *d_out_mat;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (graph.num_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, graph.num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Q, feat_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, feat_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, feat_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_mat, feat_size * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, graph.row_ptr.data(), (graph.num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, graph.col_ind.data(), graph.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), feat_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), feat_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), feat_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out_mat, 0, feat_size * sizeof(float)));

    // 4. Run CPU Baseline (Sparse Attention: SDDMM + Softmax + SpMM)
    std::cout << "\n=== SPARSE ATTENTION BASELINE ===" << std::endl;
    std::cout << "Running CPU Sparse Attention (SDDMM + Softmax + SpMM)..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    cpu_sparse_attention_csr(graph.num_nodes, feat_dim, h_Q.data(), h_K.data(), h_V.data(), 
                             graph.row_ptr.data(), graph.col_ind.data(), h_out_cpu.data());
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Sparse Attention Time: " << cpu_time.count() << " ms" << std::endl;

    // 5. Run Naive GPU Baseline (Sparse Attention)
    std::cout << "\nRunning GPU Naive Sparse Attention Kernel..." << std::endl;
    int threads_per_block = 256;
    int blocks_per_grid = (graph.num_nodes + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    // Warmup run
    f3s_naive_baseline_kernel<<<blocks_per_grid, threads_per_block>>>(d_Q, d_K, d_V, d_out_mat,
                                                                       d_row_ptr, d_col_idx,
                                                                       graph.num_nodes, feat_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_out_mat, 0, feat_size * sizeof(float))); // Reset for actual timed run

    CUDA_CHECK(cudaEventRecord(start_gpu));
    f3s_naive_baseline_kernel<<<blocks_per_grid, threads_per_block>>>(d_Q, d_K, d_V, d_out_mat,
                                                                       d_row_ptr, d_col_idx,
                                                                       graph.num_nodes, feat_dim);
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    float gpu_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu));
    std::cout << "GPU Naive Time: " << gpu_time_ms << " ms" << std::endl;

    // 6. Verify Results
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out_mat, feat_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_results(h_out_cpu, h_out_gpu)) {
        std::cout << "\nSUCCESS: GPU results match CPU baseline!" << std::endl;
        std::cout << "Speedup: " << cpu_time.count() / gpu_time_ms << "x" << std::endl;
    } else {
        std::cout << "\nFAILURE: GPU results do not match." << std::endl;
    }

    // 7. Cleanup
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out_mat));
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));

    return 0;
}