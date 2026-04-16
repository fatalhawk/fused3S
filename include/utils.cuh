#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << "code=" << err << ": " \
                      << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct GraphData {
    int num_nodes;
    int num_edges;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
};

inline GraphData load_csr_graph(const std::string& dataset_dir) {
    GraphData graph;
    std::ifstream meta(dataset_dir + "/meta.txt");
    meta >> graph.num_nodes >> graph.num_edges;

    graph.row_ptr.resize(graph.num_nodes + 1);
    std::ifstream row_file(dataset_dir + "/row_ptr.txt");
    for (int i = 0; i <= graph.num_nodes; ++i) row_file >> graph.row_ptr[i];

    graph.col_ind.resize(graph.num_edges);
    std::ifstream col_file(dataset_dir + "/col_ind.txt");
    for (int i = 0; i < graph.num_edges; ++i) col_file >> graph.col_ind[i];

    return graph;
}

struct blockSparseFormat {
    int num_row_windows;
    std::vector<int> window_offsets;     // points to the start of each 16X16 block
    std::vector<int> block_col_indices;  //the col indices of the active 16X16 blocks
};

inline std::vector<int> generate_balanced_window_order(const blockSparseFormat& format) {
    std::vector<int> window_order(format.num_row_windows);
    std::iota(window_order.begin(), window_order.end(), 0); // Fill with 0, 1, 2...

    // Sort window indices descending based on the number of active blocks they contain
    std::sort(window_order.begin(), window_order.end(), [&format](int a, int b) {
        int blocks_a = format.window_offsets[a + 1] - format.window_offsets[a];
        int blocks_b = format.window_offsets[b + 1] - format.window_offsets[b];
        return blocks_a > blocks_b; 
    });

    return window_order;
}