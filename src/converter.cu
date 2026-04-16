#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include "../include/utils.cuh"
/**
 * Converts a CSR graph representation to a block-sparse format
 * @param num_nodes: Number of nodes in the graph
 * @param row_ptr: Row pointers of the CSR representation
 * @param col_ind: Column indices of the CSR representation
 * @return: The block-sparse format
 */
blockSparseFormat convert_csr_to_block_sparse(int num_nodes, const std::vector<int>& row_ptr, const std::vector<int>& col_ind) {
    blockSparseFormat bsf;
    bsf.num_row_windows = (num_nodes + 15) / 16; // Calculate the number of 16x16 blocks

    bsf.window_offsets.push_back(0); // Start with the first block offset

    for (int window_idx = 0; window_idx < bsf.num_row_windows; ++window_idx) {
        std::set<int> active_cols_in_window; // To store unique active column indices for the current window

        //scan all nodes within the current window
        int start_node = window_idx * 16;
        int end_node = std::min(start_node + 16, num_nodes);

        for (int u = start_node; u < end_node; ++u) {
            for (int j = row_ptr[u]; j < row_ptr[u + 1]; ++j) {
                int v = col_ind[j];
                active_cols_in_window.insert(v / 16); // map raw edge to a 16X16 block column index
            }
        }

        // Append the active column indices for the current window to the block_col_indices
        for (int block_col : active_cols_in_window) {
            bsf.block_col_indices.push_back(block_col);
        } 

        // update the window offset for the next block
        bsf.window_offsets.push_back(bsf.block_col_indices.size());
    }

    std::cout << "Row Windows: " << bsf.num_row_windows << std::endl;
    std::cout << "Total Active Blocks: " << bsf.block_col_indices.size() << std::endl;
    return bsf;
}