#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "frnn/csrc/grid/grid.h"

// --- Updated Kernel Declarations ---
extern "C" void run_insert_points(float3* d_points, int* d_grid_cnt, int* d_grid_cell, int* d_grid_idx, int P, GridParams params);
extern "C" void run_find_nbrs(float3* d_points1, float3* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, float radius, float* d_dists, int* d_idxs, GridParams params);
extern "C" void run_set_identity(int* d_idx, int P);

// Updated to match your prefix_sum.cu signature: (out, in, n)
extern "C" void run_prefix_sum(int* d_out, int* d_in, int n);

int main() {
    // 1. Configuration
    int P = 10000;          
    int K = 16;             
    float radius = 0.1f;    
    
    GridParams params;
    params.min_pos = make_float3(0.0f, 0.0f, 0.0f);
    params.delta = 0.05f;
    params.resolution = make_int3(20, 20, 20);
    params.total_cells = params.resolution.x * params.resolution.y * params.resolution.z;

    // 2. Memory Allocation
    float3 *d_points;
    int *d_grid_cnt, *d_grid_offsets, *d_grid_cell, *d_grid_idx, *d_identity_indices;
    float *d_dists;
    int *d_idxs;

    cudaMalloc(&d_points, P * sizeof(float3));
    cudaMalloc(&d_grid_cnt, params.total_cells * sizeof(int));
    // Offsets usually needs total_cells + 1 to store the final total
    cudaMalloc(&d_grid_offsets, (params.total_cells + 1) * sizeof(int));
    cudaMalloc(&d_grid_cell, P * sizeof(int));
    cudaMalloc(&d_grid_idx, P * sizeof(int));
    cudaMalloc(&d_identity_indices, P * sizeof(int)); 
    cudaMalloc(&d_dists, P * K * sizeof(float));
    cudaMalloc(&d_idxs, P * K * sizeof(int));

    // 3. Initialization
    run_set_identity(d_identity_indices, P);
    cudaMemset(d_grid_cnt, 0, params.total_cells * sizeof(int));

    // 4. Execution Pipeline
    std::cout << "--- Starting FRNN Pipeline ---" << std::endl;

    std::cout << "[1/3] Inserting points into grid..." << std::endl;
    run_insert_points(d_points, d_grid_cnt, d_grid_cell, d_grid_idx, P, params);

    std::cout << "[2/3] Running Prefix Sum (Scan)..." << std::endl;
    // Calling with (output, input, n) per your .cu file
    run_prefix_sum(d_grid_offsets, d_grid_cnt, params.total_cells);

    std::cout << "[3/3] Finding Neighbors..." << std::endl;
    run_find_nbrs(
        d_points, 
        d_points, 
        d_grid_offsets, 
        d_identity_indices, 
        P, K, radius, 
        d_dists, d_idxs, 
        params
    );

    // 5. Success Check & Cleanup
    std::cout << "SUCCESS: Pipeline completed on Perlmutter." << std::endl;
    
    cudaFree(d_points);
    cudaFree(d_grid_cnt);
    cudaFree(d_grid_offsets);
    cudaFree(d_grid_cell);
    cudaFree(d_grid_idx);
    cudaFree(d_identity_indices);
    cudaFree(d_dists);
    cudaFree(d_idxs);

    return 0;
}
