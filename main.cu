#include <iostream>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>
#include "frnn/csrc/grid/grid.h"

// --- THE CONTRACTS ---
extern "C" void run_insert_points(float3* d_points, int* d_grid_cnt, int* d_grid_cell, int* d_grid_idx, int P, GridParams params);

// UPDATED: Now matching the Blelloch scan.cu signature
extern "C" void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao);

extern "C" void run_reorder_points(int* d_grid_cell, int* d_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, int P);
extern "C" void run_find_nbrs(float3* d_points1, float3* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, float radius, float* d_dists, int* d_idxs, GridParams params);

int main() {
    // 1. Configuration
    int P = 1000; 
    int K = 4;
    float radius = 0.2f;
    srand(time(NULL));
    
    GridParams params;
    params.min_pos = make_float3(0.0f, 0.0f, 0.0f);
    params.delta = 0.1f;
    params.resolution = make_int3(10, 10, 10);
    params.total_cells = 10 * 10 * 10;

    // 2. Prepare Host Data (CPU)
    std::vector<float3> h_points(P);
    for(int i = 0; i < P; ++i) {
        h_points[i] = make_float3((float)rand()/RAND_MAX, (float)rand()/RAND_MAX, (float)rand()/RAND_MAX);
    }

    // 3. Allocate Device Memory (GPU)
    float3 *d_points;
    int *d_grid_cnt, *d_grid_offsets, *d_grid_cell, *d_grid_idx, *d_sorted_idxs;
    float *d_dists;
    int *d_idxs;

    cudaMalloc(&d_points, P * sizeof(float3));
    cudaMalloc(&d_grid_cnt, params.total_cells * sizeof(int));
    cudaMalloc(&d_grid_offsets, (params.total_cells + 1) * sizeof(int));
    cudaMalloc(&d_grid_cell, P * sizeof(int));
    cudaMalloc(&d_grid_idx, P * sizeof(int));
    cudaMalloc(&d_sorted_idxs, P * sizeof(int));
    cudaMalloc(&d_dists, P * K * sizeof(float));
    cudaMalloc(&d_idxs, P * K * sizeof(int));

    // 4. Upload data to GPU
    cudaMemcpy(d_points, h_points.data(), P * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemset(d_grid_cnt, 0, params.total_cells * sizeof(int));

    // 5. THE EXECUTION PIPELINE
    std::cout << "Starting FRNN Pipeline..." << std::endl;

    // A. Insert points into grid cells
    run_insert_points(d_points, d_grid_cnt, d_grid_cell, d_grid_idx, P, params);
    
    // B. UPDATED: Blelloch Prefix Sum (Exclusive Scan)
    // true = enable Bank Conflict Avoidance Optimization
    scanLargeDeviceArray(d_grid_offsets, d_grid_cnt, params.total_cells, true);
    
    // C. Reorder points for contiguous memory access
    run_reorder_points(d_grid_cell, d_grid_idx, d_grid_offsets, d_sorted_idxs, P);
    
    // D. Final Neighbor Search
    run_find_nbrs(d_points, d_points, d_grid_offsets, d_sorted_idxs, P, K, radius, d_dists, d_idxs, params);

    // 6. Download Results and Verify
    std::vector<int> h_idxs(P * K);
    cudaMemcpy(h_idxs.data(), d_idxs, P * K * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Pipeline Finished!" << std::endl;
    std::cout << "Point 0's first neighbor ID: " << h_idxs[0] << std::endl;

    // 7. Cleanup
    cudaFree(d_points);
    cudaFree(d_grid_cnt);
    cudaFree(d_grid_offsets);
    cudaFree(d_grid_cell);
    cudaFree(d_grid_idx);
    cudaFree(d_sorted_idxs);
    cudaFree(d_dists);
    cudaFree(d_idxs);

    return 0;
}