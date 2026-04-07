#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm> // Added for random fill
#include "frnn/csrc/grid/grid.h"

extern "C" void run_insert_points(float3* d_points, int* d_grid_cnt, int* d_grid_cell, int* d_grid_idx, int P, GridParams params);
extern "C" void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao);
extern "C" void run_reorder_points(int* d_grid_cell, int* d_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, int P);
extern "C" void run_find_nbrs(float3* d_points1, float3* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, float radius, float* d_dists, int* d_idxs, GridParams params);

void benchmark_scale(int P) {
    int K = 16;
    float radius = 0.02f;
    int iterations = (P >= 1000000) ? 10 : 50; // Reduction for 1M to save time

    GridParams params;
    params.min_pos = make_float3(0,0,0);
    params.delta = radius * 1.1f; 
    params.resolution = make_int3(100, 100, 100);
    params.total_cells = 1000000;

    float3 *d_points;
    int *d_grid_cnt, *d_grid_offsets, *d_grid_cell, *d_grid_idx, *d_sorted_idxs;
    float *d_dists; int *d_idxs;
    
    cudaMalloc(&d_points, P * sizeof(float3));
    cudaMalloc(&d_grid_cnt, params.total_cells * sizeof(int));
    cudaMalloc(&d_grid_offsets, (params.total_cells + 1) * sizeof(int));
    cudaMalloc(&d_grid_cell, P * sizeof(int));
    cudaMalloc(&d_grid_idx, P * sizeof(int));
    cudaMalloc(&d_sorted_idxs, P * sizeof(int));
    cudaMalloc(&d_dists, P * K * sizeof(float));
    cudaMalloc(&d_idxs, P * K * sizeof(int));

    // --- NEW: Generate distributed random points instead of all zeros ---
    std::vector<float3> h_points(P);
    for(int i = 0; i < P; ++i) {
        h_points[i] = make_float3(
            (float)rand() / RAND_MAX, 
            (float)rand() / RAND_MAX, 
            (float)rand() / RAND_MAX
        );
    }
    cudaMemcpy(d_points, h_points.data(), P * sizeof(float3), cudaMemcpyHostToDevice);
    // --------------------------------------------------------------------

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < iterations; i++) {
        cudaMemset(d_grid_cnt, 0, params.total_cells * sizeof(int));
        run_insert_points(d_points, d_grid_cnt, d_grid_cell, d_grid_idx, P, params);
        scanLargeDeviceArray(d_grid_offsets, d_grid_cnt, params.total_cells, true);
        run_reorder_points(d_grid_cell, d_grid_idx, d_grid_offsets, d_sorted_idxs, P);
        run_find_nbrs(d_points, d_points, d_grid_offsets, d_sorted_idxs, P, K, radius, d_dists, d_idxs, params);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "P = " << P << " | Avg Time: " << (ms / iterations) << " ms" << std::endl;

    cudaFree(d_points); cudaFree(d_grid_cnt); cudaFree(d_grid_offsets);
    cudaFree(d_grid_cell); cudaFree(d_grid_idx); cudaFree(d_sorted_idxs);
    cudaFree(d_dists); cudaFree(d_idxs);
}

int main() {
    srand(42); // Ensure reproducible results for your report
    std::cout << "--- Starting Scaling Benchmarks ---" << std::endl;
    std::vector<int> scales = {10000, 50000, 100000, 200000, 1000000};
    for(int p : scales) {
        benchmark_scale(p);
    }
    return 0;
}