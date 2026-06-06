#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include "frnn/csrc/grid/grid.h"

// Updated externs to use float* and pass 'dim'
extern "C" void run_insert_points(float* d_points, int* d_grid_cnt, int* d_pc_grid_idx, int P, int dim, GridParams params);
extern "C" void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao);
extern "C" void run_reorder_points(int* d_pc_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, int P, int total_cells);
extern "C" void run_find_nbrs(float* d_points1, float* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, int dim, float radius, float* d_dists, int* d_idxs, GridParams params);

void benchmark_scale(int P, int dim) {
    int K = 16;
    float radius = 0.02f;
    int iterations = (P >= 1000000) ? 10 : 50;

    // --- PM Requirement: Dynamic Grid Scaling ---
    GridParams params;
    params.dim = dim;
    params.radius = radius;
    params.min_val = 0.0f;
    params.max_val = 1.0f;
    // Resolution is now based on radius to avoid sparseness
    params.cell_size = params.radius;   // ratio=1 (cell = radius, 3^dim shell)
    params.cell_radius = 1;
    params.res = (int)std::ceil((params.max_val - params.min_val) / params.cell_size);
    params.total_cells = std::pow(params.res, params.dim);

    float *d_points, *d_dists;
    int *d_grid_cnt, *d_grid_offsets, *d_pc_grid_idx, *d_sorted_idxs, *d_idxs;
    
    // Allocate based on N-Dimensions
    cudaMalloc(&d_points, P * dim * sizeof(float));
    cudaMalloc(&d_grid_cnt, params.total_cells * sizeof(int));
    cudaMalloc(&d_grid_offsets, (params.total_cells + 1) * sizeof(int));
    cudaMalloc(&d_pc_grid_idx, P * sizeof(int));
    cudaMalloc(&d_sorted_idxs, P * sizeof(int));
    cudaMalloc(&d_dists, P * K * sizeof(float));
    cudaMalloc(&d_idxs, P * K * sizeof(int));

    // Generate N-Dimensional random points
    std::vector<float> h_points(P * dim);
    for(int i = 0; i < P * dim; ++i) {
        h_points[i] = (float)rand() / RAND_MAX;
    }
    cudaMemcpy(d_points, h_points.data(), P * dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < iterations; i++) {
        cudaMemset(d_grid_cnt, 0, params.total_cells * sizeof(int));
        run_insert_points(d_points, d_grid_cnt, d_pc_grid_idx, P, dim, params);
        scanLargeDeviceArray(d_grid_offsets, d_grid_cnt, params.total_cells, true);
        // Note: run_reorder_points would also need an update for N-D internal logic
        run_find_nbrs(d_points, d_points, d_grid_offsets, d_sorted_idxs, P, K, dim, radius, d_dists, d_idxs, params);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "P = " << P << " | Dim = " << dim << " | Avg Time: " << (ms / iterations) << " ms" << std::endl;

    cudaFree(d_points); cudaFree(d_grid_cnt); cudaFree(d_grid_offsets);
    cudaFree(d_pc_grid_idx); cudaFree(d_sorted_idxs);
    cudaFree(d_dists); cudaFree(d_idxs);
}

int main() {
    srand(42);
    std::cout << "--- Starting N-Dimensional Scaling Benchmarks ---" << std::endl;
    
    // Testing both 3D (Spatial) and 8D (Latent Space)
    std::vector<int> dimensions = {3, 8};
    std::vector<int> scales = {10000, 100000, 1000000};

    for(int d : dimensions) {
        for(int p : scales) {
            benchmark_scale(p, d);
        }
    }
    return 0;
}