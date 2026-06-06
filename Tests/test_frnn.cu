#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "frnn/csrc/grid/grid.h"
#include "frnn/csrc/bruteforce/bruteforce.h"

// Updated externs
extern "C" void run_insert_points(float* d_points, int* d_grid_cnt, int* d_pc_grid_idx, int P, int dim, GridParams params);
extern "C" void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao);
extern "C" void run_reorder_points(int* d_pc_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, int P, int total_cells);
extern "C" void run_find_nbrs(float* d_points1, float* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, int dim, float radius, float* d_dists, int* d_idxs, GridParams params);
// Brute force also needs dim to calculate distances correctly
extern "C" void run_bruteforce(const float* d_p1, const float* d_p2, int P1, int P2, int K, int dim, float r, float* d_dists, int* d_idxs);

void run_random_trial(int trial_id, int P, int dim, float radius) {
    int K = 4;
    
    // PM Requirement: Dynamic Grid Scaling
    GridParams params;
    params.dim = dim;
    params.radius = radius;
    params.min_val = 0.0f;
    params.max_val = 1.0f;
    params.cell_size = params.radius;   // ratio=1 (cell = radius, 3^dim shell)
    params.cell_radius = 1;
    params.res = (int)std::ceil((params.max_val - params.min_val) / params.cell_size);
    params.total_cells = std::pow(params.res, params.dim);

    std::vector<float> h_points(P * dim);
    for(int i = 0; i < P * dim; ++i) {
        h_points[i] = (float)rand()/RAND_MAX;
    }

    float *d_points, *d_grid_dists, *d_bf_dists;
    int *d_grid_cnt, *d_grid_offsets, *d_pc_grid_idx, *d_sorted_idxs, *d_grid_idxs, *d_bf_idxs;

    cudaMalloc(&d_points, P * dim * sizeof(float));
    cudaMalloc(&d_grid_cnt, params.total_cells * sizeof(int));
    cudaMalloc(&d_grid_offsets, (params.total_cells + 1) * sizeof(int));
    cudaMalloc(&d_pc_grid_idx, P * sizeof(int));
    cudaMalloc(&d_sorted_idxs, P * sizeof(int));
    cudaMalloc(&d_grid_dists, P * K * sizeof(float));
    cudaMalloc(&d_grid_idxs, P * K * sizeof(int));
    cudaMalloc(&d_bf_dists, P * K * sizeof(float));
    cudaMalloc(&d_bf_idxs, P * K * sizeof(int));

    cudaMemcpy(d_points, h_points.data(), P * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_grid_cnt, 0, params.total_cells * sizeof(int));

    // Execution - Passing the new 'dim' parameter
    run_insert_points(d_points, d_grid_cnt, d_pc_grid_idx, P, dim, params);
    scanLargeDeviceArray(d_grid_offsets, d_grid_cnt, params.total_cells, true);
    run_reorder_points(d_pc_grid_idx, d_grid_offsets, d_sorted_idxs, P, params.total_cells);
    run_find_nbrs(d_points, d_points, d_grid_offsets, d_sorted_idxs, P, K, dim, radius, d_grid_dists, d_grid_idxs, params);
    run_bruteforce(d_points, d_points, P, P, K, dim, radius, d_bf_dists, d_bf_idxs);

    cudaDeviceSynchronize();

    std::vector<int> h_grid(P * K), h_bf(P * K);
    cudaMemcpy(h_grid.data(), d_grid_idxs, P * K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bf.data(), d_bf_idxs, P * K * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\n[Trial " << trial_id << " Verification (Dim: " << dim << ")]" << std::endl;
    
    // Verify first few points
    for(int i=0; i<2; i++) {
        std::vector<int> g_res, b_res;
        for(int k=0; k<K; k++) {
            g_res.push_back(h_grid[i*K+k]);
            b_res.push_back(h_bf[i*K+k]);
        }
        std::sort(g_res.begin(), g_res.end());
        std::sort(b_res.begin(), b_res.end());

        std::cout << "  Pt " << i << " Match: " << (g_res == b_res ? "YES" : "NO") << std::endl;
    }

    // Full comparison
    bool match = true;
    for (int i = 0; i < P; i++) {
        std::vector<int> g(h_grid.begin() + i*K, h_grid.begin() + (i+1)*K);
        std::vector<int> b(h_bf.begin() + i*K, h_bf.begin() + (i+1)*K);
        std::sort(g.begin(), g.end());
        std::sort(b.begin(), b.end());
        if (g != b) { match = false; break; }
    }

    std::cout << "OVERALL RESULT: " << (match ? "PASS" : "FAIL") << std::endl;

    cudaFree(d_points); cudaFree(d_grid_cnt); cudaFree(d_grid_offsets);
    cudaFree(d_pc_grid_idx); cudaFree(d_sorted_idxs);
    cudaFree(d_grid_dists); cudaFree(d_grid_idxs); cudaFree(d_bf_dists); cudaFree(d_bf_idxs);
}

int main() {
    srand(time(NULL));
    std::cout << "--- Starting Randomized N-Dimensional Accuracy Tests ---" << std::endl;
    
    // Test 3D and 8D accuracy
    int dims[] = {3, 8};
    for(int d : dims) {
        for(int i=1; i<=2; i++) {
            run_random_trial(i, 1000, d, 0.15f);
        }
    }
    return 0;
}