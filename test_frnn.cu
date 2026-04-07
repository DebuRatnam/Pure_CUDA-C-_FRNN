#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm> // Added for sorting
#include <cuda_runtime.h>
#include "frnn/csrc/grid/grid.h"
#include "frnn/csrc/bruteforce/bruteforce.h"

extern "C" void run_insert_points(float3* d_points, int* d_grid_cnt, int* d_grid_cell, int* d_grid_idx, int P, GridParams params);
extern "C" void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao);
extern "C" void run_reorder_points(int* d_grid_cell, int* d_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, int P);
extern "C" void run_find_nbrs(float3* d_points1, float3* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, float radius, float* d_dists, int* d_idxs, GridParams params);
extern "C" void run_bruteforce(const float3* d_p1, const float3* d_p2, int P1, int P2, int K, float r, float* d_dists, int* d_idxs);

void run_random_trial(int trial_id, int P, float radius) {
    int K = 4;
    GridParams params;
    params.min_pos = make_float3(0.0f, 0.0f, 0.0f);
    params.delta = radius * 1.1f; 
    params.resolution = make_int3(20, 20, 20);
    params.total_cells = 8000;

    std::vector<float3> h_points(P);
    for(int i = 0; i < P; ++i) {
        h_points[i] = make_float3((float)rand()/RAND_MAX, (float)rand()/RAND_MAX, (float)rand()/RAND_MAX);
    }

    float3 *d_points;
    int *d_grid_cnt, *d_grid_offsets, *d_grid_cell, *d_grid_idx, *d_sorted_idxs, *d_grid_idxs, *d_bf_idxs;
    float *d_grid_dists, *d_bf_dists;

    cudaMalloc(&d_points, P * sizeof(float3));
    cudaMalloc(&d_grid_cnt, params.total_cells * sizeof(int));
    cudaMalloc(&d_grid_offsets, (params.total_cells + 1) * sizeof(int));
    cudaMalloc(&d_grid_cell, P * sizeof(int));
    cudaMalloc(&d_grid_idx, P * sizeof(int));
    cudaMalloc(&d_sorted_idxs, P * sizeof(int));
    cudaMalloc(&d_grid_dists, P * K * sizeof(float));
    cudaMalloc(&d_grid_idxs, P * K * sizeof(int));
    cudaMalloc(&d_bf_dists, P * K * sizeof(float));
    cudaMalloc(&d_bf_idxs, P * K * sizeof(int));

    cudaMemcpy(d_points, h_points.data(), P * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemset(d_grid_cnt, 0, params.total_cells * sizeof(int));

    // Execution
    run_insert_points(d_points, d_grid_cnt, d_grid_cell, d_grid_idx, P, params);
    scanLargeDeviceArray(d_grid_offsets, d_grid_cnt, params.total_cells, true);
    run_reorder_points(d_grid_cell, d_grid_idx, d_grid_offsets, d_sorted_idxs, P);
    run_find_nbrs(d_points, d_points, d_grid_offsets, d_sorted_idxs, P, K, radius, d_grid_dists, d_grid_idxs, params);
    run_bruteforce(d_points, d_points, P, P, K, radius, d_bf_dists, d_bf_idxs);

    // Sync to ensure A100 is done
    cudaDeviceSynchronize();

    std::vector<int> h_grid(P * K), h_bf(P * K);
    cudaMemcpy(h_grid.data(), d_grid_idxs, P * K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bf.data(), d_bf_idxs, P * K * sizeof(int), cudaMemcpyDeviceToHost);

    // --- DEBUG PRINT (First 2 points) ---
    std::cout << "\n[Trial " << trial_id << " Verification]" << std::endl;
    for(int i=0; i<2; i++) {
        std::cout << "  Pt " << i << " Grid: ";
        for(int k=0; k<K; k++) std::cout << h_grid[i*K+k] << " ";
        std::cout << "\n  Pt " << i << " BF:   ";
        for(int k=0; k<K; k++) std::cout << h_bf[i*K+k] << " ";
        std::cout << std::endl;
    }

    // --- SORTING FOR COMPARISON ---
    for (int i = 0; i < P; i++) {
        std::sort(h_grid.begin() + i*K, h_grid.begin() + (i+1)*K);
        std::sort(h_bf.begin() + i*K, h_bf.begin() + (i+1)*K);
    }

    bool match = true;
    for(int i=0; i<P*K; i++) {
        if(h_grid[i] != h_bf[i]) {
            match = false;
            break;
        }
    }

    std::cout << "RESULT: " << (match ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    cudaFree(d_points); cudaFree(d_grid_cnt); cudaFree(d_grid_offsets);
    cudaFree(d_grid_cell); cudaFree(d_grid_idx); cudaFree(d_sorted_idxs);
    cudaFree(d_grid_dists); cudaFree(d_grid_idxs); cudaFree(d_bf_dists); cudaFree(d_bf_idxs);
}

int main() {
    srand(time(NULL));
    std::cout << "--- Starting Randomized Accuracy Tests ---" << std::endl;
    for(int i=1; i<=5; i++) {
        int random_P = 1000;
        float random_R = 0.15f; 
        run_random_trial(i, random_P, random_R);
    }
    return 0;
}