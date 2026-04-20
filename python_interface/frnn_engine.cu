#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "frnn_engine.h"
#include <iostream>
#include <cmath>

namespace py = pybind11;

// 1. Updated Externs: Matching the N-Dimensional Signatures
extern "C" void run_insert_points(float* d_points, int* d_grid_cnt, int* d_pc_grid_idx, int P, int dim, GridParams params);
extern "C" void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao);
extern "C" void run_reorder_points(int* d_pc_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, int P, int total_cells);
extern "C" void run_find_nbrs(float* d_points1, float* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, int dim, float radius, float* d_dists, int* d_idxs, GridParams params);

// Note: Constructor now takes max_points AND dim to allocate correctly
FRNNEngine::FRNNEngine(int max_points) : max_p(max_points) {
    // We assume a base dimension of 3 for allocation, or better yet, 
    // allocate for the largest expected ML latent space (e.g., 32D)
    int default_dim = 32; 
    cudaMalloc(&d_points, max_p * default_dim * sizeof(float));
    
    cudaMalloc(&d_grid_idx, max_p * sizeof(int));
    cudaMalloc(&d_sorted_idxs, max_p * sizeof(int));
    
    // Allocate for 1M cells (enough for 3D res=100 or 8D res=5)
    int max_cells = 1000000;
    cudaMalloc(&d_grid_cnt, max_cells * sizeof(int));
    cudaMalloc(&d_grid_offsets, (max_cells + 1) * sizeof(int));

    cudaMalloc(&d_dists, max_p * 128 * sizeof(float));
    cudaMalloc(&d_idxs, max_p * 128 * sizeof(int));
}

FRNNEngine::~FRNNEngine() {
    cudaFree(d_points); cudaFree(d_grid_cnt); cudaFree(d_grid_offsets);
    cudaFree(d_grid_idx); cudaFree(d_sorted_idxs);
    cudaFree(d_dists); cudaFree(d_idxs);
}

std::pair<std::vector<int>, std::vector<float>> FRNNEngine::search(std::vector<float> points_raw, int K, float radius) {
    // 1. Auto-detect Dimensions
    // If the user gave 300,000 values for 100,000 particles, dim = 3
    // If they gave 1,600,000 values for 100,000 particles, dim = 16
    int dim = points_raw.size() / max_p; 
    int P = max_p; 

    // 2. Host-to-Device Copy (N-Dimensional)
    cudaMemcpy(d_points, points_raw.data(), points_raw.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Setup Dynamic GridParams (PM Requirement)
    GridParams params;
    params.dim = dim;
    params.radius = radius;
    params.min_val = 0.0f;
    params.max_val = 1.0f;
    // Calculation avoids "sparseness"
    params.res = (int)std::ceil((params.max_val - params.min_val) / params.radius);
    params.total_cells = std::pow(params.res, params.dim);

    if(params.total_cells > 1000000) {
        throw std::runtime_error("Grid resolution too high for N-dimensions. Increase radius.");
    }

    // 4. Execution Pipeline
    cudaMemset(d_grid_cnt, 0, params.total_cells * sizeof(int));
    
    run_insert_points(d_points, d_grid_cnt, d_grid_idx, P, dim, params);
    scanLargeDeviceArray(d_grid_offsets, d_grid_cnt, params.total_cells, true);
    run_reorder_points(d_grid_idx, d_grid_offsets, d_sorted_idxs, P, params.total_cells);
    run_find_nbrs(d_points, d_points, d_grid_offsets, d_sorted_idxs, P, K, dim, radius, d_dists, d_idxs, params);

    // 5. Device-to-Host Copy
    std::vector<int> h_idxs(P * K);
    std::vector<float> h_dists(P * K);
    cudaMemcpy(h_idxs.data(), d_idxs, P * K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dists.data(), d_dists, P * K * sizeof(float), cudaMemcpyDeviceToHost);

    return {h_idxs, h_dists};
}

// --- PyBind11 Module Definition ---
PYBIND11_MODULE(frnn_cuda, m) {
    m.doc() = "N-Dimensional FRNN CUDA search for LHC Latent Spaces";
    py::class_<FRNNEngine>(m, "FRNNEngine")
        .def(py::init<int>(), py::arg("max_points"))
        .def("search", &FRNNEngine::search, py::arg("points"), py::arg("K"), py::arg("radius"));
}