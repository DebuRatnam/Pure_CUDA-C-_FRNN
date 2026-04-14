#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "frnn_engine.h"
#include <iostream>

namespace py = pybind11;

// Link to your original CUDA kernels
extern "C" void run_insert_points(float3* d_points, int* d_grid_cnt, int* d_grid_cell, int* d_grid_idx, int P, GridParams params);
extern "C" void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao);
extern "C" void run_reorder_points(int* d_grid_cell, int* d_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, int P);
extern "C" void run_find_nbrs(float3* d_points1, float3* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, float radius, float* d_dists, int* d_idxs, GridParams params);

FRNNEngine::FRNNEngine(int max_points) : max_p(max_points) {
    // Allocation happens ONCE when the engine is initialized
    cudaMalloc(&d_points, max_p * sizeof(float3));
    cudaMalloc(&d_grid_cell, max_p * sizeof(int));
    cudaMalloc(&d_grid_idx, max_p * sizeof(int));
    cudaMalloc(&d_sorted_idxs, max_p * sizeof(int));
    
    // We assume a reasonable max resolution (e.g., 100^3) for the grid
    int max_cells = 1000000;
    cudaMalloc(&d_grid_cnt, max_cells * sizeof(int));
    cudaMalloc(&d_grid_offsets, (max_cells + 1) * sizeof(int));

    // Note: K is dynamic, so we allocate based on a safe upper bound (e.g., K=128)
    cudaMalloc(&d_dists, max_p * 128 * sizeof(float));
    cudaMalloc(&d_idxs, max_p * 128 * sizeof(int));
}

FRNNEngine::~FRNNEngine() {
    cudaFree(d_points); cudaFree(d_grid_cnt); cudaFree(d_grid_offsets);
    cudaFree(d_grid_cell); cudaFree(d_grid_idx); cudaFree(d_sorted_idxs);
    cudaFree(d_dists); cudaFree(d_idxs);
}

std::pair<std::vector<int>, std::vector<float>> FRNNEngine::search(std::vector<float> points_raw, int K, float radius) {
    int P = points_raw.size() / 3;
    if (P > max_p) throw std::runtime_error("Point count exceeds max_points allocated.");

    // 1. Host-to-Device Copy
    cudaMemcpy(d_points, points_raw.data(), P * sizeof(float3), cudaMemcpyHostToDevice);

    // 2. Setup GridParams
    GridParams params;
    params.min_pos = make_float3(0,0,0);
    params.delta = radius * 1.1f;
    int res = 100; 
    params.resolution = make_int3(res, res, res);
    params.total_cells = res * res * res;

    // 3. Execution Pipeline (The same logic as your benchmark)
    cudaMemset(d_grid_cnt, 0, params.total_cells * sizeof(int));
    run_insert_points(d_points, d_grid_cnt, d_grid_cell, d_grid_idx, P, params);
    scanLargeDeviceArray(d_grid_offsets, d_grid_cnt, params.total_cells, true);
    run_reorder_points(d_grid_cell, d_grid_idx, d_grid_offsets, d_sorted_idxs, P);
    run_find_nbrs(d_points, d_points, d_grid_offsets, d_sorted_idxs, P, K, radius, d_dists, d_idxs, params);

    // 4. Device-to-Host Copy
    std::vector<int> h_idxs(P * K);
    std::vector<float> h_dists(P * K);
    cudaMemcpy(h_idxs.data(), d_idxs, P * K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dists.data(), d_dists, P * K * sizeof(float), cudaMemcpyDeviceToHost);

    return {h_idxs, h_dists};
}

// --- PyBind11 Module Definition ---
PYBIND11_MODULE(frnn_cuda, m) {
    m.doc() = "FRNN CUDA search for LHC particle tracking";
    py::class_<FRNNEngine>(m, "FRNNEngine")
        .def(py::init<int>(), py::arg("max_points"))
        .def("search", &FRNNEngine::search, py::arg("points"), py::arg("K"), py::arg("radius"));
}