#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "frnn_engine.h"
#include <iostream>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace py = pybind11;

// 1. Updated Externs: Matching the N-Dimensional Signatures
extern "C" void run_insert_points(float* d_points, int* d_grid_cnt, int* d_pc_grid_idx, int P, int dim, GridParams params);
extern "C" void run_reorder_points(int* d_pc_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, int P, int total_cells);
extern "C" void run_counting_sort(float* d_points, int* d_pc_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, float* d_points_sorted, int P, int dim);
extern "C" void run_find_nbrs(float* d_points1, float* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, int dim, float radius, float* d_dists, int* d_idxs, GridParams params);
extern "C" void run_bruteforce(const float* d_p1, const float* d_p2, int P1, int P2, int K, int dim, float r, float* d_dists, int* d_idxs);
// D=3 sorted-query / AoS grid path (see find_nbrs.cu, insert_points.cu)
extern "C" void run_counting_sort_aos3(float* d_points, int* d_pc_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, float* d_points_sorted, int P);
extern "C" void run_find_nbrs_aos3(float* d_points1_aos, float* d_points2_aos, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, float radius, float* d_dists, int* d_idxs, GridParams params);
extern "C" void run_scatter_to_orig(const float* d_dists_in, const int* d_idxs_in, const int* d_sorted_idxs, int P, int K, float* d_dists_out, int* d_idxs_out);

// Note: Constructor now takes max_points AND dim to allocate correctly
FRNNEngine::FRNNEngine(int max_points) : max_p(max_points) {
    // We assume a base dimension of 3 for allocation, or better yet, 
    // allocate for the largest expected ML latent space (e.g., 32D)
    int default_dim = 32;
    cudaMalloc(&d_points, max_p * default_dim * sizeof(float));
    cudaMalloc(&d_points_sorted, max_p * default_dim * sizeof(float));
    cudaMalloc(&d_points_sorted_aos, max_p * 3 * sizeof(float));  // D=3 AoS cell-order buffer

    cudaMalloc(&d_grid_idx, max_p * sizeof(int));
    cudaMalloc(&d_sorted_idxs, max_p * sizeof(int));

    // Allocate for 1M cells (enough for 3D res=100 or 8D res=5)
    int max_cells = 1000000;
    cudaMalloc(&d_grid_cnt, max_cells * sizeof(int));
    cudaMalloc(&d_grid_offsets, (max_cells + 1) * sizeof(int));

    cudaMalloc(&d_dists, max_p * 128 * sizeof(float));
    cudaMalloc(&d_idxs, max_p * 128 * sizeof(int));
    // Sorted-query output, unpermuted by run_scatter_to_orig into d_dists/d_idxs.
    cudaMalloc(&d_dists_sorted, max_p * 128 * sizeof(float));
    cudaMalloc(&d_idxs_sorted, max_p * 128 * sizeof(int));
}

FRNNEngine::~FRNNEngine() {
    cudaFree(d_points); cudaFree(d_points_sorted); cudaFree(d_points_sorted_aos);
    cudaFree(d_grid_cnt); cudaFree(d_grid_offsets);
    cudaFree(d_grid_idx); cudaFree(d_sorted_idxs);
    cudaFree(d_dists); cudaFree(d_idxs);
    cudaFree(d_dists_sorted); cudaFree(d_idxs_sorted);
}

std::pair<std::vector<int>, std::vector<float>> FRNNEngine::search(std::vector<float> points_raw, int K, float radius) {
    // 1. Auto-detect Dimensions
    // If the user gave 300,000 values for 100,000 particles, dim = 3
    // If they gave 1,600,000 values for 100,000 particles, dim = 16
    int dim = points_raw.size() / max_p; 
    int P = max_p; 

    // 2. Transpose AoS → SoA then copy H2D.
    // Kernels use p[d*P+i] (SoA) for coalesced warp access; Python callers still pass row-major AoS.
    std::vector<float> pts_soa(points_raw.size());
    for (int d = 0; d < dim; d++)
        for (int p = 0; p < P; p++)
            pts_soa[d * P + p] = points_raw[p * dim + d];
    cudaMemcpy(d_points, pts_soa.data(), pts_soa.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Setup Dynamic GridParams (PM Requirement)
    GridParams params;
    params.dim = dim;
    params.radius = radius;
    params.min_val = 0.0f;
    params.max_val = 1.0f;
    // Calculation avoids "sparseness"
    params.cell_size = params.radius;   // legacy ratio=1 (cell = radius, 3^dim shell)
    params.cell_radius = 1;
    params.res = (int)std::ceil((params.max_val - params.min_val) / params.cell_size);
    // Cell count in double to avoid long-long overflow when res^dim is huge (small
    // radius at high D). If the grid is infeasible (>1M cells) fall back to brute
    // force instead of failing: BF needs no grid and is correct for any radius. This
    // matches the documented auto-dispatch ("BF engages automatically for high-D").
    double total_cells_d = std::pow((double)params.res, params.dim);
    bool grid_feasible = (total_cells_d <= 1000000.0);
    params.total_cells = grid_feasible ? (long long)total_cells_d : 0;

    // 4. Execution Pipeline
    // The grid kernel checks 3^D neighboring cells per point.  When that shell
    // covers >= the entire grid (3^D >= total_cells), the per-cell overhead
    // dominates and brute-force is strictly faster.  For D=16 with res=2,
    // 3^16=43M >> 2^16=65K, so the grid is catastrophically slow there.
    // res<=1 also forces brute-force because the prefix-sum breaks on one cell.
    long long neighbor_shell = 1;
    for (int d = 0; d < dim; d++) neighbor_shell *= 3;

    if (params.res <= 1 || !grid_feasible || neighbor_shell >= (long long)params.total_cells) {
        run_bruteforce(d_points, d_points, P, P, K, dim, radius, d_dists, d_idxs);
    } else {
        cudaMemset(d_grid_cnt, 0, params.total_cells * sizeof(int));
        run_insert_points(d_points, d_grid_cnt, d_grid_idx, P, dim, params);
        thrust::exclusive_scan(
            thrust::device_ptr<int>(d_grid_cnt),
            thrust::device_ptr<int>(d_grid_cnt + params.total_cells),
            thrust::device_ptr<int>(d_grid_offsets));
        if (dim == 3) {
            // D=3 large-N fast path: cell-ordered (sorted) queries + AoS candidates so a
            // warp scans each candidate span in lockstep from one cache line. find_nbrs_aos3
            // writes coalesced in sorted order; scatter unpermutes into d_dists/d_idxs.
            run_counting_sort_aos3(d_points, d_grid_idx, d_grid_offsets, d_sorted_idxs, d_points_sorted_aos, P);
            run_find_nbrs_aos3(d_points_sorted_aos, d_points_sorted_aos, d_grid_offsets, d_sorted_idxs, P, K, radius, d_dists_sorted, d_idxs_sorted, params);
            run_scatter_to_orig(d_dists_sorted, d_idxs_sorted, d_sorted_idxs, P, K, d_dists, d_idxs);
        } else {
            run_counting_sort(d_points, d_grid_idx, d_grid_offsets, d_sorted_idxs, d_points_sorted, P, dim);
            run_find_nbrs(d_points, d_points_sorted, d_grid_offsets, d_sorted_idxs, P, K, dim, radius, d_dists, d_idxs, params);
        }
    }

    // 5. Device-to-Host Copy (GPU output is SoA: d_idxs[k*P+p], d_dists[k*P+p])
    std::vector<int>   h_idxs(P * K);
    std::vector<float> h_dists(P * K);
    cudaMemcpy(h_idxs.data(),  d_idxs,  P * K * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dists.data(), d_dists, P * K * sizeof(float),  cudaMemcpyDeviceToHost);

    // Untranspose SoA → AoS so Python callers receive row-major [p*K+k] layout
    std::vector<int>   idxs_out(P * K);
    std::vector<float> dists_out(P * K);
    for (int k = 0; k < K; k++)
        for (int p = 0; p < P; p++) {
            idxs_out[p * K + k]  = h_idxs[k * P + p];
            dists_out[p * K + k] = h_dists[k * P + p];
        }
    return {idxs_out, dists_out};
}

std::pair<uintptr_t, uintptr_t> FRNNEngine::search_gpu(
    uintptr_t dev_ptr, int N, int dim, int K, float radius)
{
    float* d_input = reinterpret_cast<float*>(dev_ptr);
    int P = N;

    GridParams params;
    params.dim        = dim;
    params.radius     = radius;
    params.min_val    = 0.0f;
    params.max_val    = 1.0f;
    params.cell_size   = params.radius;   // legacy ratio=1 (cell = radius, 3^dim shell)
    params.cell_radius = 1;
    params.res         = (int)std::ceil((params.max_val - params.min_val) / params.cell_size);
    // See search(): infeasible grid (>1M cells) falls back to brute force instead of
    // throwing. double avoids long-long overflow of res^dim at small radius / high D.
    double total_cells_d = std::pow((double)params.res, params.dim);
    bool grid_feasible = (total_cells_d <= 1000000.0);
    params.total_cells = grid_feasible ? (long long)total_cells_d : 0;

    long long neighbor_shell = 1;
    for (int d = 0; d < dim; d++) neighbor_shell *= 3;

    if (params.res <= 1 || !grid_feasible || neighbor_shell >= (long long)params.total_cells) {
        run_bruteforce(d_input, d_input, P, P, K, dim, radius, d_dists, d_idxs);
    } else {
        cudaMemset(d_grid_cnt, 0, params.total_cells * sizeof(int));
        run_insert_points(d_input, d_grid_cnt, d_grid_idx, P, dim, params);
        thrust::exclusive_scan(
            thrust::device_ptr<int>(d_grid_cnt),
            thrust::device_ptr<int>(d_grid_cnt + params.total_cells),
            thrust::device_ptr<int>(d_grid_offsets));
        if (dim == 3) {
            run_counting_sort_aos3(d_input, d_grid_idx, d_grid_offsets, d_sorted_idxs, d_points_sorted_aos, P);
            run_find_nbrs_aos3(d_points_sorted_aos, d_points_sorted_aos, d_grid_offsets, d_sorted_idxs, P, K, radius, d_dists_sorted, d_idxs_sorted, params);
            run_scatter_to_orig(d_dists_sorted, d_idxs_sorted, d_sorted_idxs, P, K, d_dists, d_idxs);
        } else {
            run_counting_sort(d_input, d_grid_idx, d_grid_offsets, d_sorted_idxs, d_points_sorted, P, dim);
            run_find_nbrs(d_input, d_points_sorted, d_grid_offsets, d_sorted_idxs,
                          P, K, dim, radius, d_dists, d_idxs, params);
        }
    }

    return {reinterpret_cast<uintptr_t>(d_idxs),
            reinterpret_cast<uintptr_t>(d_dists)};
}

std::pair<std::vector<int>, std::vector<float>> FRNNEngine::get_results(int N, int K) {
    std::vector<int>   h_idxs(N * K);
    std::vector<float> h_dists(N * K);
    cudaMemcpy(h_idxs.data(),  d_idxs,  N * K * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dists.data(), d_dists, N * K * sizeof(float),  cudaMemcpyDeviceToHost);
    return {h_idxs, h_dists};
}

// --- PyBind11 Module Definition ---
PYBIND11_MODULE(frnn_cuda, m) {
    m.doc() = "N-Dimensional FRNN CUDA search for LHC Latent Spaces";
    py::class_<FRNNEngine>(m, "FRNNEngine")
        .def(py::init<int>(), py::arg("max_points"))
        .def("search", &FRNNEngine::search,
             py::arg("points"), py::arg("K"), py::arg("radius"))
        .def("search_gpu", &FRNNEngine::search_gpu,
             py::arg("dev_ptr"), py::arg("N"), py::arg("dim"),
             py::arg("K"), py::arg("radius"))
        .def("get_results", &FRNNEngine::get_results,
             py::arg("N"), py::arg("K"));
}