#include "frnn_engine.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

// 1. Updated Externs: Matching the N-Dimensional Signatures
extern "C" void run_insert_points(float* d_points, int* d_grid_cnt, int* d_pc_grid_idx, int P, int dim, GridParams params);
extern "C" void run_reorder_points(int* d_pc_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, int P, int total_cells);
extern "C" void run_counting_sort(float* d_points, int* d_pc_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, float* d_points_sorted, int P, int dim);
extern "C" void run_find_nbrs(float* d_points1, float* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, int dim, float radius, float* d_dists, int* d_idxs, GridParams params);
extern "C" void run_find_nbrs_griddim(float* d_points1, float* d_points2, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, int full_dim, int grid_dim, float radius, float* d_dists, int* d_idxs, GridParams params);
// D=3 sorted-query / AoS grid path (see find_nbrs.cu, insert_points.cu)
extern "C" void run_counting_sort_aos3(float* d_points, int* d_pc_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, float* d_points_sorted, int P);
extern "C" void run_find_nbrs_aos3(float* d_points1_aos, float* d_points2_aos, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, float radius, float* d_dists, int* d_idxs, GridParams params);
extern "C" void run_scatter_to_orig(const float* d_dists_in, const int* d_idxs_in, const int* d_sorted_idxs, int P, int K, float* d_dists_out, int* d_idxs_out);
// Verify path (verify.cu). AoS in/out; see that file for layout.
extern "C" void run_verify_candidates(const float* d_pts, const int* d_cand, int N, int D,
                                      int O, int K, float r, float* d_out_d, int* d_out_i);
extern "C" void run_bruteforce(const float* d_p1, const float* d_p2,
                               int P1, int P2, int K, int dim, float r,
                               float* d_dists, int* d_idxs);

// Element-wise transposes bridging the engine's SoA layout and the projection
// stages' AoS layout. Treat src as (rows x cols) row-major; write dst as
// (cols x rows) row-major: dst[c*rows + r] = src[r*cols + c].
__global__ void transpose_f(const float* __restrict__ src, float* __restrict__ dst, int rows, int cols) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * cols;
    if (idx >= total) return;
    int r = (int)(idx / cols), c = (int)(idx % cols);
    dst[(long long)c * rows + r] = src[idx];
}
__global__ void transpose_i(const int* __restrict__ src, int* __restrict__ dst, int rows, int cols) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * cols;
    if (idx >= total) return;
    int r = (int)(idx / cols), c = (int)(idx % cols);
    dst[(long long)c * rows + r] = src[idx];
}

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
    // Sorted-query output; scatter_to_orig unpermutes these into d_dists/d_idxs.
    cudaMalloc(&d_dists_sorted, max_p * 128 * sizeof(float));
    cudaMalloc(&d_idxs_sorted, max_p * 128 * sizeof(int));

    // First-d grid path scratch: AoS staging for verify, first-d SoA for grid, candidates.
    cudaMalloc(&d_pts_aos,   max_p * default_dim * sizeof(float));
    cudaMalloc(&d_proj_soa,  max_p * GRID_DIM_MAX * sizeof(float));
    cudaMalloc(&d_cand,      max_p * 128 * sizeof(int));  // O up to 128
}

FRNNEngine::~FRNNEngine() {
    cudaFree(d_points); cudaFree(d_points_sorted); cudaFree(d_points_sorted_aos);
    cudaFree(d_grid_cnt); cudaFree(d_grid_offsets);
    cudaFree(d_grid_idx); cudaFree(d_sorted_idxs);
    cudaFree(d_dists); cudaFree(d_idxs);
    cudaFree(d_dists_sorted); cudaFree(d_idxs_sorted);
    cudaFree(d_pts_aos); cudaFree(d_proj_soa); cudaFree(d_cand);
}

// First-d grid with inline full-D ranking (libFRNN-style).
//   grid_dim = (D>4)?4:min(D,3)  — matches libFRNN gridDimensionCount().
//   1. Copy first grid_dim dimensions from SoA input into d_proj_soa (SoA, no alloc).
// Every point encountered by the grid scan is compared in full D and inserted
// directly into the true K-nearest heap; there is no bounded candidate stage.
// Final results land in d_dists/d_idxs (SoA [k*N+q], original ids). Returns false if
// the first-d grid is degenerate so the caller can fall back to brute-force.
bool FRNNEngine::run_firstd_search(const float* d_in_soa, int N, int D, int K, float radius) {
    const int grid_dim = (D > 4) ? 4 : std::min(D, 3);
    // Extract first grid_dim dimensions: SoA layout means the first grid_dim*N floats
    // of d_in_soa are exactly dimensions 0..grid_dim-1 for all N points.
    cudaMemcpy(d_proj_soa, d_in_soa, (size_t)grid_dim * N * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Grid params on the first grid_dim dimensions (data normalized to [0,1]).
    GridParams pp;
    pp.dim        = grid_dim;
    pp.radius     = radius;
    pp.min_val    = 0.0f;
    pp.max_val    = 1.0f;
    pp.cell_size  = radius;
    pp.cell_radius = 1;
    pp.res = (int)std::ceil((pp.max_val - pp.min_val) / pp.cell_size);
    if (pp.res < 2) pp.res = 2;
    double tc = std::pow((double)pp.res, grid_dim);
    if (tc > 1000000.0) return false;
    pp.total_cells = (long long)tc;
    long long shell = 1;
    for (int d = 0; d < grid_dim; d++) shell *= 3;
    if (shell >= (long long)pp.total_cells) return false;

    // Sort all D coordinates using cell ids formed from only the first grid_dim.
    cudaMemset(d_grid_cnt, 0, pp.total_cells * sizeof(int));
    run_insert_points(d_proj_soa, d_grid_cnt, d_grid_idx, N, grid_dim, pp);
    thrust::exclusive_scan(
        thrust::device_ptr<int>(d_grid_cnt),
        thrust::device_ptr<int>(d_grid_cnt + pp.total_cells),
        thrust::device_ptr<int>(d_grid_offsets));
    run_counting_sort(const_cast<float*>(d_in_soa), d_grid_idx, d_grid_offsets,
                      d_sorted_idxs, d_points_sorted, N, D);
    run_find_nbrs_griddim(const_cast<float*>(d_in_soa), d_points_sorted,
                          d_grid_offsets, d_sorted_idxs, N, K, D, grid_dim,
                          radius, d_dists, d_idxs, pp);
    cudaDeviceSynchronize();
    return true;
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
    // Dispatch mirrors libFRNN resolveAlgorithm():
    //   selection_factor = max(1, (K+7)/8); database_dimensions = N*D if N*D<=2M else 2M+1
    //   brute-force when database_dimensions * selection_factor <= 2M AND N fits.
    // Otherwise: full-D grid if feasible, else first-d grid + verify (libFRNN-style).
    constexpr long long kBF = 2000000LL;
    long long sel = std::max(1LL, (long long)(K + 7) / 8);
    long long db_dims = ((long long)P * dim > kBF) ? (kBF + 1LL) : (long long)P * dim;
    bool use_bf = (db_dims <= kBF / sel) &&
                  ((long long)P <= kBF / (db_dims * sel));

    long long neighbor_shell = 1;
    for (int d = 0; d < dim; d++) neighbor_shell *= 3;

    if (use_bf) {
        run_bruteforce(d_points, d_points, P, P, K, dim, radius, d_dists, d_idxs);
    } else if (params.res <= 1 || !grid_feasible || neighbor_shell >= (long long)params.total_cells) {
        // Full-D grid infeasible: first-d grid + full-D verify (libFRNN-style).
        if (!run_firstd_search(d_points, P, dim, K, radius)) {
            run_bruteforce(d_points, d_points, P, P, K, dim, radius, d_dists, d_idxs);
        }
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

    constexpr long long kBF = 2000000LL;
    long long sel = std::max(1LL, (long long)(K + 7) / 8);
    long long db_dims = ((long long)P * dim > kBF) ? (kBF + 1LL) : (long long)P * dim;
    bool use_bf = (db_dims <= kBF / sel) &&
                  ((long long)P <= kBF / (db_dims * sel));

    long long neighbor_shell = 1;
    for (int d = 0; d < dim; d++) neighbor_shell *= 3;

    if (use_bf) {
        run_bruteforce(d_input, d_input, P, P, K, dim, radius, d_dists, d_idxs);
    } else if (params.res <= 1 || !grid_feasible || neighbor_shell >= (long long)params.total_cells) {
        if (!run_firstd_search(d_input, P, dim, K, radius)) {
            run_bruteforce(d_input, d_input, P, P, K, dim, radius, d_dists, d_idxs);
        }
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
