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
// D=3 sorted-query / AoS grid path (see find_nbrs.cu, insert_points.cu)
extern "C" void run_counting_sort_aos3(float* d_points, int* d_pc_grid_idx, int* d_grid_offsets, int* d_sorted_idxs, float* d_points_sorted, int P);
extern "C" void run_find_nbrs_aos3(float* d_points1_aos, float* d_points2_aos, int* d_pc2_grid_off, int* d_sorted_idxs, int P1, int K, float radius, float* d_dists, int* d_idxs, GridParams params);
extern "C" void run_scatter_to_orig(const float* d_dists_in, const int* d_idxs_in, const int* d_sorted_idxs, int P, int K, float* d_dists_out, int* d_idxs_out);
// Projection path (project.cu / verify.cu). AoS in/out; see those files for layouts.
extern "C" int  run_pca_project(const float* d_pts, int N, int D, int k, float* d_proj01,
                                float* d_sumx, float* d_sumxx, float* d_mean, float* d_basis,
                                float* d_minmax, float* out_s, float* out_var_ratio);
extern "C" void run_verify_candidates(const float* d_pts, const int* d_cand, int N, int D,
                                      int O, int K, float r, float* d_out_d, int* d_out_i);

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

    // Projection-path scratch. PCA stats need D<=default_dim; project/verify use AoS.
    cudaMalloc(&d_pts_aos,   max_p * default_dim * sizeof(float));
    cudaMalloc(&d_sumx,      default_dim * sizeof(float));
    cudaMalloc(&d_sumxx,     (size_t)default_dim * default_dim * sizeof(float));
    cudaMalloc(&d_mean,      default_dim * sizeof(float));
    cudaMalloc(&d_basis,     default_dim * PROJ_K * sizeof(float));
    cudaMalloc(&d_minmax,    2 * PROJ_K * sizeof(float));
    cudaMalloc(&d_proj01,    max_p * PROJ_K * sizeof(float));
    cudaMalloc(&d_proj_soa,  max_p * PROJ_K * sizeof(float));
    cudaMalloc(&d_cand,      max_p * 128 * sizeof(int));  // O up to 128
}

FRNNEngine::~FRNNEngine() {
    cudaFree(d_points); cudaFree(d_points_sorted); cudaFree(d_points_sorted_aos);
    cudaFree(d_grid_cnt); cudaFree(d_grid_offsets);
    cudaFree(d_grid_idx); cudaFree(d_sorted_idxs);
    cudaFree(d_dists); cudaFree(d_idxs);
    cudaFree(d_dists_sorted); cudaFree(d_idxs_sorted);
    cudaFree(d_pts_aos); cudaFree(d_sumx); cudaFree(d_sumxx);
    cudaFree(d_mean); cudaFree(d_basis); cudaFree(d_minmax);
    cudaFree(d_proj01); cudaFree(d_proj_soa); cudaFree(d_cand);
}

// Two-stage projection search. Runs when the full-D grid is infeasible and D > PROJ_K.
//   1. transpose SoA input -> AoS; PCA-project D -> PROJ_K into [0,1]^k (scale s).
//   2. grid-search the projection at radius R*s (contractive => superset of true nbrs),
//      oversampling to O = min(K*4, 128) candidates per query.
//   3. verify: recompute full-D distances on the O candidates, keep K within R.
// Final results land in d_dists/d_idxs (SoA [k*N+q], original ids), matching the grid
// path so search()/search_gpu()/get_results() are unchanged. This is the sole non-grid
// path (brute-force was removed): every case where the full-D grid is infeasible routes
// here. Returns false if PCA stats don't fit or the projected grid is degenerate.
bool FRNNEngine::run_projection_search(const float* d_in_soa, int N, int D, int K, float radius) {
    const int O = std::min(K * 4, 128);
    const int T = 256;
    auto blocks = [&](long long n){ return (int)((n + T - 1) / T); };

    // 1. SoA (D x N) -> AoS (N x D)
    transpose_f<<<blocks((long long)D * N), T>>>(d_in_soa, d_pts_aos, D, N);

    // 2. PCA project D -> PROJ_K. s = isotropic scale; search radius = R*s.
    float s = 1.0f, var_ratio = 1.0f;
    int rc = run_pca_project(d_pts_aos, N, D, PROJ_K, d_proj01,
                             d_sumx, d_sumxx, d_mean, d_basis, d_minmax, &s, &var_ratio);
    if (rc != 0) return false;   // D too large for shared-mem stats path

    // proj01 AoS (N x k) -> SoA (k x N) for the grid pipeline
    transpose_f<<<blocks((long long)N * PROJ_K), T>>>(d_proj01, d_proj_soa, N, PROJ_K);

    // 3. Projected grid params. Feasibility mirrors the main dispatcher; if the
    // projected grid is degenerate the projection buys nothing -> brute-force.
    //
    // Idea 1 (candidate-density reduction): decouple cell_size from the search
    // radius. With cell_size = R*s (cell_div = 1) each query scans the 3^3 = 27-cell
    // shell -- a (3*R*s)^3 box that is ~6.4x the volume of the (R*s)-ball it must
    // cover, so most scanned points are corner points beyond R that get distance-
    // tested and discarded. Setting cell_size = R*s / cell_div (cell_radius = cell_div)
    // scans a (2*cell_div+1)^PROJ_K shell of finer cells whose union hugs the ball
    // more tightly: scanned box side = R*s*(2 + 1/cell_div), shrinking 3*R*s -> 2.5*R*s
    // at cell_div=2 (~1.7x fewer candidates), toward the 2*R*s limit as cell_div grows.
    // cell_radius = ceil(R*s / cell_size) = cell_div exactly covers the radius, so the
    // superset guarantee (hence correctness) is preserved. Tunable at runtime via
    // FRNN_PROJ_CELL_DIV (default 2); =1 reproduces the legacy cell=radius behavior.
    // cell_div is clamped down so the finer grid's res^PROJ_K stays within the 1M-cell
    // budget (avoids the infeasible-fallback path, which currently yields no results).
    static const bool dbg = std::getenv("FRNN_DEBUG_PROJ") != nullptr;

    // Idea 2: shrink projected search radius by sqrt(var_ratio) — the contractive
    // factor of an orthonormal PCA projection (any true neighbor with full_dist <= R
    // has proj_dist <= full_dist * sqrt(var_ratio)).  A 1.05x safety buffer guards
    // borderline cases.  rscale is clamped to [0,1] so it never inflates the radius.
    // Only applied at N >= 50000: below that the verify kernel is cheap, but a
    // smaller radius forces a finer grid (more cells) whose setup cost dominates.
    // Override at runtime with FRNN_PROJ_RSCALE (e.g. "0.9" to tune manually).
    float rscale = (N >= 50000) ? std::min(1.0f, sqrtf(var_ratio) * 1.05f) : 1.0f;
    if (const char* e = std::getenv("FRNN_PROJ_RSCALE")) {
        float v = std::atof(e);
        if (v > 0.0f && v <= 1.0f) rscale = v;
    }
    // Feasibility floor: never reduce rscale past the point where the coarsest
    // feasible grid (cell_div=1) would exceed 1M cells.
    // With cell_div=1: res = ceil(1 / (radius*s*rscale)); need res^PROJ_K <= 1M.
    // → rscale >= 1 / (floor(1M^(1/PROJ_K)) * radius * s).
    // Guard: only apply when rscale_floor < 1 (else the fallback below handles it).
    {
        float max_res = std::floor(std::pow(1000000.0, 1.0 / PROJ_K));  // 31 for PROJ_K=4
        float rscale_floor = 1.0f / (max_res * radius * s);
        if (rscale_floor < 1.0f) rscale = std::max(rscale, rscale_floor);
    }

    int cell_div = 2;
    if (const char* e = std::getenv("FRNN_PROJ_CELL_DIV")) {
        int v = std::atoi(e);
        if (v >= 1) cell_div = v;
    }

    GridParams pp;
    pp.dim = PROJ_K;
    pp.radius = radius * s * rscale;
    pp.min_val = 0.0f;
    pp.max_val = 1.0f;

    // Clamp cell_div so res = ceil(cell_div / (R*s)) keeps res^PROJ_K <= 1e6.
    while (cell_div > 1) {
        int res_try = (int)std::ceil((double)cell_div / pp.radius);
        if (std::pow((double)res_try, PROJ_K) <= 1000000.0) break;
        cell_div--;
    }

    pp.cell_size = pp.radius / (float)cell_div;
    pp.cell_radius = cell_div;
    pp.res = (int)std::ceil((pp.max_val - pp.min_val) / pp.cell_size);
    double tc = std::pow((double)pp.res, PROJ_K);
    long long shell = 1;
    for (int d = 0; d < PROJ_K; d++) shell *= (2 * cell_div + 1);  // (2*cell_div+1)^PROJ_K
    if (pp.res <= 1 || tc > 1000000.0 || shell >= (long long)tc) {
        if (dbg) std::cerr << "[proj] fallback: D=" << D << " R*s*rscale=" << pp.radius
                           << " res=" << pp.res << " cell_div=" << cell_div
                           << " s=" << s << " var=" << var_ratio << " rscale=" << rscale << "\n";
        return false;
    }
    pp.total_cells = (long long)tc;
    if (dbg) std::cerr << "[proj] engaged: D=" << D << " R*s*rscale=" << pp.radius
                       << " res=" << pp.res << " cells=" << pp.total_cells
                       << " cell_div=" << cell_div << " shell=" << shell
                       << " s=" << s << " var=" << var_ratio << " rscale=" << rscale
                       << " O=" << O << "\n";

    // Grid search on the projection at O oversample (generic N-D path; DIM=PROJ_K
    // specialization in run_find_nbrs keeps query coords register-resident).
    // No scatter_to_orig needed: the generic path writes original-query-order SoA directly.
    cudaMemset(d_grid_cnt, 0, pp.total_cells * sizeof(int));
    run_insert_points(d_proj_soa, d_grid_cnt, d_grid_idx, N, PROJ_K, pp);
    thrust::exclusive_scan(
        thrust::device_ptr<int>(d_grid_cnt),
        thrust::device_ptr<int>(d_grid_cnt + pp.total_cells),
        thrust::device_ptr<int>(d_grid_offsets));
    run_counting_sort(d_proj_soa, d_grid_idx, d_grid_offsets, d_sorted_idxs, d_points_sorted, N, PROJ_K);
    run_find_nbrs(d_proj_soa, d_points_sorted, d_grid_offsets, d_sorted_idxs,
                  N, O, PROJ_K, pp.radius, d_dists, d_idxs, pp);
    // d_idxs holds SoA [o*N+q] original candidate ids.

    // candidate ids SoA (O x N) -> AoS (N x O) for verify
    transpose_i<<<blocks((long long)O * N), T>>>(d_idxs, d_cand, O, N);

    // 4. Full-D verify -> AoS output in the sorted scratch buffers (now free).
    run_verify_candidates(d_pts_aos, d_cand, N, D, O, K, radius, d_dists_sorted, d_idxs_sorted);

    // verify AoS (N x K) -> engine SoA (K x N) convention in d_dists/d_idxs
    transpose_f<<<blocks((long long)N * K), T>>>(d_dists_sorted, d_dists, N, K);
    transpose_i<<<blocks((long long)N * K), T>>>(d_idxs_sorted, d_idxs, N, K);
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
    // The grid kernel checks 3^D neighboring cells per point.  When that shell
    // covers >= the entire grid (3^D >= total_cells), the per-cell overhead
    // dominates and brute-force is strictly faster.  For D=16 with res=2,
    // 3^16=43M >> 2^16=65K, so the grid is catastrophically slow there.
    // res<=1 also forces brute-force because the prefix-sum breaks on one cell.
    long long neighbor_shell = 1;
    for (int d = 0; d < dim; d++) neighbor_shell *= 3;

    if (params.res <= 1 || !grid_feasible || neighbor_shell >= (long long)params.total_cells) {
        // Grid infeasible: project D -> PROJ_K, grid-search the projection, verify in
        // full D. Sole non-grid path — brute-force removed. Results land in d_dists/d_idxs.
        if (!run_projection_search(d_points, P, dim, K, radius)) {
            // Projected grid also degenerate: return empty result rather than garbage.
            cudaMemset(d_dists, 0, (size_t)P * K * sizeof(float));
            cudaMemset(d_idxs, 0xFF, (size_t)P * K * sizeof(int));  // -1 per slot
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

    long long neighbor_shell = 1;
    for (int d = 0; d < dim; d++) neighbor_shell *= 3;

    if (params.res <= 1 || !grid_feasible || neighbor_shell >= (long long)params.total_cells) {
        // Grid infeasible: project D -> PROJ_K, grid-search the projection, verify in
        // full D. Sole non-grid path — brute-force removed. Results land in d_dists/d_idxs.
        if (!run_projection_search(d_input, P, dim, K, radius)) {
            cudaMemset(d_dists, 0, (size_t)P * K * sizeof(float));
            cudaMemset(d_idxs, 0xFF, (size_t)P * K * sizeof(int));
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

