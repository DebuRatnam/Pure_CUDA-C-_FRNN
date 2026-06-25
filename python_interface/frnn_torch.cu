// frnn_torch.cu — PyTorch/C++ CUDA extension for the pure-CUDA N-D FRNN engine.
//
// Why this file exists:
//   The pybind `frnn_cuda.FRNNEngine.search()` path takes a host (numpy) array,
//   transposes AoS->SoA on the CPU, copies H2D, runs the kernels, copies D2H, and
//   untransposes on the CPU — all inside the timed loop. Benchmarked against FAISS
//   and PyG (which operate on GPU-resident tensors with zero host copies), that is
//   an unfair comparison dominated by PCIe traffic, not kernel time.
//
//   This extension exposes the engine's zero-copy `search_gpu()` pipeline directly
//   to PyTorch. It accepts a CUDA tensor that is already resident on the GPU, does
//   the AoS->SoA transpose GPU->GPU, runs the grid / brute-force dispatch on the
//   tensor's device pointer, and returns CUDA tensors. No host<->device copies occur
//   in the hot path — so FRNN is timed on the same footing as PyG.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cmath>
#include "grid.h"   // GridParams (frnn/csrc/grid, added to include path by setup)

namespace py = pybind11;

// N-D engine kernel host wrappers (defined in the active .cu files; xju2 is unrelated).
extern "C" void run_insert_points(float* d_points, int* d_grid_cnt, int* d_pc_grid_idx,
                                  int P, int dim, GridParams params);
extern "C" void run_reorder_points(int* d_pc_grid_idx, int* d_grid_offsets,
                                   int* d_sorted_idxs, int P, int total_cells);
extern "C" void run_counting_sort(float* d_points, int* d_pc_grid_idx, int* d_grid_offsets,
                                  int* d_sorted_idxs, float* d_points_sorted, int P, int dim);
extern "C" void run_find_nbrs(float* d_points1, float* d_points2, int* d_pc2_grid_off,
                              int* d_sorted_idxs, int P1, int K, int dim, float radius,
                              float* d_dists, int* d_idxs, GridParams params);
extern "C" void run_bruteforce(const float* d_p1, const float* d_p2, int P1, int P2,
                               int K, int dim, float r, float* d_dists, int* d_idxs);
// D=3 sorted-query / AoS grid fast path (insert_points.cu, find_nbrs.cu). Cell-ordered
// queries keep a warp on the same candidate span (L2 broadcast); AoS candidates serve
// all 3 coords from one cache line. Output is in sorted order, unpermuted by scatter.
extern "C" void run_counting_sort_aos3(float* d_points, int* d_pc_grid_idx, int* d_grid_offsets,
                                       int* d_sorted_idxs, float* d_points_sorted, int P);
extern "C" void run_find_nbrs_aos3(float* d_points1_aos, float* d_points2_aos, int* d_pc2_grid_off,
                                   int* d_sorted_idxs, int P1, int K, float radius,
                                   float* d_dists, int* d_idxs, GridParams params);
extern "C" void run_scatter_to_orig(const float* d_dists_in, const int* d_idxs_in,
                                    const int* d_sorted_idxs, int P, int K,
                                    float* d_dists_out, int* d_idxs_out);
// Projection Stage-2: fused candidate verify (frnn/csrc/projection/verify.cu). Recomputes
// true full-D squared distance to each query's candidate ids, keeps the K nearest <= r.
extern "C" void run_verify_candidates(const float* d_pts, const int* d_cand,
                                      int N, int D, int O, int K, float r,
                                      float* d_out_d, int* d_out_i);

// Holds the reusable grid scratch buffers so repeated searches don't re-cudaMalloc.
// Neighbor index/distance outputs are returned as torch tensors (torch's caching
// allocator makes their per-call allocation effectively free after warm-up).
class FRNNTorch {
public:
    explicit FRNNTorch(int max_points) : max_p_(max_points) {
        // Mirror FRNNEngine's scratch sizing: up to 1M cells (3D res=100, 8D res=5...).
        cudaMalloc(&d_grid_cnt_,     max_cells_ * sizeof(int));
        cudaMalloc(&d_grid_offsets_, (max_cells_ + 1) * sizeof(int));
        cudaMalloc(&d_grid_idx_,     max_p_ * sizeof(int));
        cudaMalloc(&d_sorted_idxs_,  max_p_ * sizeof(int));
    }

    ~FRNNTorch() {
        cudaFree(d_grid_cnt_);
        cudaFree(d_grid_offsets_);
        cudaFree(d_grid_idx_);
        cudaFree(d_sorted_idxs_);
    }

    // points: CUDA float32 tensor, shape (N, D), row-major AoS [p*D + d].
    // Returns (idx, dist) CUDA tensors, shape (N, K), row-major AoS [p*K + k].
    std::pair<torch::Tensor, torch::Tensor>
    search(torch::Tensor points, int K, double radius, double radius_cell_ratio = 1.0) {
        TORCH_CHECK(points.is_cuda(),                 "points must be a CUDA tensor");
        TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be float32");
        TORCH_CHECK(points.dim() == 2,                "points must be 2-D (N, D)");

        const int N   = static_cast<int>(points.size(0));
        const int dim = static_cast<int>(points.size(1));
        const float r = static_cast<float>(radius);
        TORCH_CHECK(N <= max_p_, "N=", N, " exceeds max_points=", max_p_);

        // 1. AoS -> SoA on the GPU. points is (N, D) row-major; .t().contiguous()
        //    yields (D, N) row-major == p[d*N + i], the SoA layout the kernels expect.
        //    This is a device->device copy; no host round trip.
        auto pts_soa = points.t().contiguous();
        float* d_points = pts_soa.data_ptr<float>();

        // 2. Dynamic grid params. Cells are radius/ratio on a side; the neighbor loop
        //    scans (2*cell_radius+1)^dim cells. In theory ratio>1 tightens the candidate
        //    set, but MEASURED on the A100 it is ~6% slower at D=3 large-N: finer cells
        //    are mostly empty, and the extra per-cell loop overhead (5^D=125 cells at
        //    ratio=2 vs 3^D=27) outweighs the fewer distance checks. So the default is
        //    ratio=1 (legacy 3^dim); the knob stays exposed for experimentation.
        const float ratio = static_cast<float>(radius_cell_ratio > 0.0 ? radius_cell_ratio : 1.0);
        GridParams params;
        params.dim         = dim;
        params.radius      = r;
        params.min_val     = 0.0f;
        params.max_val     = 1.0f;
        params.cell_size   = r / ratio;
        params.res         = static_cast<int>(std::ceil((params.max_val - params.min_val) / params.cell_size));
        params.total_cells = static_cast<long long>(std::pow((double)params.res, dim));
        params.cell_radius = static_cast<int>(std::ceil(ratio - 1e-6f));  // = ceil(radius/cell_size)

        // 3. Allocate SoA output buffers as CUDA tensors (k*N + p layout).
        auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(points.device());
        auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
        auto idx_soa  = torch::empty({K, N}, i32);
        auto dist_soa = torch::empty({K, N}, f32);
        int*   d_idxs  = idx_soa.data_ptr<int>();
        float* d_dists = dist_soa.data_ptr<float>();

        // 4. Auto-dispatch: brute-force when the neighbor region covers the whole grid
        //    (or res<=1), grid otherwise. The cell-count limit only binds on the grid
        //    path (BF allocates no grid), so it's checked inside that branch.
        const int W = 2 * params.cell_radius + 1;
        long long neighbor_shell = 1;
        for (int d = 0; d < dim; d++) neighbor_shell *= W;

        if (params.res <= 1 || neighbor_shell >= params.total_cells) {
            run_bruteforce(d_points, d_points, N, N, K, dim, r, d_dists, d_idxs);
        } else {
            TORCH_CHECK(params.total_cells <= max_cells_,
                        "Grid too fine for D=", dim, " (cells=", params.total_cells,
                        "); lower radius_cell_ratio or raise radius.");
            cudaMemset(d_grid_cnt_, 0, params.total_cells * sizeof(int));
            run_insert_points(d_points, d_grid_cnt_, d_grid_idx_, N, dim, params);
            thrust::exclusive_scan(
                thrust::device_ptr<int>(d_grid_cnt_),
                thrust::device_ptr<int>(d_grid_cnt_ + params.total_cells),
                thrust::device_ptr<int>(d_grid_offsets_));
            // Counting sort: physically reorder coords into cell order so find_nbrs
            // streams contiguous candidates. The sorted buffer is a transient (dim, N)
            // tensor — torch's caching allocator makes this effectively free after warm-up.
            if (dim == 3) {
                // D=3 fast path: cell-ordered (sorted) queries + AoS candidates so a warp
                // scans each candidate span in lockstep from one cache line. find_nbrs_aos3
                // writes coalesced in sorted order; scatter unpermutes into d_dists/d_idxs.
                auto pts_sorted_aos = torch::empty({N, 3}, f32);   // AoS [pos*3 + d]
                auto dist_sorted    = torch::empty({K, N}, f32);
                auto idx_sorted     = torch::empty({K, N}, i32);
                float* d_points_sorted_aos = pts_sorted_aos.data_ptr<float>();
                float* d_dists_sorted      = dist_sorted.data_ptr<float>();
                int*   d_idxs_sorted       = idx_sorted.data_ptr<int>();
                run_counting_sort_aos3(d_points, d_grid_idx_, d_grid_offsets_, d_sorted_idxs_,
                                       d_points_sorted_aos, N);
                run_find_nbrs_aos3(d_points_sorted_aos, d_points_sorted_aos, d_grid_offsets_,
                                   d_sorted_idxs_, N, K, r, d_dists_sorted, d_idxs_sorted, params);
                run_scatter_to_orig(d_dists_sorted, d_idxs_sorted, d_sorted_idxs_,
                                    N, K, d_dists, d_idxs);
            } else {
                auto pts_sorted = torch::empty({dim, N}, f32);
                float* d_points_sorted = pts_sorted.data_ptr<float>();
                run_counting_sort(d_points, d_grid_idx_, d_grid_offsets_, d_sorted_idxs_,
                                  d_points_sorted, N, dim);
                run_find_nbrs(d_points, d_points_sorted, d_grid_offsets_, d_sorted_idxs_,
                              N, K, dim, r, d_dists, d_idxs, params);
            }
        }

        // 5. SoA -> AoS on the GPU so callers get row-major (N, K). Device->device.
        return {idx_soa.t().contiguous(), dist_soa.t().contiguous()};
    }

    // Projection Stage-2: fused candidate verify. Given each query's candidate ids
    // (cand, (N, O) int32 — the oversample-nearest in the projection), recompute the
    // true full-D distance on the GPU and return the K nearest within radius.
    //   points: CUDA float32 (N, D) AoS.  cand: CUDA int32 (N, O) AoS, ids or -1.
    //   Returns (idx, dist) (N, K): ids and SQUARED distance, heap order.
    std::pair<torch::Tensor, torch::Tensor>
    verify_candidates(torch::Tensor points, torch::Tensor cand, int K, double radius) {
        TORCH_CHECK(points.is_cuda() && cand.is_cuda(), "points and cand must be CUDA tensors");
        TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be float32");
        TORCH_CHECK(cand.scalar_type() == torch::kInt32,     "cand must be int32");
        TORCH_CHECK(points.dim() == 2 && cand.dim() == 2,    "points (N,D) and cand (N,O) must be 2-D");
        TORCH_CHECK(points.size(0) == cand.size(0),          "points and cand must share N");

        const int N = (int)points.size(0);
        const int D = (int)points.size(1);
        const int O = (int)cand.size(1);
        auto pts = points.contiguous();
        auto c   = cand.contiguous();

        auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(points.device());
        auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
        auto out_i = torch::empty({N, K}, i32);
        auto out_d = torch::empty({N, K}, f32);

        run_verify_candidates(pts.data_ptr<float>(), c.data_ptr<int>(),
                              N, D, O, K, (float)radius,
                              out_d.data_ptr<float>(), out_i.data_ptr<int>());
        return {out_i, out_d};
    }

private:
    int max_p_;
    static constexpr long long max_cells_ = 1000000;  // matches FRNNEngine
    int* d_grid_cnt_     = nullptr;
    int* d_grid_offsets_ = nullptr;
    int* d_grid_idx_     = nullptr;
    int* d_sorted_idxs_  = nullptr;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Zero-copy PyTorch interface to the N-D FRNN CUDA engine";
    py::class_<FRNNTorch>(m, "FRNNTorch")
        .def(py::init<int>(), py::arg("max_points"))
        .def("search", &FRNNTorch::search,
             py::arg("points"), py::arg("K"), py::arg("radius"),
             py::arg("radius_cell_ratio") = 1.0,
             "FRNN search on a GPU-resident (N, D) float32 tensor. radius_cell_ratio "
             "sets grid cell size = radius/ratio (default 1.0 = 3^D shell, fastest here; "
             ">1 uses finer cells but is slower on this kernel). Returns (idx, dist) "
             "CUDA tensors of shape (N, K).")
        .def("verify_candidates", &FRNNTorch::verify_candidates,
             py::arg("points"), py::arg("cand"), py::arg("K"), py::arg("radius"),
             "Projection Stage-2: fused full-D verify of per-query candidate ids "
             "(points (N,D) float32, cand (N,O) int32). Returns (idx, dist) (N,K): "
             "ids and squared distance of the K nearest within radius.");
}
