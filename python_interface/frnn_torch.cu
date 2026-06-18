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
            auto pts_sorted = torch::empty({dim, N}, f32);
            float* d_points_sorted = pts_sorted.data_ptr<float>();
            run_counting_sort(d_points, d_grid_idx_, d_grid_offsets_, d_sorted_idxs_,
                              d_points_sorted, N, dim);
            run_find_nbrs(d_points, d_points_sorted, d_grid_offsets_, d_sorted_idxs_,
                          N, K, dim, r, d_dists, d_idxs, params);
        }

        // 5. SoA -> AoS on the GPU so callers get row-major (N, K). Device->device.
        return {idx_soa.t().contiguous(), dist_soa.t().contiguous()};
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
             "CUDA tensors of shape (N, K).");
}
