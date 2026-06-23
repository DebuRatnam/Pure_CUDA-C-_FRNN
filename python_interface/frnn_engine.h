#ifndef FRNN_ENGINE_H
#define FRNN_ENGINE_H

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../frnn/csrc/grid/grid.h"

class FRNNEngine {
    public:
        FRNNEngine(int max_points);
        ~FRNNEngine();

        // CPU path: accepts a flat host array, copies H2D, runs search, copies D2H.
        std::pair<std::vector<int>, std::vector<float>> search(
            std::vector<float> points_raw,
            int K,
            float radius
        );

        // GPU path: accepts a raw CUDA device pointer (e.g. from a torch tensor's
        // .data_ptr()). No H2D or D2H copies — results stay in internal device buffers.
        // Returns (d_idxs_ptr, d_dists_ptr) as uintptr_t.
        std::pair<uintptr_t, uintptr_t> search_gpu(
            uintptr_t dev_ptr, int N, int dim, int K, float radius
        );

        // Copy the last search_gpu() results from device to host.
        // Call this after search_gpu() when you need the actual neighbor indices/distances.
        std::pair<std::vector<int>, std::vector<float>> get_results(int N, int K);

    private:
        int max_p;
        // GPU Pointers
        // Changed d_points from float3* to float* to support N-dimensions
        float* d_points;
        float* d_points_sorted;      // SoA coordinates in cell order (counting sort)
        float* d_points_sorted_aos;  // AoS coordinates in cell order (D=3 grid path)
        int *d_grid_cnt, *d_grid_offsets, *d_grid_idx, *d_sorted_idxs;
        float *d_dists;
        int   *d_idxs;
        // Intermediate sorted-query output; scatter_to_orig unpermutes these into d_dists/d_idxs.
        float *d_dists_sorted;
        int   *d_idxs_sorted;
};

#endif