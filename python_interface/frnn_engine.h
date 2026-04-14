#ifndef FRNN_ENGINE_H
#define FRNN_ENGINE_H

#include <vector>
#include <cuda_runtime.h>
#include "../frnn/csrc/grid/grid.h" // Points back to your core repository

class FRNNEngine {
    public:
        // Constructor: Reserves memory on the A100 based on a max point count
        FRNNEngine(int max_points);
        
        // Destructor: Automatically cleans up GPU memory when Python is done
        ~FRNNEngine();

        // The interface function that Python will call
        // It takes flattened points [x1, y1, z1, x2, y2, z2...], K, and radius
        std::pair<std::vector<int>, std::vector<float>> search(
            std::vector<float> points_raw, 
            int K, 
            float radius
        );

    private:
        int max_p;
        // GPU Pointers that stay allocated
        float3 *d_points;
        int *d_grid_cnt, *d_grid_offsets, *d_grid_cell, *d_grid_idx, *d_sorted_idxs;
        float *d_dists; 
        int *d_idxs;
};

#endif