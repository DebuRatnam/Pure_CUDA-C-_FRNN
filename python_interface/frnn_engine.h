#ifndef FRNN_ENGINE_H
#define FRNN_ENGINE_H

#include <vector>
#include <cuda_runtime.h>
#include "../frnn/csrc/grid/grid.h" 

class FRNNEngine {
    public:
        // Constructor: Reserves memory on the A100 based on a max point count
        FRNNEngine(int max_points);
        
        // Destructor: Automatically cleans up GPU memory when Python is done
        ~FRNNEngine();

        // The interface function that Python will call
        // Now handles flattened N-D points [p1_d1, p1_d2... p1_dn, p2_d1...]
        std::pair<std::vector<int>, std::vector<float>> search(
            std::vector<float> points_raw, 
            int K, 
            float radius
        );

    private:
        int max_p;
        // GPU Pointers
        // Changed d_points from float3* to float* to support N-dimensions
        float* d_points; 
        int *d_grid_cnt, *d_grid_offsets, *d_grid_idx, *d_sorted_idxs;
        float *d_dists; 
        int *d_idxs;
};

#endif