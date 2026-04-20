#ifndef GRID_H
#define GRID_H

#include <cuda_runtime.h>

/**
 * GridParams: Optimized for N-Dimensional Latent Spaces.
 * We move away from float3/int3 to support arbitrary dimensions (ML hidden layers).
 * The resolution is now calculated dynamically based on the search radius 
 * to prevent sparseness.
 */
struct GridParams {
    int dim;             // Number of dimensions (e.g., 3, 16, 128)
    float radius;        // The search radius (defines the cell size)
    float min_val;       // The minimum value in the data (origin of the hypergrid)
    float max_val;       // The maximum value in the data
    int res;             // Calculated: ceil((max_val - min_val) / radius)
    long long total_cells; // Total hyper-cells (res ^ dim)
};

/**
 * K-Nearest Neighbor Constraints
 * We keep these to ensure the local registers in find_nbrs.cu 
 * don't overflow the GPU's memory per thread.
 */
constexpr int MAX_K_CAPACITY = 128; // Max neighbors per particle
constexpr int MAX_DIM_SUPPORTED = 128; // Max latent space dimensions

// Helper to calculate 1D hash from N-D grid coordinates
// Used inside kernels to map a point to a specific hyper-cell.
__device__ __host__ inline long long calculate_cell_hash(int* coords, int res, int dim) {
    long long hash = 0;
    long long stride = 1;
    for (int i = 0; i < dim; i++) {
        hash += (long long)coords[i] * stride;
        stride *= res;
    }
    return hash;
}

#endif // GRID_H