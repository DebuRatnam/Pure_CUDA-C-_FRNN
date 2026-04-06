#ifndef GRID_H
#define GRID_H

#include <cuda_runtime.h>

/**
 * GridParams: The "Map" of our 3D Search Space.
 * Using built-in CUDA types (float3, int3) ensures 128-bit memory alignment,
 * which is the fastest way for a GPU to load data.
 */
struct GridParams {
    float3 min_pos;      // The (x, y, z) coordinates of the grid's origin
    float delta;         // The length of one side of a cubic cell (voxel)
    int3 resolution;     // Number of cells along the X, Y, and Z axes
    int total_cells;     // Pre-calculated: resolution.x * resolution.y * resolution.z
};

// Hard-coded limits to keep memory usage predictable on the GPU
#define GRID_3D_MAX_RES 64
#define GRID_2D_MAX_RES 512

// Search limits for the Neighbor Search kernels
// These determine the templates used in find_nbrs.cu
constexpr int V0_MIN_D = 2;
constexpr int V0_MAX_D = 1024;

constexpr int V1_MIN_D = 2;
constexpr int V1_MAX_D = 32;

constexpr int V2_MIN_D = 2;
constexpr int V2_MAX_D = 8;
constexpr int V2_MIN_K = 1;
constexpr int V2_MAX_K = 128;

/**
 * LEGACY MACROS (Keeping these for compatibility with index_utils.cuh)
 * These define the indices if you were still using a flat float* array.
 */
#define GRID_3D_MIN_X 0
#define GRID_3D_MIN_Y 1
#define GRID_3D_MIN_Z 2
#define GRID_3D_DELTA 3
#define GRID_3D_RES_X 4
#define GRID_3D_RES_Y 5
#define GRID_3D_RES_Z 6
#define GRID_3D_TOTAL 7
#define GRID_3D_PARAMS_SIZE 8

#define GRID_2D_MIN_X 0
#define GRID_2D_MIN_Y 1
#define GRID_2D_DELTA 2
#define GRID_2D_RES_X 3
#define GRID_2D_RES_Y 4
#define GRID_2D_TOTAL 5
#define GRID_2D_PARAMS_SIZE 6

#endif // GRID_H
