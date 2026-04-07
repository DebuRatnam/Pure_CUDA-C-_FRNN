// Tell the compiler this function exists in your other repo
extern "C" void gpu_prefix_sum(int* d_input, int* d_output, int n);
#include "grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// The Kernel: Assigns each point to a grid cell
__global__ void InsertPoints3DKernel(
    const float3* __restrict__ points, 
    int* __restrict__ grid_cnt, 
    int* __restrict__ grid_cell, 
    int* __restrict__ grid_idx, 
    int P, 
    GridParams params) 
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    float3 pt = points[p];

    // Calculate 3D grid coordinates
    int gx = floor((pt.x - params.min_pos.x) / params.delta);
    int gy = floor((pt.y - params.min_pos.y) / params.delta);
    int gz = floor((pt.z - params.min_pos.z) / params.delta);

    // Ensure we are within grid boundaries
    if (gx >= 0 && gx < params.resolution.x &&
        gy >= 0 && gy < params.resolution.y &&
        gz >= 0 && gz < params.resolution.z) 
    {
        // Compute unique 1D index for the 3D cell
        int cell_idx = gx * (params.resolution.y * params.resolution.z) + 
                       gy * params.resolution.z + 
                       gz;

        grid_cell[p] = cell_idx;
        
        // Atomic increment to count how many points are in this cell
        // This count is what your Prefix Sum will process next!
        grid_idx[p] = atomicAdd(&grid_cnt[cell_idx], 1);
    } else {
        grid_cell[p] = -1; // Out of bounds
    }
}

// C++ Wrapper to launch the kernel from your Main.cpp
extern "C" void run_insert_points(
    float3* d_points, 
    int* d_grid_cnt, 
    int* d_grid_cell, 
    int* d_grid_idx, 
    int P, 
    GridParams params) 
{
    int threads = 256;
    int blocks = (P + threads - 1) / threads;

    InsertPoints3DKernel<<<blocks, threads>>>(
        d_points, d_grid_cnt, d_grid_cell, d_grid_idx, P, params
    );
    
    // Ensure the GPU finishes before we move to Prefix Sum
    cudaDeviceSynchronize();
}

__global__ void set_identity_idx_kernel(int* d_idx, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P) d_idx[i] = i;
}

extern "C" void run_set_identity(int* d_idx, int P) {
    int threads = 256;
    int blocks = (P + threads - 1) / threads;
    set_identity_idx_kernel<<<blocks, threads>>>(d_idx, P);
}

// This kernel maps the original point index to its new "sorted" position
__global__ void ReorderPointsKernel(
    const int* __restrict__ grid_cell,  // What cell is this point in?
    const int* __restrict__ grid_idx,   // What is its local index in that cell?
    const int* __restrict__ grid_offsets, // Where does each cell start?
    int* __restrict__ sorted_idxs,      // OUTPUT: The mapping
    int P) 
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int cell = grid_cell[p];
    
    // If point is out of bounds, we don't sort it
    if (cell != -1) {
        int local_idx = grid_idx[p];
        int start_pos = grid_offsets[cell];
        
        // The magic formula: Start of the cell + position inside the cell
        int sorted_pos = start_pos + local_idx;
        
        // Store the original point index at the sorted position
        sorted_idxs[sorted_pos] = p;
    }
}

extern "C" void run_reorder_points(
    int* d_grid_cell, 
    int* d_grid_idx, 
    int* d_grid_offsets, 
    int* d_sorted_idxs, 
    int P) 
{
    int threads = 256;
    int blocks = (P + threads - 1) / threads;

    ReorderPointsKernel<<<blocks, threads>>>(
        d_grid_cell, d_grid_idx, d_grid_offsets, d_sorted_idxs, P
    );
    cudaDeviceSynchronize();
}