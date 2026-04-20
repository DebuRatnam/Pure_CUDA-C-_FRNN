#include "grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// KERNEL 1: Count how many points fall into each hyper-cell
__global__ void CountPointsNDKernel(
    const float* __restrict__ points,
    int* __restrict__ grid_cnt,
    int* __restrict__ pc_grid_idx,
    int P, int dim,
    GridParams params) 
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    long long cell_idx = 0;
    long long stride = 1;
    bool out_of_bounds = false;

    for (int d = 0; d < dim; d++) {
        float pos = points[p * dim + d];
        // Dynamic scaling: cell size = radius
        int grid_pos = floor((pos - params.min_val) / params.radius);
        
        if (grid_pos < 0 || grid_pos >= params.res) {
            out_of_bounds = true;
            break;
        }
        cell_idx += (long long)grid_pos * stride;
        stride *= params.res;
    }

    if (!out_of_bounds && cell_idx < params.total_cells) {
        atomicAdd(&grid_cnt[cell_idx], 1);
        pc_grid_idx[p] = (int)cell_idx;
    } else {
        pc_grid_idx[p] = -1; 
    }
}

// KERNEL 2: Map point indices to the sorted grid list
__global__ void ReorderIdxsKernel(
    const int* __restrict__ pc_grid_idx,
    int* __restrict__ grid_offsets, // This gets incremented by atomicAdd
    int* __restrict__ sorted_idxs,
    int P) 
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int cell_idx = pc_grid_idx[p];
    if (cell_idx != -1) {
        // Find the specific slot for this point in its cell
        int offset = atomicAdd(&grid_offsets[cell_idx], 1);
        sorted_idxs[offset] = p;
    }
}

// HOST WRAPPERS
extern "C" void run_insert_points(
    float* d_points, int* d_grid_cnt, int* d_pc_grid_idx,
    int P, int dim, GridParams params) 
{
    int threads = 256;
    int blocks = (P + threads - 1) / threads;
    CountPointsNDKernel<<<blocks, threads>>>(d_points, d_grid_cnt, d_pc_grid_idx, P, dim, params);
}

extern "C" void run_reorder_points(
    int* d_pc_grid_idx, 
    int* d_grid_offsets, 
    int* d_sorted_idxs, 
    int P,
    int total_cells) // Note: total_cells parameter added for ND compatibility
{
    int threads = 256;
    int blocks = (P + threads - 1) / threads;
    ReorderIdxsKernel<<<blocks, threads>>>(d_pc_grid_idx, d_grid_offsets, d_sorted_idxs, P);
}