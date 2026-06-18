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
        float pos = points[d * P + p];  // SoA: all coords for dim d are contiguous
        // Cell size = radius / radius_cell_ratio (see GridParams.cell_size)
        int grid_pos = floor((pos - params.min_val) / params.cell_size);
        
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

// KERNEL 3: Counting sort — physically reorder the point COORDINATES into cell order.
// ReorderIdxsKernel (above) sorts only the index list, so find_nbrs must gather each
// candidate's coordinates through that random permutation (points2[d*P + sorted_idx]),
// a dimension-strided scatter that falls out of L2 as N grows. This kernel instead
// moves the actual coordinates so points in the same cell are contiguous in memory.
// find_nbrs can then stream candidates sequentially — the difference between an
// L2-friendly contiguous read and a cache-thrashing scatter, which is what flattens
// the high-N latency slope. The scatter cost is paid once here (O(N*dim)), not once
// per candidate distance check in the hot loop.
__global__ void CountingSortNDKernel(
    const float* __restrict__ points,    // original SoA: points[d*P + p]
    const int* __restrict__ pc_grid_idx,
    int* __restrict__ grid_offsets,      // cell start offsets; incremented in place
    int* __restrict__ sorted_idxs,       // out: sorted_pos -> original index p
    float* __restrict__ sorted_points,   // out: SoA in cell order, sorted_points[d*P + sorted_pos]
    int P, int dim)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int cell_idx = pc_grid_idx[p];
    if (cell_idx == -1) return;

    // Claim this point's slot within its cell (same atomicAdd contract as ReorderIdxs,
    // so grid_offsets ends holding each cell's END offset, exactly as find_nbrs expects).
    int sorted_pos = atomicAdd(&grid_offsets[cell_idx], 1);
    sorted_idxs[sorted_pos] = p;
    for (int d = 0; d < dim; d++)
        sorted_points[d * P + sorted_pos] = points[d * P + p];
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

// Like run_reorder_points but also emits a cell-ordered copy of the coordinates in
// d_points_sorted (see CountingSortNDKernel). Callers pass d_points_sorted to
// run_find_nbrs as points2 so the candidate loop streams contiguous memory.
extern "C" void run_counting_sort(
    float* d_points,
    int* d_pc_grid_idx,
    int* d_grid_offsets,
    int* d_sorted_idxs,
    float* d_points_sorted,
    int P,
    int dim)
{
    int threads = 256;
    int blocks = (P + threads - 1) / threads;
    CountingSortNDKernel<<<blocks, threads>>>(
        d_points, d_pc_grid_idx, d_grid_offsets, d_sorted_idxs, d_points_sorted, P, dim);
}