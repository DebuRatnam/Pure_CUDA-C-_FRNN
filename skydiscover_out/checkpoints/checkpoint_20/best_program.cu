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

    // Hoist struct members to registers; replace per-dim division with a single
    // reciprocal multiply (division is ~20x slower than multiply on A100).
    const float inv_cell = 1.0f / params.cell_size;
    const float min_val = params.min_val;
    const int res = params.res;

    for (int d = 0; d < dim; d++) {
        float pos = points[d * P + p];  // SoA: all coords for dim d are contiguous
        int grid_pos = (int)floorf((pos - min_val) * inv_cell);

        if (grid_pos < 0 || grid_pos >= res) {
            out_of_bounds = true;
            break;
        }
        cell_idx += (long long)grid_pos * stride;
        stride *= res;
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
    // __ldg routes the read-only source coords through the texture/L1 read-only
    // cache, freeing the regular L1 path for the scattered SoA stores and cutting
    // load latency on the cell-order scatter (the dominant cost of this kernel).
    for (int d = 0; d < dim; d++)
        sorted_points[d * P + sorted_pos] = __ldg(&points[d * P + p]);
}

// HOST WRAPPERS
extern "C" void run_insert_points(
    float* d_points, int* d_grid_cnt, int* d_pc_grid_idx,
    int P, int dim, GridParams params) 
{
    /* 256-thread blocks: proven sweet spot for this atomic-heavy, memory-bound
       counting kernel on A100. Larger blocks increase intra-block atomic
       contention without occupancy gains since the SM is already saturated. */
    const int threads = 256;
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

// AoS D=3 variant: same contract as CountingSortNDKernel but writes sorted_points in
// AoS layout (sorted_points[pos*3 + d]) instead of SoA.  When queries are spatially
// sorted, warp threads hit the same candidate p2_idx in lockstep, so a single AoS
// cache line covers all 3 coords for that candidate rather than 3 separate SoA fetches.
__global__ void CountingSortAoS3Kernel(
    const float* __restrict__ points,    // input SoA: points[d*P + p]
    const int* __restrict__ pc_grid_idx,
    int* __restrict__ grid_offsets,      // incremented in place; ends up holding cell ends
    int* __restrict__ sorted_idxs,       // out: sorted_pos -> original index p
    float* __restrict__ sorted_points,   // out: AoS layout, sorted_points[pos*3 + d]
    int P)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int cell_idx = pc_grid_idx[p];
    if (cell_idx == -1) return;
    int sorted_pos = atomicAdd(&grid_offsets[cell_idx], 1);
    sorted_idxs[sorted_pos] = p;
    // Load the three coords through the read-only cache (__ldg) then store them as a
    // single vectorized float3 write. The compiler emits this as one 12-byte
    // contiguous burst (vs three separate 4-byte stores), so the find_nbrs candidate
    // loop consumes all three coords from one cache line with a single store op here.
    const float x = __ldg(&points[0 * P + p]);
    const float y = __ldg(&points[1 * P + p]);
    const float z = __ldg(&points[2 * P + p]);
    *reinterpret_cast<float3*>(sorted_points + sorted_pos * 3) =
        make_float3(x, y, z);
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

// AoS D=3 counting sort: emits d_points_sorted in AoS ([x,y,z] per point) for the
// AoS grid find_nbrs path. dim is fixed at 3 by the caller.
extern "C" void run_counting_sort_aos3(
    float* d_points,
    int* d_pc_grid_idx,
    int* d_grid_offsets,
    int* d_sorted_idxs,
    float* d_points_sorted,
    int P)
{
    int threads = 256;
    int blocks = (P + threads - 1) / threads;
    CountingSortAoS3Kernel<<<blocks, threads>>>(
        d_points, d_pc_grid_idx, d_grid_offsets, d_sorted_idxs, d_points_sorted, P);
}