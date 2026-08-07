#include "grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
 * =============================================================================
 * insert_points.cu — Stage 1 of the grid FRNN pipeline: spatial hash build
 * =============================================================================
 *
 * Transforms an unsorted SoA point array into a cell-ordered representation
 * that find_nbrs.cu can traverse with coalesced memory access. The pipeline
 * has three logical steps:
 *
 *   1. CountPointsNDKernel  — Each point determines its hypercell and atomically
 *      increments that cell's count in d_grid_cnt. Also records each point's
 *      cell index in d_pc_grid_idx.
 *
 *   2. Prefix scan (Thrust, in frnn_engine.cu) — Converts d_grid_cnt into
 *      d_grid_offsets: the exclusive prefix sum so d_grid_offsets[c] is the
 *      start index in the sorted array for cell c.
 *
 *   3. CountingSortNDKernel (or the D=3 AoS variant) — Uses d_grid_offsets to
 *      scatter each point into its position in the sorted output. Both the
 *      sorted index list (d_sorted_idxs) and the sorted coordinate buffer
 *      (d_points_sorted) are written so find_nbrs can stream candidates without
 *      a gather indirection.
 *
 * All inputs and outputs use SoA layout (p[d*P + i]) except d_points_sorted_aos
 * (the D=3 AoS fast-path buffer), which is written as p[i*3 + d].
 */

/*
 * CountPointsNDKernel — assigns each point to a hypercell and counts occupancy.
 * One thread per point. Reads each coordinate from SoA global memory through
 * the read-only cache (__ldg), computes a row-major cell hash, and atomically
 * increments the cell counter. Points outside [min_val, max_val] are tagged
 * with pc_grid_idx = -1 and skipped in all subsequent kernels.
 *
 * Key variables:
 *   inv_cell      — reciprocal of cell_size; replaces division with multiply
 *   cell_idx      — accumulated row-major hash of this point's hypercell
 *   pc_grid_idx[p]— output: which cell point p belongs to (−1 if OOB)
 */
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

    // Hoist invariants into registers. Replace the per-dimension division by a
    // single reciprocal-multiply: floorf((pos-min)*inv_cell) is fewer ops than a
    // division on A100 (div ~20x slower than mul) and is FMA-friendly. Keeping
    // 256-thread blocks / no __ldg / no unroll preserves the parent's tuning that
    // an over-aggressive prior attempt regressed.
    const float inv_cell = 1.0f / params.cell_size;
    const float min_val = params.min_val;
    const int res = params.res;

    for (int d = 0; d < dim; d++) {
        // __ldg routes streaming SoA reads through the read-only cache, freeing
        // the regular L1 path for the atomic-heavy grid_cnt writes.
        float pos = __ldg(&points[d * P + p]);  // SoA: coords for dim d contiguous
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

/*
 * ReorderIdxsKernel — scatters original point indices into cell-sorted order.
 * One thread per point. Uses an atomicAdd on the target cell's offset slot to
 * claim a unique position in sorted_idxs. Used only by the legacy index-only
 * path (run_reorder_points); the coordinate-copying CountingSortNDKernel is
 * preferred and replaces this in the main pipeline.
 *
 * Key variables:
 *   cell_idx    — which cell this point belongs to (from pc_grid_idx)
 *   offset      — the claimed position within that cell (atomicAdd result)
 *   sorted_idxs — output: original point index stored at the claimed slot
 */
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

/*
 * CountingSortNDKernel — counting sort that writes both sorted indices and sorted
 * coordinates into cell order (SoA layout). Each thread claims its cell slot via
 * atomicAdd, writes its original index to sorted_idxs, and copies all D coordinates
 * from the input SoA into the output SoA at the sorted position. This lets
 * find_nbrs stream candidates sequentially instead of gathering through the index
 * list, which is the key difference that flattens the high-N latency slope.
 *
 * Key variables:
 *   sorted_pos            — claimed position in cell-sorted order (atomicAdd result)
 *   sorted_idxs[sorted_pos] — output: original index p stored at sorted_pos
 *   sorted_points[d*P + sorted_pos] — output: coordinate d of point p at sorted_pos
 */
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
    // __ldg routes read-only source coords through the L1 read-only cache,
    // freeing the regular L1 path for the scattered SoA stores.
    for (int d = 0; d < dim; d++)
        sorted_points[d * P + sorted_pos] = __ldg(&points[d * P + p]);
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

/*
 * CountingSortAoS3Kernel — D=3 counting sort that writes coordinates in AoS layout
 * ([x,y,z] interleaved per point) instead of SoA. When warp threads are spatially
 * sorted and therefore hitting the same candidate p2_idx in lockstep, AoS packs
 * all 3 coords of that candidate into a single 12-byte cache line. A single
 * float3 store writes all three coords atomically-free; three separate SoA stores
 * would need three far-apart cache lines.
 *
 * Key variables:
 *   sorted_pos               — claimed position in cell order (atomicAdd result)
 *   sorted_points[pos*3 + d] — output: AoS layout; one float3 per sorted point
 *   x/y/z                    — three coords loaded via __ldg and stored as make_float3
 */
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
    // Load three coords via read-only cache, emit a single vectorized float3
    // (12-byte) store so find_nbrs consumes all coords from one cache line.
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