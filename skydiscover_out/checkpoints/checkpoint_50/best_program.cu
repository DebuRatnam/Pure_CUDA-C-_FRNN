#include "grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void insert_neighbor(float* local_dists, int* local_idxs, int K, float d2, int idx2) {
    // Early-exit: candidate is no better than current worst — skip entirely
    if (d2 >= local_dists[0]) return;

    local_dists[0] = d2;
    local_idxs[0]  = idx2;

    // Sift-down the replaced root to restore max-heap order.
    // Bounded loop (depth ≤ floor(log2(128)) = 7) lets the compiler unroll.
    int i = 0;
    #pragma unroll 7
    for (int depth = 0; depth < 7; depth++) {
        int left = 2 * i + 1, right = 2 * i + 2, largest = i;
        if (left  < K && local_dists[left]  > local_dists[largest]) largest = left;
        if (right < K && local_dists[right] > local_dists[largest]) largest = right;
        if (largest == i) break;
        float td = local_dists[i];  local_dists[i] = local_dists[largest]; local_dists[largest] = td;
        int   ti = local_idxs[i];   local_idxs[i]  = local_idxs[largest];  local_idxs[largest]  = ti;
        i = largest;
    }
}

// CAP is the per-thread heap capacity, fixed at compile time so the arrays are sized to
// the actual K (dispatched below) instead of the 128 worst case. K=16 then uses 64 B/thread
// of local memory instead of 1 KB, easing local-memory traffic and L1 pressure. The count
// itself (K) stays a runtime arg, so any K <= CAP is still correct.
//
// DIM is the compile-time dimension. When DIM>0 the per-thread coord arrays (q,
// cell_coords) are sized to DIM — 3 floats at D=3 instead of MAX_DIM_SUPPORTED=128 —
// so they live in REGISTERS instead of local memory, and the dim loops unroll. This
// is the dominant per-candidate cost: q[d] is read on every distance computation, so
// a register read vs a local-memory (DRAM-backed) read is the high-N slope difference
// vs xju2. DIM=0 falls back to a runtime dim (arbitrary D, MAX_DIM_SUPPORTED arrays).
template<int CAP, int DIM>
__global__ void FindNbrsNDKernel(
    const float* __restrict__ points1,
    const float* __restrict__ points2,
    const int* __restrict__ pc2_grid_off,
    const int* __restrict__ sorted_points2_idxs,
    int P1, int K, int dim, float r2,
    float* __restrict__ dists,
    int* __restrict__ idxs,
    GridParams params)
{
    int p1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (p1 >= P1) return;

    // K-nearest scratch heap, sized to CAP (>= K) at compile time.
    float local_dists[CAP];
    int local_idxs[CAP];
    #pragma unroll 16
    for (int k = 0; k < K; ++k) {
        local_dists[k] = r2;
        local_idxs[k] = -1;
    }
    // Tracks heap root (current worst accepted neighbor). Narrows as good
    // neighbors accumulate, allowing early-exit on most candidates.
    float max_dist_sq = r2;

    // 1. Cache the query coords (used for both AABB pruning and distance checks)
    //    and compute its hypergrid cell coordinates in the same pass. With DIM>0
    //    these arrays are DIM-sized and register-resident; ndim is a compile-time
    //    constant so all the dim loops below fully unroll.
    constexpr int DCAP = (DIM > 0) ? DIM : MAX_DIM_SUPPORTED;
    const int ndim = (DIM > 0) ? DIM : dim;
    float q[DCAP];
    int cell_coords[DCAP];
    #pragma unroll
    for (int d = 0; d < ndim; d++) {
        float pos = points1[d * P1 + p1];  // SoA
        q[d] = pos;
        int grid_pos = floor((pos - params.min_val) / params.cell_size); // cell = radius / ratio
        cell_coords[d] = max(0, min(grid_pos, params.res - 1));          // clamp to grid
    }

    // 2. Enumerate the (2*cell_radius+1)^dim cells around the query's cell. The fixed
    //    shell over-covers the search ball: with cell=radius the 3^dim box has ~6.5×
    //    the ball's volume, so most of those cells lie (mostly) outside the sphere.
    //    For each candidate cell we compute the min squared distance from the query
    //    to that cell's axis-aligned bounding box and skip the WHOLE cell when it is
    //    already beyond the current worst neighbor. This is the exact ball-overlap
    //    prune (à la xju2): out-of-ball cells are never scanned, and as the heap fills
    //    max_dist_sq shrinks so the pruned region tightens dynamically.
    const int W = 2 * params.cell_radius + 1;
    long long num_neighbor_cells = 1;
    #pragma unroll
    for (int d = 0; d < ndim; d++) num_neighbor_cells *= W;

    for (long long nc = 0; nc < num_neighbor_cells; nc++) {
        long long neighbor_hash = 0;
        long long s = 1;
        bool valid = true;
        long long tmp = nc;
        float cell_min_dist_sq = 0.0f;  // min ||q - cell_AABB||^2

        #pragma unroll
        for (int d = 0; d < ndim; d++) {
            int offset = (int)(tmp % W) - params.cell_radius;  // maps 0..W-1 → -cr..+cr
            tmp /= W;
            int coord = cell_coords[d] + offset;
            if (coord < 0 || coord >= params.res) { valid = false; break; }
            // Min distance from q[d] to this cell's [lo, hi) extent along dim d.
            float lo = params.min_val + coord * params.cell_size;
            float hi = lo + params.cell_size;
            float delta = (q[d] < lo) ? (lo - q[d]) : (q[d] > hi ? q[d] - hi : 0.0f);
            cell_min_dist_sq += delta * delta;
            neighbor_hash += (long long)coord * s;
            s *= params.res;
        }
        if (!valid) continue;
        // Exact ball-overlap prune: nearest possible point in this cell is already
        // no closer than our worst accepted neighbor → no point inside can qualify.
        if (cell_min_dist_sq >= max_dist_sq) continue;

        // After run_reorder_points, pc2_grid_off[c] holds the END of cell c
        // (atomicAdd increments in-place during scatter-sort).
        // So cell c spans sorted_idxs[ off[c-1] .. off[c]-1 ].
        int start = (neighbor_hash == 0) ? 0 : pc2_grid_off[neighbor_hash - 1];
        int end   = pc2_grid_off[neighbor_hash];

        // points2 is the counting-sorted coordinate buffer (run_counting_sort), so
        // candidates in this cell occupy a CONTIGUOUS span [start, end). Reading
        // points2[d*P1 + p2_idx] across the loop streams sequential memory instead of
        // gathering through the random permutation — this is the cache-locality win.
        // The original index is only needed when a candidate is actually accepted, so
        // the sorted_points2_idxs lookup is deferred past the early-exit test.
        for (int p2_idx = start; p2_idx < end; ++p2_idx) {
            float d2 = 0.0f;
            #pragma unroll
            for (int d = 0; d < ndim; d++) {
                float diff = q[d] - points2[d * P1 + p2_idx];  // contiguous SoA stream
                d2 += diff * diff;
            }
            if (d2 >= max_dist_sq) continue;
            int original_idx2 = sorted_points2_idxs[p2_idx];  // map back to original id, only on accept
            insert_neighbor(local_dists, local_idxs, K, d2, original_idx2);
            max_dist_sq = local_dists[0];
        }
    }

    // 3. Write back to Global Memory (SoA: coalesced warp stores)
    #pragma unroll 16
    for (int k = 0; k < K; ++k) {
        dists[k * P1 + p1] = local_dists[k];
        idxs[k * P1 + p1]  = local_idxs[k];
    }
}

// D=3 grid kernel with SORTED QUERIES + AoS candidate layout.
//
// Two changes over FindNbrsNDKernel, both aimed at the high-N latency slope:
//
//  1. SORTED QUERIES. Thread p1 here is a *sorted (cell-order) position*, not an
//     original point id. points1_aos is the cell-ordered query buffer, so consecutive
//     threads in a warp are spatially adjacent points sharing (nearly) the same grid
//     cell. They enumerate the same neighbor cells and scan the same ~3^3 candidate
//     span in lockstep, so each candidate cache line is fetched once and broadcast to
//     the whole warp from L2/L1 — instead of 32 threads scattering across the domain
//     (original input order is randomized) and each missing cache. Output is produced
//     in sorted order (dists[k*P1+p1]); the engine's scatter kernel unpermutes it back
//     to original query order via sorted_idxs.
//
//  2. AoS CANDIDATES. With the warp locked onto one p2_idx at a time, the 3 coords of
//     that candidate sit in a single 12-byte span (one cache line) in AoS, so the whole
//     warp's distance check is served by one fetch. The SoA path would need 3 separate
//     fetches (one per dim) from three far-apart regions. Query coords are read once
//     into registers, so their layout is immaterial.
template<int CAP>
__global__ void FindNbrsAoS3Kernel(
    const float* __restrict__ points1_aos,   // queries, AoS cell order: points1_aos[p1*3 + d]
    const float* __restrict__ points2_aos,   // candidates, AoS cell order: points2_aos[p2*3 + d]
    const int* __restrict__ pc2_grid_off,
    const int* __restrict__ sorted_points2_idxs,
    int P1, int K, float r2,
    float* __restrict__ dists,               // out, sorted-query order: dists[k*P1 + p1]
    int* __restrict__ idxs,
    GridParams params)
{
    constexpr int DIM = 3;
    int p1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (p1 >= P1) return;

    float local_dists[CAP];
    int   local_idxs[CAP];
    #pragma unroll
    for (int k = 0; k < K; ++k) { local_dists[k] = r2; local_idxs[k] = -1; }
    float max_dist_sq = r2;

    // Query coords + cell, read once into registers (AoS load is one-time, cached here).
    float q[DIM];
    int cell_coords[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        float pos = points1_aos[p1 * DIM + d];
        q[d] = pos;
        int grid_pos = floor((pos - params.min_val) / params.cell_size);
        cell_coords[d] = max(0, min(grid_pos, params.res - 1));
    }

    const int W = 2 * params.cell_radius + 1;
    long long num_neighbor_cells = 1;
    #pragma unroll
    for (int d = 0; d < DIM; d++) num_neighbor_cells *= W;

    for (long long nc = 0; nc < num_neighbor_cells; nc++) {
        long long neighbor_hash = 0;
        long long s = 1;
        bool valid = true;
        long long tmp = nc;
        float cell_min_dist_sq = 0.0f;

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            int offset = (int)(tmp % W) - params.cell_radius;
            tmp /= W;
            int coord = cell_coords[d] + offset;
            if (coord < 0 || coord >= params.res) { valid = false; break; }
            float lo = params.min_val + coord * params.cell_size;
            float hi = lo + params.cell_size;
            float delta = (q[d] < lo) ? (lo - q[d]) : (q[d] > hi ? q[d] - hi : 0.0f);
            cell_min_dist_sq += delta * delta;
            neighbor_hash += (long long)coord * s;
            s *= params.res;
        }
        if (!valid) continue;
        if (cell_min_dist_sq >= max_dist_sq) continue;

        int start = (neighbor_hash == 0) ? 0 : pc2_grid_off[neighbor_hash - 1];
        int end   = pc2_grid_off[neighbor_hash];

        // Warp scans this contiguous span in lockstep; AoS row = one cache line per candidate.
        for (int p2_idx = start; p2_idx < end; ++p2_idx) {
            const float* c = &points2_aos[p2_idx * DIM];
            float dx = q[0] - c[0], dy = q[1] - c[1], dz = q[2] - c[2];
            float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 >= max_dist_sq) continue;
            int original_idx2 = sorted_points2_idxs[p2_idx];
            insert_neighbor(local_dists, local_idxs, K, d2, original_idx2);
            max_dist_sq = local_dists[0];
        }
    }

    // Write in sorted-query order with COALESCED stores (consecutive threads -> consecutive
    // p1). The scatter kernel below does the random unpermute in a cheap bandwidth-bound
    // pass, keeping this compute-heavy kernel's global writes coalesced.
    #pragma unroll 16
    for (int k = 0; k < K; ++k) {
        dists[k * P1 + p1] = local_dists[k];
        idxs[k * P1 + p1]  = local_idxs[k];
    }
}

// Unpermute the sorted-query output (dists_in/idxs_in, laid out [k*P + sorted_pos]) back
// into original query order (dists_out/idxs_out, [k*P + orig]). One thread per sorted
// position; sorted_idxs[sp] gives the original id. Neighbor IDs are already original
// (mapped via sorted_points2_idxs in the kernel), so only the query axis is unpermuted.
__global__ void ScatterToOrigKernel(
    const float* __restrict__ dists_in, const int* __restrict__ idxs_in,
    const int* __restrict__ sorted_idxs,
    int P, int K,
    float* __restrict__ dists_out, int* __restrict__ idxs_out)
{
    int sp = blockIdx.x * blockDim.x + threadIdx.x;
    if (sp >= P) return;
    int orig = sorted_idxs[sp];
    for (int k = 0; k < K; ++k) {
        dists_out[k * P + orig] = dists_in[k * P + sp];
        idxs_out[k * P + orig]  = idxs_in[k * P + sp];
    }
}

extern "C" void run_scatter_to_orig(
    const float* d_dists_in, const int* d_idxs_in,
    const int* d_sorted_idxs,
    int P, int K,
    float* d_dists_out, int* d_idxs_out)
{
    int threads = 256;
    int blocks = (P + threads - 1) / threads;
    ScatterToOrigKernel<<<blocks, threads>>>(
        d_dists_in, d_idxs_in, d_sorted_idxs, P, K, d_dists_out, d_idxs_out);
}

// Host wrapper for the D=3 sorted-query / AoS path. d_points1_aos and d_points2_aos are
// both the cell-ordered AoS buffer (self-search), output is in sorted-query order.
extern "C" void run_find_nbrs_aos3(
    float* d_points1_aos, float* d_points2_aos,
    int* d_pc2_grid_off, int* d_sorted_idxs,
    int P1, int K, float radius,
    float* d_dists, int* d_idxs,
    GridParams params)
{
    int threads = 256;
    int blocks = (P1 + threads - 1) / threads;
    float r2 = radius * radius;
    #define LAUNCH_AOS3(CAP) FindNbrsAoS3Kernel<CAP><<<blocks, threads>>>( \
        d_points1_aos, d_points2_aos, d_pc2_grid_off, d_sorted_idxs, \
        P1, K, r2, d_dists, d_idxs, params)
    if      (K <= 16) LAUNCH_AOS3(16);
    else if (K <= 32) LAUNCH_AOS3(32);
    else if (K <= 64) LAUNCH_AOS3(64);
    else              LAUNCH_AOS3(128);
    #undef LAUNCH_AOS3
    cudaDeviceSynchronize();
}

extern "C" void run_find_nbrs(
    float* d_points1, float* d_points2,
    int* d_pc2_grid_off, int* d_sorted_idxs,
    int P1, int K, int dim, float radius,
    float* d_dists, int* d_idxs,
    GridParams params)
{
    int threads = 256;
    int blocks = (P1 + threads - 1) / threads;
    float r2 = radius * radius;

    // Two-level dispatch: pick the smallest compile-time heap capacity CAP that holds K,
    // and a compile-time DIM for the dims that actually reach the grid path (2/3/4) so the
    // coord arrays are register-resident and the dim loops unroll. DIM=0 is the runtime
    // fallback for any other dimension (correct for all D, just without the small-D win).
    #define LAUNCH_FIND_NBRS(CAP, DIM) FindNbrsNDKernel<CAP, DIM><<<blocks, threads>>>( \
        d_points1, d_points2, d_pc2_grid_off, d_sorted_idxs, \
        P1, K, dim, r2, d_dists, d_idxs, params)
    #define DISPATCH_K(DIM) do { \
        if      (K <= 16)  LAUNCH_FIND_NBRS(16,  DIM); \
        else if (K <= 32)  LAUNCH_FIND_NBRS(32,  DIM); \
        else if (K <= 64)  LAUNCH_FIND_NBRS(64,  DIM); \
        else               LAUNCH_FIND_NBRS(128, DIM); \
    } while (0)
    switch (dim) {
        case 2:  DISPATCH_K(2);  break;
        case 3:  DISPATCH_K(3);  break;
        case 4:  DISPATCH_K(4);  break;
        default: DISPATCH_K(0);  break;  // runtime dim, MAX_DIM_SUPPORTED arrays
    }
    #undef DISPATCH_K
    #undef LAUNCH_FIND_NBRS
    cudaDeviceSynchronize();
}