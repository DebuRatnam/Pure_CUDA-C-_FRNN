#include "grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
 * =============================================================================
 * find_nbrs.cu — Stage 2 of the grid FRNN pipeline: neighbor search
 * =============================================================================
 *
 * Given the cell-sorted point array built by insert_points.cu, this file
 * implements the neighbor-search kernels that find the K nearest points within
 * radius r for every query.
 *
 * Each query thread iterates over the shell of (2*cell_radius+1)^D neighboring
 * cells. For each neighboring cell it uses an AABB lower-bound test to prune
 * cells that cannot contain a point closer than the current heap worst, then
 * scans the candidate points in that cell and maintains a per-thread max-heap
 * of the K nearest found so far.
 *
 * Two kernel variants cover the two main deployment cases:
 *
 *   FindNbrsNDKernel<CAP, DIM> — general N-D kernel, templated on both heap
 *     capacity and dimension. DIM>0 bakes the dimension in at compile time so
 *     query coordinates live in registers and the dim loops unroll fully.
 *     DIM=0 falls back to a runtime dimension (correct for any D).
 *
 *   FindNbrsAoS3Kernel<CAP> — D=3 fast path with sorted queries (cell-order)
 *     and AoS candidates. Sorted queries mean warp threads share the same
 *     neighbor cells and hit the same candidate cache lines in lockstep,
 *     turning random L2 misses into broadcast hits.
 *
 * ScatterToOrigKernel unpermutes the sorted-query output back to original
 * query order after FindNbrsAoS3Kernel.
 *
 * All outputs are SoA: dists[k*P + p], idxs[k*P + p].
 */

/*
 * insert_neighbor_t<CAP> — max-heap replace-root + sift-down for the grid path.
 * Replaces the current worst neighbor (heap root, local_dists[0]) with the new
 * candidate (d2, idx2) and restores the heap invariant. The depth is a
 * compile-time constant derived from CAP, enabling full loop unroll.
 *
 * Key variables:
 *   local_dists[0] — heap root, the current worst accepted squared distance
 *   CAP            — compile-time heap capacity; determines unrolled sift depth
 *   i              — current node being sifted down toward the leaves
 */
// Templated sift-down: CAP known at compile time so loop depth is exact
// and the compiler can fully unroll and eliminate dead branches.
template<int CAP>
__device__ __forceinline__ float insert_neighbor_t(float* __restrict__ local_dists, int* __restrict__ local_idxs, int K, float d2, int idx2) {
    /* Max-heap replace-root + sift-down. CAP is compile-time so loop depth
       (ceil(log2(CAP))) is a compile-time constant, enabling full unroll.
       Returns new heap root (worst accepted distance) without extra reload. */
    local_dists[0] = d2;
    local_idxs[0]  = idx2;
    int i = 0;
    constexpr int DEPTH = (CAP <= 16) ? 4 : (CAP <= 32) ? 5 : (CAP <= 64) ? 6 : 7;
    #pragma unroll
    for (int depth = 0; depth < DEPTH; depth++) {
        int left = 2 * i + 1, right = 2 * i + 2, largest = i;
        if (left  < K && local_dists[left]  > local_dists[largest]) largest = left;
        if (right < K && local_dists[right] > local_dists[largest]) largest = right;
        if (largest == i) break;
        float td = local_dists[i];  local_dists[i] = local_dists[largest]; local_dists[largest] = td;
        int   ti = local_idxs[i];   local_idxs[i]  = local_idxs[largest];  local_idxs[largest]  = ti;
        i = largest;
    }
    return local_dists[0];
}

/*
 * FindNbrsNDKernel<CAP, DIM> — N-dimensional grid neighbor search.
 * One thread per query point. Loads the query into register arrays (when DIM>0),
 * then iterates over all (2*cell_radius+1)^D neighboring cells. For each cell it
 * computes the AABB minimum distance to prune cells that cannot beat the current
 * heap worst, then walks the candidate span doing FMA distance accumulation and
 * heap insertion. Output is SoA: dists[k*P1 + p1], idxs[k*P1 + p1].
 *
 * Key variables:
 *   q[DCAP]        — register-resident query coordinates; size DIM (compile-time)
 *                    or MAX_DIM_SUPPORTED (runtime fallback). Register vs local
 *                    memory is the dominant high-N slope difference vs xju2.
 *   max_dist_sq    — current heap-worst accepted squared distance; shrinks as
 *                    better neighbors are found, tightening cell pruning over time
 *   cell_min_dist_sq — AABB lower bound to a candidate cell; skips the cell if
 *                    this already exceeds max_dist_sq
 */
template<int CAP, int DIM>
__global__ void __launch_bounds__(256, 4) FindNbrsNDKernel(
    const float* __restrict__ points1,
    const float* __restrict__ points2,
    const int* __restrict__ pc2_grid_off,
    const int* __restrict__ sorted_points2_idxs,
    int P1, int K, int dim, float r2,
    float* __restrict__ dists,
    int* __restrict__ idxs,
    GridParams params)
{
    /* N-dimensional fixed-radius nearest neighbor kernel.
       Templated on CAP (heap size) and DIM (dimension) for register-resident
       arrays and fully unrolled loops. Uses AABB cell pruning, __ldg() for
       all read-only loads, FMA distance accumulation, and templated sift-down. */
    int p1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (p1 >= P1) return;

    // K-nearest scratch heap, sized to CAP at compile time. Initialize full CAP
    // so the compiler knows the array bounds statically (avoids runtime checks).
    float local_dists[CAP];
    int local_idxs[CAP];
    #pragma unroll
    for (int k = 0; k < CAP; ++k) {
        local_dists[k] = r2;
        local_idxs[k] = -1;
    }
    float max_dist_sq = r2;

    constexpr int DCAP = (DIM > 0) ? DIM : MAX_DIM_SUPPORTED;
    const int ndim = (DIM > 0) ? DIM : dim;
    float q[DCAP];
    int cell_coords[DCAP];
    const float inv_cs = 1.0f / params.cell_size;
    #pragma unroll
    for (int d = 0; d < ndim; d++) {
        float pos = __ldg(&points1[d * P1 + p1]);  // SoA, read-only cache
        q[d] = pos;
        int grid_pos = (int)((pos - params.min_val) * inv_cs);
        cell_coords[d] = max(0, min(grid_pos, params.res - 1));
    }

    const int   cr_l   = params.cell_radius;
    const int   res_l  = params.res;
    const float minv_l = params.min_val;
    const float cs_l   = params.cell_size;
    const int W = 2 * cr_l + 1;
    long long num_neighbor_cells = 1;
    #pragma unroll
    for (int d = 0; d < ndim; d++) num_neighbor_cells *= W;

    for (long long nc = 0; nc < num_neighbor_cells; nc++) {
        long long neighbor_hash = 0;
        long long s = 1;
        bool valid = true;
        long long tmp = nc;
        float cell_min_dist_sq = 0.0f;

        #pragma unroll
        for (int d = 0; d < ndim; d++) {
            int offset = (int)(tmp % W) - cr_l;
            tmp /= W;
            int coord = cell_coords[d] + offset;
            if (coord < 0 || coord >= res_l) { valid = false; break; }
            float lo = minv_l + coord * cs_l;
            float hi = lo + cs_l;
            float delta = (q[d] < lo) ? (lo - q[d]) : (q[d] > hi ? q[d] - hi : 0.0f);
            cell_min_dist_sq = __fmaf_rn(delta, delta, cell_min_dist_sq);
            neighbor_hash += (long long)coord * s;
            s *= res_l;
        }
        if (!valid) continue;
        if (cell_min_dist_sq >= max_dist_sq) continue;

        int start = (neighbor_hash == 0) ? 0 : __ldg(&pc2_grid_off[neighbor_hash - 1]);
        int end   = __ldg(&pc2_grid_off[neighbor_hash]);

        for (int p2_idx = start; p2_idx < end; ++p2_idx) {
            float d2 = 0.0f;
            #pragma unroll
            for (int d = 0; d < ndim; d++) {
                float diff = q[d] - __ldg(&points2[d * P1 + p2_idx]);
                d2 = __fmaf_rn(diff, diff, d2);
            }
            if (d2 >= max_dist_sq) continue;
            int original_idx2 = __ldg(&sorted_points2_idxs[p2_idx]);
            max_dist_sq = insert_neighbor_t<CAP>(local_dists, local_idxs, K, d2, original_idx2);
        }
    }

    // Write back: SoA coalesced stores, unroll over full CAP with guard
    #pragma unroll
    for (int k = 0; k < CAP; ++k) {
        if (k < K) {
            dists[k * P1 + p1] = local_dists[k];
            idxs[k * P1 + p1]  = local_idxs[k];
        }
    }
}

/* Grid on GRID_DIM coordinates, but rank every point encountered by its exact
   FULL-D distance.  This is the high-dimensional libFRNN-style path: the grid
   is only an acceleration structure and never an approximate pre-selector.

   Queries are consumed in the same cell-sorted order as the database.  Thus
   neighboring warp lanes usually traverse the same cells and candidate spans,
   improving cache reuse and reducing control-flow divergence.  Results are
   written directly to the original query id, so no output scatter kernel is
   required. */
template<int CAP, int GRID_DIM>
__global__ void FindNbrsGridDimKernel(
    const float* __restrict__ points1,
    const float* __restrict__ points2,
    const int* __restrict__ pc2_grid_off,
    const int* __restrict__ sorted_points2_idxs,
    int P1, int K, int full_dim, float r2,
    float* __restrict__ dists,
    int* __restrict__ idxs,
    GridParams params)
{
    int sorted_p1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_p1 >= P1) return;
    int p1 = __ldg(&sorted_points2_idxs[sorted_p1]);

    float local_dists[CAP];
    int local_idxs[CAP];
    #pragma unroll
    for (int k = 0; k < CAP; ++k) {
        local_dists[k] = r2;
        local_idxs[k] = -1;
    }
    float max_dist_sq = r2;

    float q[MAX_DIM_SUPPORTED];
    for (int d = 0; d < full_dim; ++d)
        q[d] = __ldg(&points1[(long long)d * P1 + p1]);

    int cell_coords[GRID_DIM];
    const float inv_cs = 1.0f / params.cell_size;
    #pragma unroll
    for (int d = 0; d < GRID_DIM; ++d) {
        int grid_pos = (int)((q[d] - params.min_val) * inv_cs);
        cell_coords[d] = max(0, min(grid_pos, params.res - 1));
    }

    constexpr int NUM_OFFSETS = 1;
    const int W = 2 * params.cell_radius + 1;
    long long num_neighbor_cells = NUM_OFFSETS;
    #pragma unroll
    for (int d = 0; d < GRID_DIM; ++d) num_neighbor_cells *= W;

    for (long long nc = 0; nc < num_neighbor_cells; ++nc) {
        long long tmp = nc, hash = 0, stride = 1;
        float cell_min_dist_sq = 0.0f;
        bool valid = true;
        #pragma unroll
        for (int d = 0; d < GRID_DIM; ++d) {
            int coord = cell_coords[d] + (int)(tmp % W) - params.cell_radius;
            tmp /= W;
            if (coord < 0 || coord >= params.res) { valid = false; break; }
            float lo = params.min_val + coord * params.cell_size;
            float hi = lo + params.cell_size;
            float delta = q[d] < lo ? lo - q[d] : (q[d] > hi ? q[d] - hi : 0.0f);
            cell_min_dist_sq = __fmaf_rn(delta, delta, cell_min_dist_sq);
            hash += (long long)coord * stride;
            stride *= params.res;
        }
        if (!valid || cell_min_dist_sq >= max_dist_sq) continue;

        int start = hash == 0 ? 0 : __ldg(&pc2_grid_off[hash - 1]);
        int end = __ldg(&pc2_grid_off[hash]);
        for (int p2_idx = start; p2_idx < end; ++p2_idx) {
            float d2 = 0.0f;
            for (int d = 0; d < full_dim; ++d) {
                float diff = q[d] - __ldg(&points2[(long long)d * P1 + p2_idx]);
                d2 = __fmaf_rn(diff, diff, d2);
            }
            if (d2 >= max_dist_sq) continue;
            int original_idx2 = __ldg(&sorted_points2_idxs[p2_idx]);
            max_dist_sq = insert_neighbor_t<CAP>(local_dists, local_idxs, K, d2, original_idx2);
        }
    }

    #pragma unroll
    for (int k = 0; k < CAP; ++k) if (k < K) {
        dists[k * P1 + p1] = local_dists[k];
        idxs[k * P1 + p1] = local_idxs[k];
    }
}

/*
 * FindNbrsAoS3Kernel<CAP> — D=3 grid search with sorted queries and AoS candidates.
 * Identical search logic to FindNbrsNDKernel but with two structural changes:
 *   (1) Sorted queries — p1 indexes the cell-ordered buffer, so warp threads are
 *       spatially adjacent and enumerate the same neighbor cells in lockstep. Each
 *       candidate cache line is broadcast across the warp from L2/L1 rather than
 *       fetched 32× independently.
 *   (2) AoS candidates — all 3 coords of a candidate sit in a single 12-byte span
 *       so the warp's distance computation needs one cache line, not three.
 * Output is in sorted-query order; ScatterToOrigKernel unpermutes it afterward.
 *
 * Key variables:
 *   qx/qy/qz    — scalar register-resident query coordinates for the fixed D=3 case
 *   max_dist_sq — current heap-worst distance; shrinks as better neighbors are found
 *   cptr        — incrementing pointer into the AoS candidate buffer (advances 3 floats per step)
 */
template<int CAP>
__global__ void __launch_bounds__(128, 8) FindNbrsAoS3Kernel(
    const float* __restrict__ points1_aos,
    const float* __restrict__ points2_aos,
    const int* __restrict__ pc2_grid_off,
    const int* __restrict__ sorted_points2_idxs,
    int P1, int K, float r2,
    float* __restrict__ dists,
    int* __restrict__ idxs,
    GridParams params)
{
    /* D=3 fixed-radius NN with sorted queries (cell-order) and AoS candidates.
       Sorted queries ensure warp threads share the same neighbor cells, so
       candidate data fetched via __ldg broadcasts across the warp from L2/L1.
       No shared-memory tiling: avoids __syncthreads() overhead inside divergent
       cell loops. Incremental AABB pruning skips whole z/y/x cell slabs.
       Templated on CAP for compile-time heap sizing and exact sift-down depth. */
    constexpr int DIM = 3;
    int p1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (p1 >= P1) return;

    float local_dists[CAP];
    int   local_idxs[CAP];
    #pragma unroll
    for (int k = 0; k < CAP; ++k) { local_dists[k] = r2; local_idxs[k] = -1; }
    float max_dist_sq = r2;

    const int cr = params.cell_radius;
    const int res = params.res;
    const float minv = params.min_val;
    const float cs = params.cell_size;
    const float inv_cs = 1.0f / cs;

    float qx = __ldg(&points1_aos[p1 * DIM + 0]);
    float qy = __ldg(&points1_aos[p1 * DIM + 1]);
    float qz = __ldg(&points1_aos[p1 * DIM + 2]);
    int gx = max(0, min((int)((qx - minv) * inv_cs), res - 1));
    int gy = max(0, min((int)((qy - minv) * inv_cs), res - 1));
    int gz = max(0, min((int)((qz - minv) * inv_cs), res - 1));
    int lox_c = max(0, gx - cr), hix_c = min(res - 1, gx + cr);
    int loy_c = max(0, gy - cr), hiy_c = min(res - 1, gy + cr);
    int loz_c = max(0, gz - cr), hiz_c = min(res - 1, gz + cr);

    // Nested cell traversal with incremental AABB distances and per-level pruning.
    // Sorted queries mean warp threads share the same cells -> L2 broadcast hits.
    float loz = minv + loz_c * cs;
    for (int cz = loz_c; cz <= hiz_c; ++cz, loz += cs) {
        float hz = loz + cs;
        float dz0 = (qz < loz) ? (loz - qz) : (qz > hz ? qz - hz : 0.0f);
        float ddz = dz0 * dz0;
        if (ddz >= max_dist_sq) continue;
        long long base_z = (long long)cz * res;
        float loy = minv + loy_c * cs;
        for (int cy = loy_c; cy <= hiy_c; ++cy, loy += cs) {
            float hy = loy + cs;
            float dy0 = (qy < loy) ? (loy - qy) : (qy > hy ? qy - hy : 0.0f);
            float ddyz = ddz + dy0 * dy0;
            if (ddyz >= max_dist_sq) continue;
            long long base_yz = (base_z + cy) * res;
            float lox = minv + lox_c * cs;
            for (int cx = lox_c; cx <= hix_c; ++cx, lox += cs) {
                float hx = lox + cs;
                float dx0 = (qx < lox) ? (lox - qx) : (qx > hx ? qx - hx : 0.0f);
                float cell_min_d2 = ddyz + dx0 * dx0;
                if (cell_min_d2 >= max_dist_sq) continue;

                long long nhash = base_yz + cx;
                int start = (nhash == 0) ? 0 : __ldg(&pc2_grid_off[nhash - 1]);
                int end   = __ldg(&pc2_grid_off[nhash]);

                // Walk candidates with incrementing pointer; __ldg for RO cache.
                // FMA-fused distance; defer original-idx lookup past distance gate.
                const float* cptr = points2_aos + (long long)start * DIM;
                for (int p2_idx = start; p2_idx < end; ++p2_idx, cptr += DIM) {
                    float dx = qx - __ldg(&cptr[0]);
                    float dy = qy - __ldg(&cptr[1]);
                    float dz = qz - __ldg(&cptr[2]);
                    float d2 = __fmaf_rn(dx, dx, __fmaf_rn(dy, dy, dz * dz));
                    if (d2 >= max_dist_sq) continue;
                    int orig2 = __ldg(&sorted_points2_idxs[p2_idx]);
                    max_dist_sq = insert_neighbor_t<CAP>(local_dists, local_idxs, K, d2, orig2);
                }
            }
        }
    }

    // Coalesced SoA stores (consecutive p1 -> consecutive addresses).
    for (int k = 0; k < K; ++k) {
        dists[k * P1 + p1] = local_dists[k];
        idxs[k * P1 + p1]  = local_idxs[k];
    }
}

/*
 * ScatterToOrigKernel — unpermutes sorted-query output back to original query order.
 * One thread per sorted position sp. Reads coalesced (input indexed by sp), writes
 * scattered (output indexed by original id). Neighbor ids in the input are already
 * in original space (mapped by sorted_points2_idxs inside FindNbrsAoS3Kernel), so
 * only the query axis needs unpermuting.
 *
 * Key variables:
 *   sp           — sorted-position index (this thread's identity in the kernel)
 *   orig         — original query index recovered via sorted_idxs[sp]
 *   dists_in/idxs_in[k*P + sp] — input in sorted order; output to [k*P + orig]
 */
__global__ void ScatterToOrigKernel(
    const float* __restrict__ dists_in, const int* __restrict__ idxs_in,
    const int* __restrict__ sorted_idxs,
    int P, int K,
    float* __restrict__ dists_out, int* __restrict__ idxs_out)
{
    /* Unpermute sorted-query output to original order. Reads of dists_in/idxs_in are
       coalesced across the warp (consecutive sp), the permutation index is fetched via
       the read-only cache (__ldg), and the k-loop is lightly unrolled for the common
       small-K cases to reduce loop overhead. */
    int sp = blockIdx.x * blockDim.x + threadIdx.x;
    if (sp >= P) return;
    int orig = __ldg(&sorted_idxs[sp]);
    #pragma unroll 8
    for (int k = 0; k < K; ++k) {
        dists_out[k * P + orig] = __ldg(&dists_in[k * P + sp]);
        idxs_out[k * P + orig]  = __ldg(&idxs_in[k * P + sp]);
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
    // 128 threads/block balances occupancy against register pressure from the
    // unrolled distance + heap code on A100, while keeping coalesced SoA stores.
    int threads = 128;
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

extern "C" void run_find_nbrs_griddim(
    float* d_points1, float* d_points2,
    int* d_pc2_grid_off, int* d_sorted_idxs,
    int P1, int K, int full_dim, int grid_dim, float radius,
    float* d_dists, int* d_idxs, GridParams params)
{
    int threads = 128;
    int blocks = (P1 + threads - 1) / threads;
    float r2 = radius * radius;
    #define LAUNCH_GRID_DIM(CAP, GDIM) FindNbrsGridDimKernel<CAP, GDIM><<<blocks, threads>>>( \
        d_points1, d_points2, d_pc2_grid_off, d_sorted_idxs, P1, K, full_dim, r2, \
        d_dists, d_idxs, params)
    #define DISPATCH_GRID_K(GDIM) do { \
        if      (K <= 16) LAUNCH_GRID_DIM(16, GDIM); \
        else if (K <= 32) LAUNCH_GRID_DIM(32, GDIM); \
        else if (K <= 64) LAUNCH_GRID_DIM(64, GDIM); \
        else              LAUNCH_GRID_DIM(128, GDIM); \
    } while (0)
    if (grid_dim == 4) DISPATCH_GRID_K(4);
    else if (grid_dim == 3) DISPATCH_GRID_K(3);
    else DISPATCH_GRID_K(2);
    #undef DISPATCH_GRID_K
    #undef LAUNCH_GRID_DIM
    cudaDeviceSynchronize();
}
