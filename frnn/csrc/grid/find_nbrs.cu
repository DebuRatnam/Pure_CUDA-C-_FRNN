#include "grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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