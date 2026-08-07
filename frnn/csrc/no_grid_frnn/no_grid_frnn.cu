#include "no_grid_frnn.h"
#include <device_launch_parameters.h>
#include <float.h>
#include <cstdlib>
#include <cstdio>

/*
 * =============================================================================
 * no_grid_frnn.cu — Tiled brute-force FRNN: fallback for high-D / large-radius
 * =============================================================================
 *
 * Implements O(N^2) fixed-radius nearest-neighbor search without any spatial
 * index. FRNNEngine dispatches here when the uniform grid in insert_points.cu /
 * find_nbrs.cu is infeasible (3^D ≥ total_cells, total_cells > 1M, or res ≤ 1).
 *
 * The key optimization is shared-memory tiling: each block cooperatively loads
 * BLOCKDIM reference points into shared memory in AoS layout (coords contiguous
 * per point), then every query thread in the block scans the tile. This amortizes
 * the global memory bandwidth cost by a factor of blockDim.x — the tile is fetched
 * once and reused by all 128–512 threads.
 *
 * Two kernel variants:
 *
 *   TiledBruteforceNDKernel<CAP> — general N-D. Query stored in a 128-float
 *     local array (BF_MAX_DIM); distance uses float4 loads when dim%4==0.
 *
 *   TiledBruteforce16Kernel<CAP> — D=16 specialization. Query is held in 4
 *     float4 registers (q0..q3) so it is never re-fetched from local memory.
 *     Distance computation is fully unrolled to 4 float4 subtract-accumulate
 *     blocks, matching the throughput of a cuBLAS GEMM kernel at this size.
 *
 * run_bruteforce selects the block size to maximize reference reuse (bigger
 * blocks = wider tiles = fewer global fetches), subjects to shared-memory limits,
 * and dispatches the right kernel and CAP via the DISPATCH_BF / LAUNCH_BF macros.
 */

// Engine hard limit (D <= 128); matches MAX_DIM_SUPPORTED in grid.h.
constexpr int BF_MAX_DIM = 128;

/*
 * bf_dist2 — squared L2 distance between query q and tile entry t.
 * When dim is divisible by 4, uses 128-bit float4 loads (one LDS.128 per 4
 * coords) instead of four scalar loads, halving the load instruction count on
 * A100. Both q and &tile[tj*dim] are 16-byte aligned so the reinterpret cast is
 * valid. Falls back to a scalar loop for non-multiple-of-4 dims.
 *
 * Key variables:
 *   q    — 16-byte-aligned register array of the query point (loaded once per tile)
 *   t    — pointer into shared memory tile at &tile[tj*dim] (AoS, 16B aligned)
 *   nv   — number of float4 words when dim%4==0 (= dim/4)
 */
__device__ __forceinline__ float bf_dist2(
    const float* __restrict__ q, const float* __restrict__ t, int dim)
{
    float d2 = 0.0f;
    if ((dim & 3) == 0) {
        const float4* q4 = reinterpret_cast<const float4*>(q);
        const float4* t4 = reinterpret_cast<const float4*>(t);
        const int nv = dim >> 2;
        #pragma unroll 4
        for (int v = 0; v < nv; v++) {
            float4 a = q4[v];
            float4 b = t4[v];
            float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z, dw = a.w - b.w;
            d2 += dx * dx + dy * dy + dz * dz + dw * dw;
        }
    } else {
        for (int d = 0; d < dim; d++) {
            float diff = q[d] - t[d];
            d2 += diff * diff;
        }
    }
    return d2;
}

/*
 * bf_insert_neighbor — max-heap replace-root + sift-down for the brute-force path.
 * Guards with an early exit if d2 ≥ heap root, then replaces the root and sifts
 * down. Depth is fixed at 7 iterations (covers K up to 128). Same algorithm as
 * insert_neighbor_t in find_nbrs.cu but without the CAP template (used here
 * because the brute-force kernel was not originally templated on K).
 *
 * Key variables:
 *   local_dists[0] — heap root; the current worst (largest) accepted distance
 *   d2             — candidate squared distance; only inserted if d2 < local_dists[0]
 *   i              — current node index during sift-down toward the leaves
 */
__device__ void bf_insert_neighbor(float* local_dists, int* local_idxs, int K, float d2, int idx2) {
    if (d2 >= local_dists[0]) return;

    local_dists[0] = d2;
    local_idxs[0]  = idx2;

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

/*
 * TiledBruteforceNDKernel<CAP> — generic N-D tiled brute-force kernel.
 * Each block cooperatively loads BLOCKDIM reference points into shared memory
 * (AoS layout, one float3..float128 row per point), then every query thread
 * scans the tile and calls bf_dist2 for each entry. Slides the tile window
 * across all P2 reference points with two __syncthreads() barriers per tile.
 *
 * Key variables:
 *   tile[]          — shared memory buffer: blockDim.x points × dim coords, AoS,
 *                     16-byte aligned so bf_dist2 can use float4 loads
 *   q[BF_MAX_DIM]   — register-cached query coords (SoA global → contiguous local,
 *                     loaded once before the tile loop)
 *   local_dists[CAP]— per-thread K-heap sized to CAP (not 128), holding K best dists
 */
template<int CAP>
__global__ void TiledBruteforceNDKernel(
    const float* __restrict__ p1,
    const float* __restrict__ p2,
    int P1, int P2, int K, int dim, float r2,
    float* __restrict__ dists,
    int*   __restrict__ idxs)
{
    // Shared memory tile: blockDim.x reference points × dim coords.
    // __align__(16) so the AoS rows can be read with float4 (see bf_dist2).
    extern __shared__ __align__(16) float tile[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_dists[CAP];
    int   local_idxs[CAP];
    for (int k = 0; k < K; k++) { local_dists[k] = r2; local_idxs[k] = -1; }

    // Cache this thread's query point once: SoA global -> contiguous, 16-byte
    // aligned local buffer. The tile loop then reads it from L1 instead of
    // re-fetching strided global memory on every tile element, and bf_dist2 can
    // load it as float4. Coalesced load: for fixed d, consecutive threads read
    // p1[d*P1 + i] at consecutive addresses.
    __align__(16) float q[BF_MAX_DIM];
    if (i < P1)
        for (int d = 0; d < dim; d++) q[d] = p1[d * P1 + i];

    // Slide tile window over all reference points
    for (int t0 = 0; t0 < P2; t0 += blockDim.x) {

        // Cooperatively load tile: thread tx loads reference point (t0 + tx)
        int j = t0 + threadIdx.x;
        if (j < P2)
            for (int d = 0; d < dim; d++)
                tile[threadIdx.x * dim + d] = p2[d * P2 + j];  // SoA global load; tile stays AoS in smem
        __syncthreads();

        // Every query thread scans the loaded tile from shared memory
        if (i < P1) {
            int tile_n = min((int)blockDim.x, P2 - t0);
            for (int tj = 0; tj < tile_n; tj++) {
                float d2 = bf_dist2(q, &tile[tj * dim], dim);  // float4-vectorized
                if (d2 < r2)
                    bf_insert_neighbor(local_dists, local_idxs, K, d2, t0 + tj);
            }
        }
        __syncthreads();
    }

    if (i < P1) {
        #pragma unroll 16
        for (int k = 0; k < K; k++) {
            dists[k * P1 + i] = local_dists[k];
            idxs[k * P1 + i]  = local_idxs[k];
        }
    }
}

/*
 * TiledBruteforce16Kernel<CAP> — D=16 specialization of the tiled brute-force kernel.
 * Same tile-slide scheme as TiledBruteforceNDKernel, but dimension 16 is baked in at
 * compile time for three micro-architectural wins: (1) the 16-coord query is held in
 * 4 float4 registers (q0..q3) instead of a 128-float local array, eliminating L1
 * local-memory re-reads on every tile entry; (2) the distance is fully unrolled to 4
 * float4 subtract-accumulate groups with no loop or dim%4 branch; (3) the tile stride
 * is the constant 16, simplifying address arithmetic. The `dim` arg is ignored.
 *
 * Key variables:
 *   q0..q3   — four float4 registers holding all 16 query coords; never re-fetched
 *   t4[]     — tile entry reinterpreted as float4 pointer; 4× LDS.128 from smem
 *   tile[]   — shared memory buffer, AoS at stride 16 (blockDim.x × 16 floats)
 */
template<int CAP>
__global__ void TiledBruteforce16Kernel(
    const float* __restrict__ p1,
    const float* __restrict__ p2,
    int P1, int P2, int K, int dim, float r2,
    float* __restrict__ dists,
    int*   __restrict__ idxs)
{
    constexpr int DIM = 16;
    extern __shared__ __align__(16) float tile[];   // blockDim.x rows × 16 coords, AoS

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_dists[CAP];
    int   local_idxs[CAP];
    for (int k = 0; k < K; k++) { local_dists[k] = r2; local_idxs[k] = -1; }

    // Query in registers: 16 coords = 4 float4. SoA global -> for fixed d, consecutive
    // threads read p1[d*P1 + i] at consecutive addresses (coalesced).
    float4 q0, q1, q2, q3;
    if (i < P1) {
        q0 = make_float4(p1[ 0*P1+i], p1[ 1*P1+i], p1[ 2*P1+i], p1[ 3*P1+i]);
        q1 = make_float4(p1[ 4*P1+i], p1[ 5*P1+i], p1[ 6*P1+i], p1[ 7*P1+i]);
        q2 = make_float4(p1[ 8*P1+i], p1[ 9*P1+i], p1[10*P1+i], p1[11*P1+i]);
        q3 = make_float4(p1[12*P1+i], p1[13*P1+i], p1[14*P1+i], p1[15*P1+i]);
    }

    for (int t0 = 0; t0 < P2; t0 += blockDim.x) {
        int j = t0 + threadIdx.x;
        if (j < P2)
            #pragma unroll
            for (int d = 0; d < DIM; d++)
                tile[threadIdx.x * DIM + d] = p2[d * P2 + j];   // SoA global load; tile is AoS
        __syncthreads();

        if (i < P1) {
            int tile_n = min((int)blockDim.x, P2 - t0);
            for (int tj = 0; tj < tile_n; tj++) {
                // &tile[tj*16] is 16-byte aligned (tj*64 bytes) -> 4× LDS.128 from smem.
                const float4* t4 = reinterpret_cast<const float4*>(&tile[tj * DIM]);
                float4 b0 = t4[0], b1 = t4[1], b2 = t4[2], b3 = t4[3];
                float dx, dy, dz, dw, d2;
                dx = q0.x-b0.x; dy = q0.y-b0.y; dz = q0.z-b0.z; dw = q0.w-b0.w;
                d2  = dx*dx + dy*dy + dz*dz + dw*dw;
                dx = q1.x-b1.x; dy = q1.y-b1.y; dz = q1.z-b1.z; dw = q1.w-b1.w;
                d2 += dx*dx + dy*dy + dz*dz + dw*dw;
                dx = q2.x-b2.x; dy = q2.y-b2.y; dz = q2.z-b2.z; dw = q2.w-b2.w;
                d2 += dx*dx + dy*dy + dz*dz + dw*dw;
                dx = q3.x-b3.x; dy = q3.y-b3.y; dz = q3.z-b3.z; dw = q3.w-b3.w;
                d2 += dx*dx + dy*dy + dz*dz + dw*dw;
                if (d2 < r2)
                    bf_insert_neighbor(local_dists, local_idxs, K, d2, t0 + tj);
            }
        }
        __syncthreads();
    }

    if (i < P1) {
        #pragma unroll 16
        for (int k = 0; k < K; k++) {
            dists[k * P1 + i] = local_dists[k];
            idxs[k * P1 + i]  = local_idxs[k];
        }
    }
}

/*
 * run_bruteforce — host wrapper that selects block size and dispatches the tiled kernel.
 * Computes the largest block size that keeps shared memory within the device limit and
 * enough blocks to fill all 80 A100 SMs. For D=16 floors at 256 threads (128 is
 * suboptimal there — too little reference reuse per tile load). Opts into the A100's
 * larger dynamic shared-memory limit when the tile exceeds the default 48 KB.
 * Dispatches TiledBruteforce16Kernel for dim==16, TiledBruteforceNDKernel otherwise,
 * both with the smallest compile-time CAP that holds K.
 *
 * Key variables:
 *   threads — block size; controls tile width (= reference reuse per tile fetch)
 *   smem    — shared memory per block = threads × dim × 4 bytes; the tile size
 *   DISPATCH_BF / LAUNCH_BF — two-axis dispatch macros: K-cap × kernel variant
 */
extern "C" void run_bruteforce(
    const float* d_p1, const float* d_p2,
    int P1, int P2, int K, int dim, float r,
    float* d_dists, int* d_idxs)
{
    int threads = 512;        // measured best at D16 (2.3x over 128 at N=200K)
    // Shrink at small N to keep the SMs filled, but floor at 256 for D=16 — 128 is
    // suboptimal there (too little reference reuse per tile load).
    int min_threads = (dim == 16) ? 256 : 128;
    while (threads > min_threads && (P1 + threads - 1) / threads < 80) threads >>= 1;
    if (const char* e = std::getenv("FRNN_BF_THREADS")) {
        int t = std::atoi(e);
        if (t >= 32 && t <= 1024) threads = t;
    }

    int dev = 0, smem_max = 48 * 1024;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int max_threads = smem_max / (int)(dim * sizeof(float));   // tile must fit shared mem
    if (threads > max_threads) threads = max_threads;
    threads = (threads / 32) * 32;
    if (threads < 32) threads = 32;

    int    blocks = (P1 + threads - 1) / threads;
    float  r2     = r * r;
    size_t smem   = (size_t)threads * dim * sizeof(float);

    // Dispatch on two compile-time axes: the kernel (D=16 specialization vs generic
    // runtime-dim) and the smallest heap capacity that holds K. For tiles over 48 KB, opt
    // the chosen instantiation into the A100's larger dynamic shared-memory limit.
    #define LAUNCH_BF(KERNEL, CAP) do {                                     \
        if (smem > 48u * 1024u)                                            \
            cudaFuncSetAttribute(KERNEL<CAP>,                              \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);   \
        KERNEL<CAP><<<blocks, threads, smem>>>(                           \
            d_p1, d_p2, P1, P2, K, dim, r2, d_dists, d_idxs);             \
    } while (0)
    #define DISPATCH_BF(CAP) do {                                          \
        if (dim == 16) LAUNCH_BF(TiledBruteforce16Kernel, CAP);           \
        else           LAUNCH_BF(TiledBruteforceNDKernel, CAP);           \
    } while (0)
    if      (K <= 16)  DISPATCH_BF(16);
    else if (K <= 32)  DISPATCH_BF(32);
    else if (K <= 64)  DISPATCH_BF(64);
    else               DISPATCH_BF(128);
    #undef DISPATCH_BF
    #undef LAUNCH_BF

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::fprintf(stderr, "[run_bruteforce] launch failed (threads=%d, smem=%zu): %s\n",
                     threads, smem, cudaGetErrorString(err));
    cudaDeviceSynchronize();
}
