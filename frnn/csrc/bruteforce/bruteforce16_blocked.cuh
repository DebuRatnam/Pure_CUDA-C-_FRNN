#pragma once
// Register-blocked D=16 tiled brute-force: each thread owns RM query points
// instead of one. The shared-memory reference tile is loaded once per thread
// and reused across all RM queries held in registers, so per-query SMEM
// reference traffic drops by a factor of RM. Pure CUDA-core FMA, exact fp32,
// no tensor cores, no x^2/cross reformulation -> no catastrophic cancellation
// at the radius boundary. Fusion + per-query max-heap top-K unchanged.
//
// Tradeoff (engine handbook diagnostics B/C): RM queries -> RM heaps in
// registers. At CAP=16 a single heap is 32 regs (dist+idx); RM=2 doubles it.
// Keep RM small (2 default, 4 only for K<=8) and check ptxas register count.
//
// Drop-in companion to TiledBruteforce16Kernel in bruteforce.cu. Wire it into
// run_bruteforce() via the snippet at the bottom of this file.

#include <device_launch_parameters.h>
#include <float.h>

namespace frnn_bf16_blocked {

// Per-query max-heap insert (mirrors bf_insert_neighbor in bruteforce.cu).
// Operates on this query's slice of the per-thread heap arrays.
__device__ __forceinline__ void insert_neighbor(
    float* __restrict__ ld, int* __restrict__ li, int K, float d2, int idx2)
{
    if (d2 >= ld[0]) return;
    ld[0] = d2;
    li[0] = idx2;
    int i = 0;
    #pragma unroll 7
    for (int depth = 0; depth < 7; depth++) {
        int left = 2 * i + 1, right = 2 * i + 2, largest = i;
        if (left  < K && ld[left]  > ld[largest]) largest = left;
        if (right < K && ld[right] > ld[largest]) largest = right;
        if (largest == i) break;
        float td = ld[i]; ld[i] = ld[largest]; ld[largest] = td;
        int   ti = li[i]; li[i] = li[largest]; li[largest] = ti;
        i = largest;
    }
}

// RM = queries per thread (register-blocking factor). CAP = compile-time heap
// capacity (>= K), keeps the heap arrays sized to the dispatched K.
//
// Query layout for coalescing: the block owns blockDim.x * RM consecutive
// queries, partitioned into RM groups of blockDim.x. Thread tx slot m maps to
//   q = blockIdx.x * (blockDim.x * RM) + m * blockDim.x + tx
// so for fixed (m, d) consecutive threads read p1[d*P1 + q] at consecutive
// addresses -> coalesced SoA load.
template<int RM, int CAP>
__global__ void TiledBruteforce16BlockedKernel(
    const float* __restrict__ p1,
    const float* __restrict__ p2,
    int P1, int P2, int K, int dim, float r2,
    float* __restrict__ dists,
    int*   __restrict__ idxs)
{
    constexpr int DIM = 16;
    extern __shared__ __align__(16) float tile[];   // blockDim.x rows x 16 coords, AoS

    const int tx        = threadIdx.x;
    const int q_base    = blockIdx.x * (blockDim.x * RM);

    // This thread's RM global query indices.
    int qg[RM];
    #pragma unroll
    for (int m = 0; m < RM; m++) qg[m] = q_base + m * blockDim.x + tx;

    // RM queries in registers (4 float4 each), and RM independent heaps.
    float4 q0[RM], q1[RM], q2[RM], q3[RM];
    float  ld[RM][CAP];
    int    li[RM][CAP];

    #pragma unroll
    for (int m = 0; m < RM; m++) {
        for (int k = 0; k < K; k++) { ld[m][k] = r2; li[m][k] = -1; }
        if (qg[m] < P1) {
            const int q = qg[m];
            q0[m] = make_float4(p1[ 0*P1+q], p1[ 1*P1+q], p1[ 2*P1+q], p1[ 3*P1+q]);
            q1[m] = make_float4(p1[ 4*P1+q], p1[ 5*P1+q], p1[ 6*P1+q], p1[ 7*P1+q]);
            q2[m] = make_float4(p1[ 8*P1+q], p1[ 9*P1+q], p1[10*P1+q], p1[11*P1+q]);
            q3[m] = make_float4(p1[12*P1+q], p1[13*P1+q], p1[14*P1+q], p1[15*P1+q]);
        }
    }

    // Slide tile window over all reference points.
    for (int t0 = 0; t0 < P2; t0 += blockDim.x) {
        // Cooperative tile load: thread tx loads reference point (t0 + tx).
        int j = t0 + tx;
        if (j < P2)
            #pragma unroll
            for (int d = 0; d < DIM; d++)
                tile[tx * DIM + d] = p2[d * P2 + j];   // SoA global -> AoS smem
        __syncthreads();

        int tile_n = min((int)blockDim.x, P2 - t0);
        for (int tj = 0; tj < tile_n; tj++) {
            // Load this reference ONCE from smem; reuse across all RM queries.
            // &tile[tj*16] is 16-byte aligned (tj*64 bytes) -> 4x LDS.128.
            const float4* t4 = reinterpret_cast<const float4*>(&tile[tj * DIM]);
            const float4 b0 = t4[0], b1 = t4[1], b2 = t4[2], b3 = t4[3];
            const int ref_idx = t0 + tj;

            #pragma unroll
            for (int m = 0; m < RM; m++) {
                if (qg[m] >= P1) continue;
                float dx, dy, dz, dw, d2;
                dx = q0[m].x-b0.x; dy = q0[m].y-b0.y; dz = q0[m].z-b0.z; dw = q0[m].w-b0.w;
                d2  = dx*dx + dy*dy + dz*dz + dw*dw;
                dx = q1[m].x-b1.x; dy = q1[m].y-b1.y; dz = q1[m].z-b1.z; dw = q1[m].w-b1.w;
                d2 += dx*dx + dy*dy + dz*dz + dw*dw;
                dx = q2[m].x-b2.x; dy = q2[m].y-b2.y; dz = q2[m].z-b2.z; dw = q2[m].w-b2.w;
                d2 += dx*dx + dy*dy + dz*dz + dw*dw;
                dx = q3[m].x-b3.x; dy = q3[m].y-b3.y; dz = q3[m].z-b3.z; dw = q3[m].w-b3.w;
                d2 += dx*dx + dy*dy + dz*dz + dw*dw;
                if (d2 < r2)
                    insert_neighbor(ld[m], li[m], K, d2, ref_idx);
            }
        }
        __syncthreads();
    }

    // Write RM queries' results (SoA output: dists[k*P1 + q]).
    #pragma unroll
    for (int m = 0; m < RM; m++) {
        if (qg[m] >= P1) continue;
        const int q = qg[m];
        #pragma unroll 16
        for (int k = 0; k < K; k++) {
            dists[k * P1 + q] = ld[m][k];
            idxs[k * P1 + q]  = li[m][k];
        }
    }
}

} // namespace frnn_bf16_blocked

// ---------------------------------------------------------------------------
// Wiring into run_bruteforce() in bruteforce.cu (dim == 16 path):
//
//   #include "bruteforce16_blocked.cuh"
//   using frnn_bf16_blocked::TiledBruteforce16BlockedKernel;
//
//   constexpr int RM = 2;  // queries per thread; try 4 only for K <= 8
//   int blocks = (P1 + threads * RM - 1) / (threads * RM);   // <-- divide by RM
//   size_t smem = (size_t)threads * dim * sizeof(float);     // tile size unchanged
//
//   #define LAUNCH_BFB(CAP) do {                                              \
//       if (smem > 48u*1024u)                                                 \
//           cudaFuncSetAttribute(TiledBruteforce16BlockedKernel<RM,CAP>,      \
//               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);      \
//       TiledBruteforce16BlockedKernel<RM,CAP><<<blocks, threads, smem>>>(    \
//           d_p1, d_p2, P1, P2, K, dim, r2, d_dists, d_idxs);                 \
//   } while (0)
//   if      (K <= 16) LAUNCH_BFB(16);
//   else if (K <= 32) LAUNCH_BFB(32);
//   else if (K <= 64) LAUNCH_BFB(64);
//   else              LAUNCH_BFB(128);
//
// Notes:
//   * grid divides by RM because each thread now covers RM queries. The block
//     still loads `threads` refs into smem; tile size and smem are unchanged.
//   * Verify exactness against the existing kernel: same (P1,P2,K,r) must give
//     identical dists/idxs (radius search is deterministic up to tie order).
//   * Tune: ptxas register count must stay < 64/thread for decent occupancy.
//       nvcc -arch=sm_80 --ptxas-options=-v -c bruteforce16_blocked.cuh ...
//     If RM=2 + CAP=16 spills, drop threads (e.g. 512 -> 256) or keep RM=2 and
//     accept lower occupancy; measure sm__warps_active and HBM %peak with ncu.
// ---------------------------------------------------------------------------
