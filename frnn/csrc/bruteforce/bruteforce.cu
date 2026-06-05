#include "bruteforce.h"
#include <device_launch_parameters.h>
#include <float.h>

// Engine hard limit (D <= 128); matches MAX_DIM_SUPPORTED in grid.h.
constexpr int BF_MAX_DIM = 128;

// Squared L2 distance between a query point q and a tile point t, both laid out
// contiguously over the `dim` axis. When dim is a multiple of 4 (the D=8/16
// brute-force cases) we issue 128-bit float4 loads — one LDS.128 per 4 coords
// instead of four LDS.32 — and accumulate four squared diffs per iteration.
// Both operands are 16-byte aligned in that path: q is __align__(16), and
// &tile[tj*dim] is a multiple of 16 bytes because dim*4 is. Non-multiple-of-4
// dims fall back to the scalar loop so the kernel stays correct for any D.
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

// Tiled brute-force: each block loads a tile of reference points into shared
// memory once, then all query threads in the block reuse it.
// This reduces global memory traffic by a factor of blockDim.x compared to
// the naive per-thread approach.
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

    float local_dists[128];
    int   local_idxs[128];
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

extern "C" void run_bruteforce(
    const float* d_p1, const float* d_p2,
    int P1, int P2, int K, int dim, float r,
    float* d_dists, int* d_idxs)
{
    // 128 threads → smem = 128 × dim × 4 bytes
    // For dim=16: 8 KB/block → fits 6 blocks/SM on A100 (48 KB default smem).
    // A block-size sweep on the A100 confirmed 128 is best/tied for all cases
    // except very large high-D (D16/N100K), where it costs ~8%; smaller blocks
    // win at N≤10K because larger ones can't fill the 108 SMs.
    int   threads = 128;
    int   blocks  = (P1 + threads - 1) / threads;
    float r2      = r * r;
    size_t smem   = (size_t)threads * dim * sizeof(float);

    TiledBruteforceNDKernel<<<blocks, threads, smem>>>(
        d_p1, d_p2, P1, P2, K, dim, r2, d_dists, d_idxs);
    cudaDeviceSynchronize();
}
