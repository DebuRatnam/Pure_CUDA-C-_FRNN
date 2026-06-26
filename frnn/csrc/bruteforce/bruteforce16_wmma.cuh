#pragma once
// Tensor-core (WMMA) fused FRNN for D=16. The expensive O(N^2 * D) term in the
// squared-L2 distance is the Gram dot product q . t; everything else is O(N*D).
// So instead of the direct CUDA-core sum_(q-t)^2, factor:
//
//     d2(i,j) = ||qi||^2 + ||tj||^2 - 2 (qi . tj)
//
// and compute the (qi . tj) block G = Q . T^T on the A100 tensor cores (TF32),
// which deliver ~8-16x the FMA throughput of the CUDA cores. This is the only
// lever that closes the gap to FlashLib/FAISS at large N where D=16 brute-force
// is compute-bound (the RM register-blocked kernel showed SMEM/registers are NOT
// the bottleneck -> 0-spill, no speedup -> it's the FMAs).
//
// Fusion (FlashAttention-style): the N x N Gram matrix is never materialized in
// HBM. Each block computes a 16 x 16 G tile in tensor-core fragments, adds the
// precomputed norms, radius-filters, and feeds survivors straight into the
// per-query max-heap top-K -- the G tile lives in SMEM/registers and dies there.
//
// Precision: TF32 keeps only a 10-bit mantissa, so the GEMM dot product carries
// ~1.5e-2 absolute error for unit-cube coords at D=16. That is fine for *finding*
// candidates, but the in/out radius decision and the reported distances must be
// exact. So every kept neighbor's distance is RECOMPUTED with the exact CUDA-core
// sum_(q-t)^2 from global memory (only K per query -> cheap), and any candidate
// whose exact distance lands >= r^2 is dropped. Output distances are therefore
// bitwise the same exact fp32 the direct kernel produces; only the in/out call at
// the radius boundary (|d2 - r2| < tf32 eps) can differ -- the band the validator
// already treats as ambiguous.
//
// Opt-in via FRNN_BF16_WMMA=1 in run_bruteforce(); default path untouched.

#include <mma.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace frnn_bf16_wmma {

using namespace nvcuda;

// Per-query max-heap insert (mirrors bf_insert_neighbor in bruteforce.cu): root
// ld[0] is the current K-th-nearest; replace + sift down when a closer point lands.
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

// ||point||^2 per point, SoA input p[d*P + i]. One thread per point.
__global__ void sqnorms_kernel(const float* __restrict__ p, int P, int dim,
                               float* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    float s = 0.0f;
    for (int d = 0; d < dim; d++) { float v = p[d * P + i]; s += v * v; }
    out[i] = s;
}

// One warp (32 lanes) per block; the block owns 16 consecutive query rows. Lanes
// 0..15 each own one query and its top-K heap; all 32 lanes cooperate on the
// tensor-core GEMM and the SMEM tile staging.
//
// WMMA shape 16x16x8 (TF32 on sm_80). D=16 -> the K-axis (=dim) is two 8-wide
// mma steps. To get G = Q . T^T with Q,T stored row-major [point][dim] in SMEM,
// load Q as row_major matrix_a and T as col_major matrix_b at the SAME pointer:
// a row-major [N][K] buffer read col_major [K][N] is exactly its transpose.
template<int CAP>
__global__ void wmma_frnn16_kernel(
    const float* __restrict__ p1, const float* __restrict__ p2,
    const float* __restrict__ sqn1, const float* __restrict__ sqn2,
    int P1, int P2, int K, float r2,
    float* __restrict__ dists, int* __restrict__ idxs)
{
    constexpr int DIM = 16, M = 16, NT = 16;
    __shared__ float sQ[M * DIM];    // query tile, row-major [m][d]
    __shared__ float sT[NT * DIM];   // ref   tile, row-major [n][d]
    __shared__ float sG[M * NT];     // Gram tile G[m][n] = q_m . t_n

    const int lane = threadIdx.x;          // 0..31
    const int m0   = blockIdx.x * M;        // first query row of this block

    // Stage the query tile once (queries are fixed for the whole block).
    for (int e = lane; e < M * DIM; e += 32) {
        int r = e / DIM, d = e % DIM, gi = m0 + r;
        sQ[e] = (gi < P1) ? p1[d * P1 + gi] : 0.0f;   // SoA global -> row-major smem
    }

    // Per-lane query heap (lanes 0..15 active). Pad with r2 / -1 like the direct kernel.
    float ld[CAP];
    int   li[CAP];
    const bool active = (lane < M) && (m0 + lane < P1);
    if (lane < M) { for (int k = 0; k < K; k++) { ld[k] = r2; li[k] = -1; } }
    const float qn = active ? sqn1[m0 + lane] : 0.0f;
    __syncthreads();

    // Slide the reference window over all P2 points, 16 refs at a time.
    for (int n0 = 0; n0 < P2; n0 += NT) {
        for (int e = lane; e < NT * DIM; e += 32) {
            int r = e / DIM, d = e % DIM, gj = n0 + r;
            sT[e] = (gj < P2) ? p2[d * P2 + gj] : 0.0f;
        }
        __syncthreads();

        // Tensor-core GEMM: G(16x16) = Q(16x16) . T(16x16)^T, fp32 accumulate.
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int kd = 0; kd < DIM; kd += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b;
            wmma::load_matrix_sync(a, sQ + kd, DIM);   // row-major Q, k-subtile at col kd
            wmma::load_matrix_sync(b, sT + kd, DIM);   // col-major read of row-major T = T^T
            #pragma unroll
            for (int t = 0; t < a.num_elements; t++) a.x[t] = wmma::__float_to_tf32(a.x[t]);
            #pragma unroll
            for (int t = 0; t < b.num_elements; t++) b.x[t] = wmma::__float_to_tf32(b.x[t]);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(sG, acc, NT, wmma::mem_row_major);   // sG[m*NT + n]
        __syncthreads();

        // Fuse: norms + radius + top-K. Lane m owns query row m; scan the 16 refs.
        if (active) {
            #pragma unroll
            for (int jj = 0; jj < NT; jj++) {
                int gj = n0 + jj;
                if (gj >= P2) continue;
                // d2 = ||q||^2 + ||t||^2 - 2 q.t  (TF32 q.t; exact recompute below).
                float d2 = qn + sqn2[gj] - 2.0f * sG[lane * NT + jj];
                if (d2 < 0.0f) d2 = 0.0f;
                if (d2 < r2) insert_neighbor(ld, li, K, d2, gj);
            }
        }
        __syncthreads();
    }

    // Exact recompute of every kept neighbor (direct sum_(q-t)^2 from global, the
    // same expression the default kernel uses) -> exact output distances. Drop any
    // candidate whose exact distance is actually >= r^2 (TF32 false positive).
    if (active) {
        const int qi = m0 + lane;
        for (int k = 0; k < K; k++) {
            int j = li[k];
            if (j >= 0) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    float diff = p1[d * P1 + qi] - p2[d * P2 + j];
                    s += diff * diff;
                }
                if (s < r2) { ld[k] = s; }
                else        { ld[k] = r2; li[k] = -1; }   // drop false positive
            }
            dists[k * P1 + qi] = ld[k];
            idxs[k * P1 + qi]  = li[k];
        }
    }
}

// Host launcher: precompute norms, dispatch the smallest heap capacity holding K.
// SMEM is static (3 * 16 * 16 * 4 = 3072 B), well under 48 KB -> no opt-in needed.
inline void run_bruteforce16_wmma(
    const float* d_p1, const float* d_p2,
    int P1, int P2, int K, float r2,
    float* d_dists, int* d_idxs)
{
    float *d_sqn1 = nullptr, *d_sqn2 = nullptr;
    cudaMalloc(&d_sqn1, sizeof(float) * P1);
    cudaMalloc(&d_sqn2, sizeof(float) * P2);

    const int tb = 256;
    sqnorms_kernel<<<(P1 + tb - 1) / tb, tb>>>(d_p1, P1, 16, d_sqn1);
    sqnorms_kernel<<<(P2 + tb - 1) / tb, tb>>>(d_p2, P2, 16, d_sqn2);

    const int blocks = (P1 + 15) / 16;     // 16 queries per block
    #define LAUNCH_W(CAP) \
        wmma_frnn16_kernel<CAP><<<blocks, 32>>>( \
            d_p1, d_p2, d_sqn1, d_sqn2, P1, P2, K, r2, d_dists, d_idxs)
    if      (K <= 16) LAUNCH_W(16);
    else if (K <= 32) LAUNCH_W(32);
    else if (K <= 64) LAUNCH_W(64);
    else              LAUNCH_W(128);
    #undef LAUNCH_W

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        std::fprintf(stderr, "[run_bruteforce/wmma] launch failed (P1=%d P2=%d K=%d): %s\n",
                     P1, P2, K, cudaGetErrorString(e));
    cudaDeviceSynchronize();

    cudaFree(d_sqn1);
    cudaFree(d_sqn2);
}

} // namespace frnn_bf16_wmma
