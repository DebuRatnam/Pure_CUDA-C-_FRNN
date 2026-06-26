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
// Precision: TF32 keeps only a 10-bit mantissa, so the TF32 dot product carries
// ~4e-3 absolute error on d2 for unit-cube coords at D=16. That is fine for
// *finding* candidates but NOT for ranking them: a K-capacity heap ordered by the
// approximate d2 can evict the true K-th nearest in favour of a slightly-farther
// point whenever the TF32 error exceeds the gap between the K-th and (K+1)-th true
// distances -- which is tiny at a dense radius boundary (the exact failure mode of
// the first version). Recomputing only the survivors cannot recover a true
// neighbor that was already dropped.
//
// Fix (candidate over-generation + exact rerank): gen into an OVERSIZED heap of
// capacity GCAP = ~2K at a RELAXED radius RR2 = r2*(1+slack), so every true top-K
// neighbor survives the approximate reshuffling (the count actually within r2 is
// ~K by radius construction, so GCAP=2K covers it). Then RECOMPUTE all GCAP
// candidates with the exact CUDA-core sum_(q-t)^2 from global memory, keep those
// with exact d2 < r2, and select the true K nearest by exact distance. Output
// distances are therefore the same exact fp32 the direct kernel produces, and the
// neighbor *set* matches except at the literal radius boundary (the band the
// validator treats as ambiguous).
//
// Opt-in via FRNN_BF16_WMMA=1 in run_bruteforce(); default path untouched.

#include <mma.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

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

// GEMM-blocked: each block owns TILE_M = WARPS*16 consecutive query rows; one
// warp per 16-row M-subtile. ALL warps stream the SAME ref tile from SMEM, so a
// 16-ref window is loaded from global once per block and reused WARPS times --
// global ref traffic drops from N^2 (one-warp-per-16-queries) to N^2/WARPS, the
// dominant cost at large N. Within each warp lanes 0..15 own one query + its
// top-K heap; all 32 lanes cooperate on the tensor-core GEMM and tile staging.
//
// Query fragments are loaded + TF32-converted ONCE before the ref loop and held
// in registers (queries are fixed for the block), so only the ref fragments are
// (re)loaded/converted per 16-ref window.
//
// WMMA shape 16x16x8 (TF32 on sm_80). D=16 -> the K-axis (=dim) is two 8-wide
// mma steps. To get G = Q . T^T with Q,T stored row-major [point][dim] in SMEM,
// load Q as row_major matrix_a and T as col_major matrix_b at the SAME pointer:
// a row-major [N][K] buffer read col_major [K][N] is exactly its transpose.
//
// GCAP = gen-heap capacity (compile-time, ~2K) -> slack so TF32 reshuffling near
// the radius boundary cannot evict the true top-K before the exact rerank.
template<int GCAP>
__global__ void wmma_frnn16_kernel(
    const float* __restrict__ p1, const float* __restrict__ p2,
    const float* __restrict__ sqn1, const float* __restrict__ sqn2,
    int P1, int P2, int K, float r2,
    float* __restrict__ dists, int* __restrict__ idxs)
{
    constexpr int DIM = 16, M = 16, NT = 16, WARPS = 8, TILE_M = WARPS * M; // 128
    __shared__ float sQ[TILE_M * DIM];   // query tile, row-major [m][d]   (8 KB)
    __shared__ float sT[NT * DIM];       // ref   tile, row-major [n][d]   (1 KB)
    __shared__ float sG[TILE_M * NT];    // Gram tiles G[m][n] = q_m . t_n (8 KB)

    const int tid  = threadIdx.x;          // 0..255
    const int warp = tid >> 5;             // 0..7  -> M-subtile
    const int lane = tid & 31;             // 0..31
    const int m0   = blockIdx.x * TILE_M;  // first query row of this block

    // Relaxed gen radius: keep approximate candidates a bit past r2 so a true
    // in-radius neighbor whose TF32 d2 overshoots is still captured. 5% + 1e-3
    // dwarfs the ~4e-3 TF32 d2 error; the exact rerank below re-imposes r2.
    const float RR2 = r2 * 1.05f + 1e-3f;

    // Stage the whole query tile once (queries are fixed for the block).
    for (int e = tid; e < TILE_M * DIM; e += blockDim.x) {
        int r = e / DIM, d = e % DIM, gi = m0 + r;
        sQ[e] = (gi < P1) ? p1[d * P1 + gi] : 0.0f;   // SoA global -> row-major smem
    }
    __syncthreads();

    // Persistent query fragments for this warp's 16 rows: load + TF32 once, reuse
    // across the entire ref loop (only the ref fragments change per window).
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> qa[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        wmma::load_matrix_sync(qa[s], sQ + warp * M * DIM + s * 8, DIM);
        #pragma unroll
        for (int t = 0; t < qa[s].num_elements; t++) qa[s].x[t] = wmma::__float_to_tf32(qa[s].x[t]);
    }

    // Per-lane oversized gen heap (lanes 0..15 of each warp active). RR2 / -1 pad.
    const int  qrow   = warp * M + lane;                 // local query row 0..127
    const int  qi     = m0 + qrow;                       // global query
    const bool active = (lane < M) && (qi < P1);
    float ld[GCAP];
    int   li[GCAP];
    if (lane < M) { for (int g = 0; g < GCAP; g++) { ld[g] = RR2; li[g] = -1; } }
    const float qn = active ? sqn1[qi] : 0.0f;

    // Slide the reference window over all P2 points, 16 refs at a time.
    for (int n0 = 0; n0 < P2; n0 += NT) {
        for (int e = tid; e < NT * DIM; e += blockDim.x) {
            int r = e / DIM, d = e % DIM, gj = n0 + r;
            sT[e] = (gj < P2) ? p2[d * P2 + gj] : 0.0f;
        }
        __syncthreads();

        // Tensor-core GEMM: this warp's G(16x16) = Q_warp(16x16) . T(16x16)^T.
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b;
            wmma::load_matrix_sync(b, sT + s * 8, DIM);   // col-major read of row-major T = T^T
            #pragma unroll
            for (int t = 0; t < b.num_elements; t++) b.x[t] = wmma::__float_to_tf32(b.x[t]);
            wmma::mma_sync(acc, qa[s], b, acc);
        }
        wmma::store_matrix_sync(sG + warp * M * NT, acc, NT, wmma::mem_row_major);
        __syncthreads();

        // Fuse: norms + relaxed-radius into the oversized gen heap. Lane owns query
        // row qrow; scan the 16 refs. Ordering uses the TF32 d2 (approximate),
        // hence the GCAP slack + relaxed RR2 so no true top-K member is evicted.
        if (active) {
            #pragma unroll
            for (int jj = 0; jj < NT; jj++) {
                int gj = n0 + jj;
                if (gj >= P2) continue;
                // d2 = ||q||^2 + ||t||^2 - 2 q.t  (TF32 q.t; exact rerank below).
                float d2 = qn + sqn2[gj] - 2.0f * sG[qrow * NT + jj];
                if (d2 < 0.0f) d2 = 0.0f;
                if (d2 < RR2) insert_neighbor(ld, li, GCAP, d2, gj);
            }
        }
        __syncthreads();
    }

    // Exact rerank. Recompute every GCAP candidate with the direct sum_(q-t)^2
    // (the exact expression the default kernel uses); keep exact d2 < r2, mark the
    // rest dead (+inf). Then selection-sort the K nearest by EXACT distance into
    // the output -- this is what recovers the true top-K from the over-generated,
    // approximately-ordered candidate pool.
    if (active) {
        #pragma unroll
        for (int g = 0; g < GCAP; g++) {
            int j = li[g];
            if (j >= 0) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    float diff = p1[d * P1 + qi] - p2[d * P2 + j];
                    s += diff * diff;
                }
                ld[g] = (s < r2) ? s : FLT_MAX;   // exact; drop if outside r2
                if (s >= r2) li[g] = -1;
            } else {
                ld[g] = FLT_MAX;                  // empty slot
            }
        }
        // Pull the K smallest exact distances in ascending order. Padding when the
        // query has < K in-radius neighbors is (r2, -1), matching the direct kernel.
        for (int k = 0; k < K; k++) {
            float best = FLT_MAX;
            int   bg   = -1;
            for (int g = 0; g < GCAP; g++) {
                if (ld[g] < best) { best = ld[g]; bg = g; }
            }
            if (bg >= 0 && best < r2) {
                dists[k * P1 + qi] = best;
                idxs[k * P1 + qi]  = li[bg];
                ld[bg] = FLT_MAX;                 // consume
            } else {
                dists[k * P1 + qi] = r2;
                idxs[k * P1 + qi]  = -1;
            }
        }
    }
}

// Host launcher: precompute norms, dispatch the smallest heap capacity holding K.
// SMEM is static (sQ 8 KB + sT 1 KB + sG 8 KB = 17 KB), under 48 KB -> no opt-in.
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

    const int blocks = (P1 + 127) / 128;   // TILE_M = 128 queries per block (8 warps)
    // GCAP = gen-heap capacity ~2K (slack so TF32 reshuffling can't evict the true
    // top-K), capped at 128 = engine K limit. K<=64 still gets 2x slack; K<=128 has
    // none (the boundary is then far inside r2, so approx ordering is reliable).
    #define LAUNCH_W(GCAP) \
        wmma_frnn16_kernel<GCAP><<<blocks, 256>>>( \
            d_p1, d_p2, d_sqn1, d_sqn2, P1, P2, K, r2, d_dists, d_idxs)
    if      (K <= 16) LAUNCH_W(32);
    else if (K <= 32) LAUNCH_W(64);
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
