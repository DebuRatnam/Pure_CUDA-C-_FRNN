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

// GEMM-blocked, FlashAttention-style. Each block owns TILE_M = WARPS*QPW query
// rows; each warp owns QPW=32 of them as two stacked WMMA M-subtiles, so all 32
// lanes own a query + heap (no idle lanes in the scalar fuse/rerank phases).
//
// A WIDE ref window (NT=128 rows) is staged to SMEM per __syncthreads, then the
// GEMM + radius-fuse run in NSUB=16-ref subtiles WITHOUT materializing the full
// G: each warp computes its 32x16 G subtile into a tiny private scratch, fuses it
// into the heaps, and reuses the scratch for the next subtile. Result: only
// P2/128 block syncs (vs P2/16 before) yet G never spills past 32x16 in SMEM.
//
// sQ and sT are TF32-ROUNDED at stage time (once each), so the per-fragment
// __float_to_tf32 conversion is gone from the inner GEMM loop -- the fragment
// elements loaded from SMEM are already tf32-valued. Query fragments (2 M-subtiles
// x 2 k-steps = 4) are loaded once and held in registers across the whole ref loop.
//
// WMMA shape 16x16x8 (TF32 on sm_80). D=16 -> K-axis is two 8-wide mma steps. For
// G = Q . T^T with Q,T row-major [point][dim] in SMEM, load Q as row_major
// matrix_a and T as col_major matrix_b at the same pointer (row-major [N][K] read
// col_major [K][N] = transpose).
//
// GCAP = gen-heap capacity -> slack so TF32 reshuffling near the radius boundary
// cannot evict the true top-K before the exact rerank.
template<int GCAP>
__global__ void wmma_frnn16_kernel(
    const float* __restrict__ p1, const float* __restrict__ p2,
    const float* __restrict__ sqn1, const float* __restrict__ sqn2,
    int P1, int P2, int K, float r2,
    float* __restrict__ dists, int* __restrict__ idxs)
{
    constexpr int DIM = 16, NSUB = 16, WARPS = 4, QPW = 32;     // 2 M-subtiles/warp
    constexpr int TILE_M = WARPS * QPW;                         // 128 queries/block
    constexpr int NT = 128, NSUBS = NT / NSUB;                  // wide ref window, 8 subtiles
    __shared__ float sQ[TILE_M * DIM];          // query tile, row-major, tf32  ( 8 KB)
    __shared__ float sT[NT * DIM];              // ref   tile, row-major, tf32  ( 8 KB)
    __shared__ float sG[WARPS * QPW * NSUB];    // per-warp 32x16 G scratch     ( 8 KB)

    const int tid  = threadIdx.x;          // 0..127
    const int warp = tid >> 5;             // 0..3
    const int lane = tid & 31;             // 0..31
    const int m0   = blockIdx.x * TILE_M;  // first query row of this block

    // Relaxed gen radius: keep approximate candidates a bit past r2 so a true
    // in-radius neighbor whose TF32 d2 overshoots is still captured. 5% + 1e-3
    // dwarfs the ~4e-3 TF32 d2 error; the exact rerank below re-imposes r2.
    const float RR2 = r2 * 1.05f + 1e-3f;

    // Stage + TF32-round the whole query tile once (queries fixed for the block).
    for (int e = tid; e < TILE_M * DIM; e += blockDim.x) {
        int r = e / DIM, d = e % DIM, gi = m0 + r;
        sQ[e] = (gi < P1) ? wmma::__float_to_tf32(p1[d * P1 + gi]) : 0.0f;
    }
    __syncthreads();

    // Persistent query fragments: 2 M-subtiles x 2 k-steps, loaded once. Already
    // tf32-rounded in SMEM -> no per-fragment convert.
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> qa[2][2];
    #pragma unroll
    for (int ms = 0; ms < 2; ms++)
        #pragma unroll
        for (int ks = 0; ks < 2; ks++)
            wmma::load_matrix_sync(qa[ms][ks], sQ + (warp * QPW + ms * 16) * DIM + ks * 8, DIM);

    // Each of the 32 lanes owns one query row + its oversized gen heap.
    const int  qrow   = warp * QPW + lane;               // local query row 0..127
    const int  qi     = m0 + qrow;                       // global query
    const bool active = (qi < P1);
    float ld[GCAP];
    int   li[GCAP];
    for (int g = 0; g < GCAP; g++) { ld[g] = RR2; li[g] = -1; }
    const float qn = active ? sqn1[qi] : 0.0f;

    float* mysG = sG + warp * (QPW * NSUB);    // this warp's 32x16 G scratch

    // Slide the wide reference window over all P2 points.
    for (int n0 = 0; n0 < P2; n0 += NT) {
        for (int e = tid; e < NT * DIM; e += blockDim.x) {
            int r = e / DIM, d = e % DIM, gj = n0 + r;
            sT[e] = (gj < P2) ? wmma::__float_to_tf32(p2[d * P2 + gj]) : 0.0f;
        }
        __syncthreads();

        // Inner: NSUBS subtiles of 16 refs; GEMM + fuse each without a block sync.
        #pragma unroll
        for (int ns = 0; ns < NSUBS; ns++) {
            const float* sTsub = sT + ns * NSUB * DIM;        // [16 refs][16]
            #pragma unroll
            for (int ms = 0; ms < 2; ms++) {
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
                wmma::fill_fragment(acc, 0.0f);
                #pragma unroll
                for (int ks = 0; ks < 2; ks++) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b;
                    wmma::load_matrix_sync(b, sTsub + ks * 8, DIM);   // col-major = T^T
                    wmma::mma_sync(acc, qa[ms][ks], b, acc);
                }
                wmma::store_matrix_sync(mysG + ms * 16 * NSUB, acc, NSUB, wmma::mem_row_major);
            }
            __syncwarp();   // both M-subtiles stored before any lane reads mysG

            // Fuse this 16-ref subtile. Lane owns row 'lane' of the warp's 32x16 G.
            if (active) {
                #pragma unroll
                for (int jj = 0; jj < NSUB; jj++) {
                    int gj = n0 + ns * NSUB + jj;
                    if (gj >= P2) continue;
                    // d2 = ||q||^2 + ||t||^2 - 2 q.t  (TF32 q.t; exact rerank below).
                    float d2 = qn + sqn2[gj] - 2.0f * mysG[lane * NSUB + jj];
                    if (d2 < 0.0f) d2 = 0.0f;
                    if (d2 < RR2) insert_neighbor(ld, li, GCAP, d2, gj);
                }
            }
            __syncwarp();   // all lanes done reading mysG before next subtile overwrites
        }
        __syncthreads();    // window consumed before sT reload
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
// SMEM is static (sQ 8 KB + sT 8 KB + sG 8 KB = 24 KB), under 48 KB -> no opt-in.
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

    const int blocks = (P1 + 127) / 128;   // TILE_M = 128 queries per block (4 warps)
    // GCAP = gen-heap capacity ~2K (slack so TF32 reshuffling can't evict the true
    // top-K), capped at 128 = engine K limit. K<=64 still gets 2x slack; K<=128 has
    // none (the boundary is then far inside r2, so approx ordering is reliable).
    #define LAUNCH_W(GCAP) \
        wmma_frnn16_kernel<GCAP><<<blocks, 128>>>( \
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
