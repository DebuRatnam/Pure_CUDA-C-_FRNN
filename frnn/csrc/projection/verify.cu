/*
 * =============================================================================
 * verify.cu — Full-D candidate verification stage for projection-FRNN (LEGACY)
 * =============================================================================
 *
 * NOTE: This file is NOT compiled by CMakeLists.txt. It is retained as reference
 * for the projection-FRNN approach but is dead code in the current build.
 *
 * Implements Stage 2 (the verification step) of the two-stage projection FRNN:
 *
 *   Stage 1 (project.cu + grid FRNN): project the N points into k-D PCA space and
 *   find the O nearest neighbors in projection space at radius r*s. This produces
 *   an (N, O) candidate table that is a superset of the true D-dimensional R-neighbors
 *   (guaranteed by the contractive, orthonormal projection).
 *
 *   Stage 2 (this file): for each query, recompute the true full-D squared L2
 *   distance to each of its O candidates, keep the K with d^2 ≤ r^2. This is a
 *   fused gather-compute-filter kernel: no (N × O × D) temporary is materialized,
 *   which was the dominant cost of the original PyTorch implementation.
 *
 * One thread per query. Query coords are cached in a register array (qreg) once,
 * then reused for all O candidates. A max-heap of size CAP (templated, smallest that
 * holds K) accumulates the K nearest. Output is in AoS layout to match the engine
 * convention for the projection path.
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int VERIFY_MAX_DIM = 128;   // matches engine MAX_DIM_SUPPORTED

/*
 * verify_insert<CAP> — max-heap replace-root + sift-down for the verify kernel.
 * Same algorithm as insert_neighbor_t in find_nbrs.cu. Guards on d[0] (heap root)
 * so only candidates that improve on the current worst are inserted. CAP is
 * compile-time so the sift depth (4–7 levels) is a constant and the loop unrolls.
 *
 * Key variables:
 *   d[0]  — heap root; current worst (largest) accepted squared distance; entry guard
 *   CAP   — compile-time heap capacity; determines the unrolled sift depth
 *   i     — current node index during sift-down toward the leaves
 */
template<int CAP>
__device__ __forceinline__ void verify_insert(float* d, int* ix, int K, float d2, int j) {
    if (d2 >= d[0]) return;
    d[0] = d2; ix[0] = j;
    int i = 0;
    constexpr int DEPTH = (CAP <= 16) ? 4 : (CAP <= 32) ? 5 : (CAP <= 64) ? 6 : 7;
    #pragma unroll
    for (int depth = 0; depth < DEPTH; depth++) {
        int l = 2 * i + 1, r = 2 * i + 2, large = i;
        if (l < K && d[l] > d[large]) large = l;
        if (r < K && d[r] > d[large]) large = r;
        if (large == i) break;
        float td = d[i]; d[i] = d[large]; d[large] = td;
        int   ti = ix[i]; ix[i] = ix[large]; ix[large] = ti;
        i = large;
    }
}

/*
 * VerifyKernel<CAP> — one thread per query; recomputes full-D distances to each candidate.
 * Caches the query's D coordinates in a local array (qreg) once, then iterates
 * over the O candidates from the projection-space search, computing the squared L2
 * distance in the original D-dimensional space and inserting into a K-heap when
 * below r^2. Skips candidates with id = -1 (empty projection slots).
 *
 * Key variables:
 *   qreg[VERIFY_MAX_DIM] — register-resident query coordinates; loaded once, reused O times
 *   cand[q*O + o]        — o-th candidate original index for query q; -1 = empty
 *   O                    — oversample count: number of projection-space candidates per query
 */
template<int CAP>
__global__ void VerifyKernel(
    const float* __restrict__ pts,    // (N, D) AoS  [q*D + d]
    const int*   __restrict__ cand,   // (N, O) AoS  [q*O + o], original ids or -1
    int N, int D, int O, int K, float r2,
    float* __restrict__ out_d,        // (N, K) AoS, squared distance, heap order
    int*   __restrict__ out_i)        // (N, K) AoS, neighbor ids (-1 = empty)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= N) return;

    float ld[CAP]; int li[CAP];
    #pragma unroll
    for (int k = 0; k < CAP; k++) { ld[k] = r2; li[k] = -1; }

    // Cache the query coords once; reused for every candidate.
    float qreg[VERIFY_MAX_DIM];
    const float* qp = pts + (long long)q * D;
    for (int d = 0; d < D; d++) qreg[d] = qp[d];

    for (int o = 0; o < O; o++) {
        int j = cand[(long long)q * O + o];
        if (j < 0) continue;
        const float* pp = pts + (long long)j * D;
        float d2 = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; d++) { float df = qreg[d] - pp[d]; d2 += df * df; }
        if (d2 < r2) verify_insert<CAP>(ld, li, K, d2, j);
    }

    for (int k = 0; k < K; k++) {
        out_d[(long long)q * K + k] = ld[k];
        out_i[(long long)q * K + k] = li[k];
    }
}

/*
 * run_verify_candidates — host wrapper for VerifyKernel.
 * Squares the radius, selects the smallest compile-time CAP that holds K, and
 * launches one thread per query point. Output arrays (d_out_d, d_out_i) are
 * AoS layout: out[q*K + k] for the k-th neighbor of query q.
 *
 * Key variables:
 *   r2   — squared radius (r^2); the kernel filters candidates by d^2 < r2
 *   O    — oversample factor: number of projection-space candidates per query
 *   N    — total query/point count; determines grid size
 */
extern "C" void run_verify_candidates(
    const float* d_pts, const int* d_cand,
    int N, int D, int O, int K, float r,
    float* d_out_d, int* d_out_i)
{
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    float r2    = r * r;
    #define LAUNCH_VERIFY(CAP) \
        VerifyKernel<CAP><<<blocks, threads>>>(d_pts, d_cand, N, D, O, K, r2, d_out_d, d_out_i)
    if      (K <= 16) LAUNCH_VERIFY(16);
    else if (K <= 32) LAUNCH_VERIFY(32);
    else if (K <= 64) LAUNCH_VERIFY(64);
    else              LAUNCH_VERIFY(128);
    #undef LAUNCH_VERIFY
    cudaDeviceSynchronize();
}
