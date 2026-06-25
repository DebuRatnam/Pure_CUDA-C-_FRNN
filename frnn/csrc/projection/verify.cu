// verify.cu — fused candidate-verify kernel for the projection two-stage FRNN.
//
// Stage 2 of the projection method: given each query's candidate ids (the
// oversample-nearest in the 3D projection, from the grid search), recompute the
// TRUE full-D squared distance to each candidate, keep the K nearest with d^2 <= r^2.
//
// One thread per query. The query coords are cached in a local array; the K-heap
// (sized to the smallest compile-time CAP that holds K) lives in registers/local.
// No (N, oversample, D) temporary is materialized — that gather+reduce was the
// dominant cost of the torch implementation this replaces. Output is squared
// distance in heap order, matching the native engine convention.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int VERIFY_MAX_DIM = 128;   // matches engine MAX_DIM_SUPPORTED

// Max-heap replace-root + sift-down (same scheme as bruteforce.cu). Keeps the K
// smallest d^2 seen. CAP is compile-time so the sift depth is a constant.
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

// Host wrapper. r is the (full-D) radius; the kernel compares squared distances.
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
