// project.cu — pure C++/CUDA PCA projection (Stage 0 of the projection FRNN).
//
// Projects (N, D) points onto their top-k principal axes and isotropically rescales
// into [0,1]^k, entirely on the GPU. No cuBLAS / cuSOLVER:
//   - the O(N) work (mean, covariance accumulation, projection, min/max) is custom kernels;
//   - the tiny O(D^3) eigendecomposition of the D x D covariance runs on the host via
//     cyclic Jacobi (D is small — 16/32 — so this is microseconds and avoids a library dep).
//
// Orthonormal (eigenvector) basis => the projection is contractive, so a radius-(R*s)
// search in the projection returns a superset of the true D-dim R-neighbors. The caller
// (search_projected) then verifies in full D to get the exact answer.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <numeric>

// ---- float atomic min/max (CAS loop; works on both global and shared addresses) ----
__device__ __forceinline__ float atomicMinFloat(float* addr, float val) {
    int* a = (int*)addr; int old = *a, assumed;
    do { assumed = old;
         if (__int_as_float(assumed) <= val) break;
         old = atomicCAS(a, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}
__device__ __forceinline__ float atomicMaxFloat(float* addr, float val) {
    int* a = (int*)addr; int old = *a, assumed;
    do { assumed = old;
         if (__int_as_float(assumed) >= val) break;
         old = atomicCAS(a, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

// ---- Accumulate sum_x[D] and sum_xx[D*D] (upper triangle) via block-shared partials ----
// Shared layout: [D sums][D*D cross-products]. Requires (D + D*D)*4 bytes of shared mem.
__global__ void AccumStatsKernel(const float* __restrict__ pts, int N, int D,
                                 float* __restrict__ sumx, float* __restrict__ sumxx) {
    extern __shared__ float sh[];
    float* ssx  = sh;            // D
    float* ssxx = sh + D;        // D*D
    for (int t = threadIdx.x; t < D + D * D; t += blockDim.x) sh[t] = 0.0f;
    __syncthreads();

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < N; i += (long long)gridDim.x * blockDim.x) {
        const float* p = pts + i * D;
        for (int a = 0; a < D; a++) {
            float va = p[a];
            atomicAdd(&ssx[a], va);
            for (int b = a; b < D; b++) atomicAdd(&ssxx[a * D + b], va * p[b]);
        }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < D;     t += blockDim.x) atomicAdd(&sumx[t],  ssx[t]);
    for (int t = threadIdx.x; t < D * D; t += blockDim.x) atomicAdd(&sumxx[t], ssxx[t]);
}

// Project onto the centered basis: proj[i,c] = sum_d (pts[i,d]-mean[d]) * basis[d*k+c].
__global__ void ProjectCenteredKernel(const float* __restrict__ pts, int N, int D, int k,
                                      const float* __restrict__ mean,
                                      const float* __restrict__ basis,
                                      float* __restrict__ proj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float* p = pts + (long long)i * D;
    for (int c = 0; c < k; c++) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++) acc += (p[d] - mean[d]) * basis[d * k + c];
        proj[(long long)i * k + c] = acc;
    }
}

// Per-axis min/max over the projection (block-shared reduction, then one atomic per axis).
__global__ void MinMaxAxisKernel(const float* __restrict__ proj, int N, int k,
                                 float* __restrict__ gmn, float* __restrict__ gmx) {
    extern __shared__ float sh[];
    float* smn = sh;            // k
    float* smx = sh + k;        // k
    for (int c = threadIdx.x; c < k; c += blockDim.x) { smn[c] = FLT_MAX; smx[c] = -FLT_MAX; }
    __syncthreads();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < N; i += (long long)gridDim.x * blockDim.x) {
        const float* r = proj + i * k;
        for (int c = 0; c < k; c++) { atomicMinFloat(&smn[c], r[c]); atomicMaxFloat(&smx[c], r[c]); }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        atomicMinFloat(&gmn[c], smn[c]); atomicMaxFloat(&gmx[c], smx[c]);
    }
}

// proj01[i,c] = (proj[i,c] - mn[c]) * s   (isotropic scale s, per-axis offset)
__global__ void RescaleKernel(float* __restrict__ proj, int N, int k,
                              const float* __restrict__ mn, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float* r = proj + (long long)i * k;
    for (int c = 0; c < k; c++) r[c] = (r[c] - mn[c]) * s;
}

// ---- host: cyclic Jacobi eigendecomposition of a symmetric D x D matrix ----
// A (row-major, D*D) is overwritten; evec (row-major, columns = eigenvectors), eval[D].
static void jacobi_eigh(std::vector<double>& A, int D,
                        std::vector<double>& eval, std::vector<double>& evec) {
    evec.assign(D * D, 0.0);
    for (int i = 0; i < D; i++) evec[i * D + i] = 1.0;
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int p = 0; p < D; p++) for (int q = p + 1; q < D; q++) off += A[p * D + q] * A[p * D + q];
        if (off < 1e-20) break;
        for (int p = 0; p < D; p++) {
            for (int q = p + 1; q < D; q++) {
                double apq = A[p * D + q];
                if (fabs(apq) < 1e-30) continue;
                double app = A[p * D + p], aqq = A[q * D + q];
                double phi = 0.5 * atan2(2.0 * apq, aqq - app);
                double c = cos(phi), s = sin(phi);
                for (int i = 0; i < D; i++) {          // rotate rows/cols p,q
                    double aip = A[i * D + p], aiq = A[i * D + q];
                    A[i * D + p] = c * aip - s * aiq;
                    A[i * D + q] = s * aip + c * aiq;
                }
                for (int i = 0; i < D; i++) {
                    double api = A[p * D + i], aqi = A[q * D + i];
                    A[p * D + i] = c * api - s * aqi;
                    A[q * D + i] = s * api + c * aqi;
                }
                for (int i = 0; i < D; i++) {          // accumulate eigenvectors
                    double vip = evec[i * D + p], viq = evec[i * D + q];
                    evec[i * D + p] = c * vip - s * viq;
                    evec[i * D + q] = s * vip + c * viq;
                }
            }
        }
    }
    eval.resize(D);
    for (int i = 0; i < D; i++) eval[i] = A[i * D + i];
}

// Largest shared-mem footprint AccumStats can use: (D + D*D) floats must fit ~48 KB.
static inline bool accum_shared_fits(int D) { return (size_t)(D + D * D) * sizeof(float) <= 48000; }

// run_pca_project: fill d_proj01 (N,k) in [0,1], set *out_s (distance scale) and
// *out_var_ratio (top-k variance fraction). Returns 0 on success, 1 if D is too large
// for the shared-memory stats path (caller should fall back to brute force).
// Scratch (caller-owned, device): sumx[D], sumxx[D*D], mean[D], basis[D*k], minmax[2*k].
extern "C" int run_pca_project(
    const float* d_pts, int N, int D, int k,
    float* d_proj01,
    float* d_sumx, float* d_sumxx, float* d_mean, float* d_basis, float* d_minmax,
    float* out_s, float* out_var_ratio)
{
    if (!accum_shared_fits(D)) return 1;

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    int red_blocks = std::min(blocks, 1024);

    cudaMemset(d_sumx,  0, D * sizeof(float));
    cudaMemset(d_sumxx, 0, (size_t)D * D * sizeof(float));
    size_t shmem = (size_t)(D + D * D) * sizeof(float);
    AccumStatsKernel<<<red_blocks, threads, shmem>>>(d_pts, N, D, d_sumx, d_sumxx);

    std::vector<float> hsx(D), hsxx((size_t)D * D);
    cudaMemcpy(hsx.data(),  d_sumx,  D * sizeof(float),            cudaMemcpyDeviceToHost);
    cudaMemcpy(hsxx.data(), d_sumxx, (size_t)D * D * sizeof(float), cudaMemcpyDeviceToHost);

    // mean + covariance (symmetric; mirror the upper triangle), in double.
    std::vector<double> mean(D), cov((size_t)D * D);
    for (int d = 0; d < D; d++) mean[d] = (double)hsx[d] / N;
    for (int a = 0; a < D; a++)
        for (int b = a; b < D; b++) {
            double c = (double)hsxx[a * D + b] / N - mean[a] * mean[b];
            cov[a * D + b] = c; cov[b * D + a] = c;
        }

    std::vector<double> eval, evec;
    jacobi_eigh(cov, D, eval, evec);

    // rank dims by eigenvalue descending; take top-k.
    std::vector<int> ord(D);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return eval[a] > eval[b]; });

    double tot = 0.0, top = 0.0;
    for (int d = 0; d < D; d++) tot += std::max(0.0, eval[d]);
    for (int c = 0; c < k; c++) top += std::max(0.0, eval[ord[c]]);
    *out_var_ratio = (tot > 1e-20) ? (float)(top / tot) : 1.0f;

    // basis (D x k): column c = eigenvector ord[c]; evec column j is evec[i*D+j].
    std::vector<float> hbasis((size_t)D * k), hmean(D);
    for (int d = 0; d < D; d++) hmean[d] = (float)mean[d];
    for (int d = 0; d < D; d++)
        for (int c = 0; c < k; c++) hbasis[d * k + c] = (float)evec[d * D + ord[c]];
    cudaMemcpy(d_mean,  hmean.data(),  D * sizeof(float),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_basis, hbasis.data(), (size_t)D * k * sizeof(float), cudaMemcpyHostToDevice);

    ProjectCenteredKernel<<<blocks, threads>>>(d_pts, N, D, k, d_mean, d_basis, d_proj01);

    // init min/max accumulators, reduce, copy back, compute isotropic scale.
    std::vector<float> init(2 * k);
    for (int c = 0; c < k; c++) { init[c] = FLT_MAX; init[k + c] = -FLT_MAX; }
    cudaMemcpy(d_minmax, init.data(), 2 * k * sizeof(float), cudaMemcpyHostToDevice);
    size_t mmsh = (size_t)2 * k * sizeof(float);
    MinMaxAxisKernel<<<red_blocks, threads, mmsh>>>(d_proj01, N, k, d_minmax, d_minmax + k);

    std::vector<float> hmm(2 * k);
    cudaMemcpy(hmm.data(), d_minmax, 2 * k * sizeof(float), cudaMemcpyDeviceToHost);
    float grange = 0.0f;
    for (int c = 0; c < k; c++) grange = std::max(grange, hmm[k + c] - hmm[c]);
    if (grange <= 0.0f) grange = 1.0f;
    float s = 1.0f / grange;
    *out_s = s;

    RescaleKernel<<<blocks, threads>>>(d_proj01, N, k, d_minmax, s);
    cudaDeviceSynchronize();
    return 0;
}
