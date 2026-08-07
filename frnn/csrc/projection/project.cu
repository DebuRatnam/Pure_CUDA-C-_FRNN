/*
 * =============================================================================
 * project.cu — PCA projection stage for the projection-FRNN algorithm (LEGACY)
 * =============================================================================
 *
 * NOTE: This file is NOT compiled by CMakeLists.txt. It is retained as reference
 * for the projection-FRNN approach but is dead code in the current build.
 *
 * Implements Stage 0 of the two-stage projection FRNN: projects (N, D) points
 * onto their top-k principal components and isotropically rescales into [0,1]^k,
 * entirely on the GPU. The resulting low-dimensional representation is then
 * searched with the grid FRNN kernel at an inflated radius (R * s, where s is
 * the isotropic scale factor), which yields a superset of the true D-dimensional
 * R-neighbors. verify.cu then filters this candidate set by recomputing exact
 * full-D distances.
 *
 * Design choices:
 *   - No cuBLAS / cuSOLVER: the O(N) work (mean, covariance, projection, min/max)
 *     is handled by custom CUDA kernels.
 *   - The O(D^3) eigendecomposition of the D×D covariance matrix runs on the CPU
 *     via cyclic Jacobi (D ≤ 32, so this is microseconds and avoids a library dep).
 *   - An orthonormal basis ensures the projection is contractive, guaranteeing the
 *     inflated-radius search in k-D returns a superset of the true D-D neighbors.
 *
 * Pipeline: AccumStatsKernel → host Jacobi → ProjectCenteredKernel →
 *           MinMaxAxisKernel → RescaleKernel (all orchestrated by run_pca_project).
 */
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

/*
 * AccumStatsKernel — accumulates per-axis sums and the upper triangle of the
 * outer-product matrix needed to form the empirical covariance. Each block
 * maintains a shared-memory partial in [D sums | D*D cross-products], then
 * atomically contributes its partial to the global accumulators. The covariance
 * is then computed on the CPU from these sums (see run_pca_project).
 *
 * Key variables:
 *   sh[]      — shared partial: ssx[D] followed by ssxx[D*D] (upper triangle only)
 *   N         — total point count; loop stride = gridDim.x*blockDim.x for coverage
 *   sumx/sumxx— global device accumulators for the mean and cross-product sums
 */
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

/*
 * ProjectCenteredKernel — projects each point onto the k PCA basis vectors.
 * One thread per point. Subtracts the per-axis mean, then computes the dot
 * product with each of the k eigenvectors (columns of `basis`). Output is
 * the k-dimensional projection of each point before normalization.
 *
 * Key variables:
 *   mean[D]       — per-axis mean, subtracted before projection to center the data
 *   basis[D*k]    — column-major PCA basis: basis[d*k + c] = eigenvector c, coord d
 *   proj[i*k + c] — output: projection score of point i onto principal axis c
 */
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

/*
 * MinMaxAxisKernel — computes per-axis min and max of the projected coordinates.
 * Each block reduces its slice of rows into shared-memory accumulators (one
 * atomicMinFloat / atomicMaxFloat per coord per row), then contributes to global
 * accumulators. The resulting range is used by RescaleKernel to compute the
 * isotropic scale s = 1 / max_range.
 *
 * Key variables:
 *   smn[k] / smx[k] — block-shared per-axis min/max accumulators (2k floats)
 *   gmn / gmx        — global device min/max per axis, updated atomically
 *   proj[i*k + c]    — input: projection values from ProjectCenteredKernel
 */
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

/*
 * RescaleKernel — isotropically rescales all projected coordinates into [0,1]^k.
 * Applies proj01 = (proj - mn) * s in-place. The scale s is the same for all k
 * axes (isotropic), which preserves the relative distances between points so that
 * a radius-r*s search in the projected space returns a superset of radius-r neighbors
 * in the original D-dimensional space.
 *
 * Key variables:
 *   s        — isotropic scale = 1 / max_range (max range across all k axes)
 *   mn[k]    — per-axis minimum; shifts each axis so its minimum maps to 0
 *   proj[i*k + c] — in-place: unnormalized projection in, [0,1] out
 */
__global__ void RescaleKernel(float* __restrict__ proj, int N, int k,
                              const float* __restrict__ mn, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float* r = proj + (long long)i * k;
    for (int c = 0; c < k; c++) r[c] = (r[c] - mn[c]) * s;
}

/*
 * jacobi_eigh — host-side cyclic Jacobi eigendecomposition of a symmetric D×D matrix.
 * Iterates Jacobi sweeps (annihilating off-diagonal elements pairwise) until the
 * sum of squared off-diagonal elements falls below 1e-20 or 100 sweeps complete.
 * Runs on the CPU because D ≤ 32 — a full GPU eigen-solver (cuSOLVER) would cost
 * more in launch overhead than this entire function takes to run.
 *
 * Key variables:
 *   A[D*D]    — input: symmetric covariance matrix (row-major); overwritten in place
 *   evec[D*D] — output: column j is the eigenvector for eigenvalue eval[j]
 *   phi       — Jacobi rotation angle for the current (p,q) element
 */
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

/*
 * run_pca_project — orchestrates the full GPU PCA + normalization pipeline.
 * Runs AccumStatsKernel → D2H copy → host Jacobi → D2H upload of mean and basis
 * → ProjectCenteredKernel → MinMaxAxisKernel → RescaleKernel in sequence.
 * Returns 0 on success; returns 1 if D is too large for the shared-memory stats
 * path (caller should fall back to brute force). All device scratch buffers are
 * caller-owned to avoid repeated allocation across multiple search calls.
 *
 * Key variables:
 *   d_proj01     — output: (N, k) normalized projection in [0, 1]^k
 *   out_s        — output: isotropic scale s; caller multiplies radius by s when
 *                  searching in projection space to get a superset of D-D neighbors
 *   out_var_ratio — output: fraction of total variance captured by the top-k axes;
 *                  low values indicate the projection loses significant structure
 */
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
