#include "bruteforce.h"
#include <device_launch_parameters.h>
#include <float.h>

// Max-Heap Insertion logic remains the same (it only cares about distance values)
__device__ void bf_insert_neighbor(float* local_dists, int* local_idxs, int K, float d2, int idx2) {
    if (d2 < local_dists[0]) {
        local_dists[0] = d2;
        local_idxs[0] = idx2;
        int i = 0;
        while (true) {
            int left = 2 * i + 1, right = 2 * i + 2, largest = i;
            if (left < K && local_dists[left] > local_dists[largest]) largest = left;
            if (right < K && local_dists[right] > local_dists[largest]) largest = right;
            if (largest != i) {
                float td = local_dists[i]; local_dists[i] = local_dists[largest]; local_dists[largest] = td;
                int ti = local_idxs[i]; local_idxs[i] = local_idxs[largest]; local_idxs[largest] = ti;
                i = largest;
            } else break;
        }
    }
}

__global__ void BruteforceNDKernel(
    const float* p1_ptr, const float* p2_ptr,
    int P1, int P2, int K, int dim, float r2,
    float* dists, int* idxs) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P1) return;

    float local_dists[128];
    int local_idxs[128];

    for (int k = 0; k < K; k++) {
        local_dists[k] = r2; 
        local_idxs[k] = -1;
    }

    // Exhaustive search: Check against EVERY point in the second set
    for (int j = 0; j < P2; j++) {
        float d2 = 0.0f;
        
        // Loop over dimensions for N-D distance
        for (int d = 0; d < dim; d++) {
            float diff = p1_ptr[i * dim + d] - p2_ptr[j * dim + d];
            d2 += diff * diff;
        }

        if (d2 < r2) {
            bf_insert_neighbor(local_dists, local_idxs, K, d2, j);
        }
    }

    for (int k = 0; k < K; k++) {
        dists[i * K + k] = local_dists[k];
        idxs[i * K + k] = local_idxs[k];
    }
}

extern "C" void run_bruteforce(
    const float* d_p1, const float* d_p2, 
    int P1, int P2, int K, int dim, float r,
    float* d_dists, int* d_idxs) 
{
    int threads = 256;
    int blocks = (P1 + threads - 1) / threads;
    float r2 = r * r;

    BruteforceNDKernel<<<blocks, threads>>>(d_p1, d_p2, P1, P2, K, dim, r2, d_dists, d_idxs);
    cudaDeviceSynchronize();
}