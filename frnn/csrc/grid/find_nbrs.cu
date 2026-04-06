#include "grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Standard Max-Heap Insertion to keep track of K-nearest neighbors
__device__ void insert_neighbor(float* local_dists, int* local_idxs, int K, float d2, int idx2) {
    // We maintain a max-heap where local_dists[0] is always the largest distance
    if (d2 < local_dists[0]) {
        local_dists[0] = d2;
        local_idxs[0] = idx2;

        // Sift down to maintain max-heap property
        int i = 0;
        while (true) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int largest = i;

            if (left < K && local_dists[left] > local_dists[largest]) largest = left;
            if (right < K && local_dists[right] > local_dists[largest]) largest = right;

            if (largest != i) {
                // Swap distances
                float tmp_d = local_dists[i];
                local_dists[i] = local_dists[largest];
                local_dists[largest] = tmp_d;

                // Swap indices
                int tmp_idx = local_idxs[i];
                local_idxs[i] = local_idxs[largest];
                local_idxs[largest] = tmp_idx;

                i = largest;
            } else {
                break;
            }
        }
    }
}

__global__ void FindNbrs3DKernelV2(
    const float3* __restrict__ points1,      
    const float3* __restrict__ points2,      
    const int* __restrict__ pc2_grid_off,    
    const int* __restrict__ sorted_points2_idxs, 
    int P1, int K, float r2, 
    float* __restrict__ dists, 
    int* __restrict__ idxs,
    GridParams params) 
{
    int p1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (p1 >= P1) return;

    float3 pt1 = points1[p1];
    
    // Using fixed size for registers; K must be <= 128
    float local_dists[128]; 
    int local_idxs[128];

    for (int k = 0; k < K; ++k) {
        local_dists[k] = r2; // Initialize with search radius squared
        local_idxs[k] = -1;
    }

    int gx = floor((pt1.x - params.min_pos.x) / params.delta);
    int gy = floor((pt1.y - params.min_pos.y) / params.delta);
    int gz = floor((pt1.z - params.min_pos.z) / params.delta);

    for (int ix = gx - 1; ix <= gx + 1; ix++) {
        for (int iy = gy - 1; iy <= gy + 1; iy++) {
            for (int iz = gz - 1; iz <= gz + 1; iz++) {
                
                if (ix < 0 || ix >= params.resolution.x || 
                    iy < 0 || iy >= params.resolution.y || 
                    iz < 0 || iz >= params.resolution.z) continue;

                int cell_idx = ix * (params.resolution.y * params.resolution.z) + 
                               iy * params.resolution.z + iz;

                int start = pc2_grid_off[cell_idx];
                int end = pc2_grid_off[cell_idx + 1];

                for (int p2_idx = start; p2_idx < end; ++p2_idx) {
                    int original_idx2 = sorted_points2_idxs[p2_idx];
                    float3 pt2 = points2[original_idx2];

                    float d2 = (pt1.x - pt2.x)*(pt1.x - pt2.x) + 
                               (pt1.y - pt2.y)*(pt1.y - pt2.y) + 
                               (pt1.z - pt2.z)*(pt1.z - pt2.z);

                    if (d2 < r2) {
                        insert_neighbor(local_dists, local_idxs, K, d2, original_idx2);
                    }
                }
            }
        }
    }

    for (int k = 0; k < K; ++k) {
        dists[p1 * K + k] = local_dists[k];
        idxs[p1 * K + k] = local_idxs[k];
    }
}

extern "C" void run_find_nbrs(
    float3* d_points1, float3* d_points2, 
    int* d_pc2_grid_off, int* d_sorted_idxs,
    int P1, int K, float radius,
    float* d_dists, int* d_idxs,
    GridParams params) 
{
    int threads = 256;
    int blocks = (P1 + threads - 1) / threads;
    float r2 = radius * radius;

    FindNbrs3DKernelV2<<<blocks, threads>>>(
        d_points1, d_points2, d_pc2_grid_off, d_sorted_idxs, P1, K, r2, d_dists, d_idxs, params
    );
    cudaDeviceSynchronize();
}
