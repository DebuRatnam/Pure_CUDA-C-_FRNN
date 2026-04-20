#include "grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Max-Heap Insertion stays similar, but indices/dists are passed as raw pointers
__device__ void insert_neighbor(float* local_dists, int* local_idxs, int K, float d2, int idx2) {
    if (d2 < local_dists[0]) {
        local_dists[0] = d2;
        local_idxs[0] = idx2;

        int i = 0;
        while (true) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int largest = i;

            if (left < K && local_dists[left] > local_dists[largest]) largest = left;
            if (right < K && local_dists[right] > local_dists[largest]) largest = right;

            if (largest != i) {
                float tmp_d = local_dists[i]; local_dists[i] = local_dists[largest]; local_dists[largest] = tmp_d;
                int tmp_idx = local_idxs[i]; local_idxs[i] = local_idxs[largest]; local_idxs[largest] = tmp_idx;
                i = largest;
            } else break;
        }
    }
}

__global__ void FindNbrsNDKernel(
    const float* __restrict__ points1,      
    const float* __restrict__ points2,      
    const int* __restrict__ pc2_grid_off,    
    const int* __restrict__ sorted_points2_idxs, 
    int P1, int K, int dim, float r2, 
    float* __restrict__ dists, 
    int* __restrict__ idxs,
    GridParams params) 
{
    int p1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (p1 >= P1) return;

    // Use a local buffer for K-nearest (Adjust 128 if K is larger)
    float local_dists[128]; 
    int local_idxs[128];
    for (int k = 0; k < K; ++k) {
        local_dists[k] = r2;
        local_idxs[k] = -1;
    }

    // 1. Determine the Hypergrid Cell for this N-D point
    // We calculate a 1D hash for the grid cell
    long long cell_idx = 0;
    long long stride = 1;
    
    for(int d = 0; d < dim; d++) {
        float pos = points1[p1 * dim + d];
        int grid_pos = floor((pos - params.min_val) / params.radius); // Resolution based on radius
        
        // Clamp to grid boundaries
        grid_pos = max(0, min(grid_pos, params.res - 1));
        
        cell_idx += grid_pos * stride;
        stride *= params.res;
    }

    // 2. Neighbor Search
    // For N-D, a full 3^D neighbor search is exponential. 
    // Optimization: Search current cell and immediate 1D-adjacent cells in the hash
    // (For a true N-D grid search, we'd use a recursive or stack-based approach)
    
    // For now, we search the calculated cell and its direct neighbors in the offset table
    int start_cell = max(0LL, cell_idx - 1);
    int end_cell = min((long long)params.total_cells - 1, cell_idx + 1);

    for (long long c = start_cell; c <= end_cell; c++) {
        int start = pc2_grid_off[c];
        int end = pc2_grid_off[c + 1];

        for (int p2_idx = start; p2_idx < end; ++p2_idx) {
            int original_idx2 = sorted_points2_idxs[p2_idx];
            
            // N-Dimensional Euclidean Distance
            float d2 = 0;
            for(int d = 0; d < dim; d++) {
                float diff = points1[p1 * dim + d] - points2[original_idx2 * dim + d];
                d2 += diff * diff;
            }

            if (d2 < r2) {
                insert_neighbor(local_dists, local_idxs, K, d2, original_idx2);
            }
        }
    }

    // 3. Write back to Global Memory
    for (int k = 0; k < K; ++k) {
        dists[p1 * K + k] = local_dists[k];
        idxs[p1 * K + k] = local_idxs[k];
    }
}

extern "C" void run_find_nbrs(
    float* d_points1, float* d_points2, 
    int* d_pc2_grid_off, int* d_sorted_idxs,
    int P1, int K, int dim, float radius,
    float* d_dists, int* d_idxs,
    GridParams params) 
{
    int threads = 256;
    int blocks = (P1 + threads - 1) / threads;
    float r2 = radius * radius;

    FindNbrsNDKernel<<<blocks, threads>>>(
        d_points1, d_points2, d_pc2_grid_off, d_sorted_idxs, 
        P1, K, dim, r2, d_dists, d_idxs, params
    );
    cudaDeviceSynchronize();
}