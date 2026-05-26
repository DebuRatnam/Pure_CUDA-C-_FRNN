#include "grid.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void insert_neighbor(float* local_dists, int* local_idxs, int K, float d2, int idx2) {
    // Early-exit: candidate is no better than current worst — skip entirely
    if (d2 >= local_dists[0]) return;

    local_dists[0] = d2;
    local_idxs[0]  = idx2;

    // Sift-down the replaced root to restore max-heap order.
    // Bounded loop (depth ≤ floor(log2(128)) = 7) lets the compiler unroll.
    int i = 0;
    #pragma unroll 7
    for (int depth = 0; depth < 7; depth++) {
        int left = 2 * i + 1, right = 2 * i + 2, largest = i;
        if (left  < K && local_dists[left]  > local_dists[largest]) largest = left;
        if (right < K && local_dists[right] > local_dists[largest]) largest = right;
        if (largest == i) break;
        float td = local_dists[i];  local_dists[i] = local_dists[largest]; local_dists[largest] = td;
        int   ti = local_idxs[i];   local_idxs[i]  = local_idxs[largest];  local_idxs[largest]  = ti;
        i = largest;
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
    #pragma unroll 16
    for (int k = 0; k < K; ++k) {
        local_dists[k] = r2;
        local_idxs[k] = -1;
    }
    // Tracks heap root (current worst accepted neighbor). Narrows as good
    // neighbors accumulate, allowing early-exit on most candidates.
    float max_dist_sq = r2;

    // 1. Determine the Hypergrid Cell for this N-D point
    // We calculate a 1D hash for the grid cell
    long long cell_idx = 0;
    long long stride = 1;
    
    for(int d = 0; d < dim; d++) {
        float pos = points1[d * P1 + p1];  // SoA
        int grid_pos = floor((pos - params.min_val) / params.radius); // Resolution based on radius
        
        // Clamp to grid boundaries
        grid_pos = max(0, min(grid_pos, params.res - 1));
        
        cell_idx += grid_pos * stride;
        stride *= params.res;
    }

    // 2. Decode cell_idx into per-dimension coordinates
    int cell_coords[MAX_DIM_SUPPORTED];
    {
        long long tmp = cell_idx;
        for (int d = 0; d < dim; d++) {
            cell_coords[d] = (int)(tmp % params.res);
            tmp /= params.res;
        }
    }

    // 3. Enumerate all 3^dim neighboring cells (offsets -1, 0, +1 per dimension)
    int num_neighbor_cells = 1;
    for (int d = 0; d < dim; d++) num_neighbor_cells *= 3;

    for (int nc = 0; nc < num_neighbor_cells; nc++) {
        long long neighbor_hash = 0;
        long long s = 1;
        bool valid = true;
        int tmp = nc;

        for (int d = 0; d < dim; d++) {
            int offset = (tmp % 3) - 1;   // maps 0,1,2 → -1,0,+1
            tmp /= 3;
            int coord = cell_coords[d] + offset;
            if (coord < 0 || coord >= params.res) { valid = false; break; }
            neighbor_hash += (long long)coord * s;
            s *= params.res;
        }
        if (!valid) continue;

        // After run_reorder_points, pc2_grid_off[c] holds the END of cell c
        // (atomicAdd increments in-place during scatter-sort).
        // So cell c spans sorted_idxs[ off[c-1] .. off[c]-1 ].
        int start = (neighbor_hash == 0) ? 0 : pc2_grid_off[neighbor_hash - 1];
        int end   = pc2_grid_off[neighbor_hash];

        for (int p2_idx = start; p2_idx < end; ++p2_idx) {
            int original_idx2 = sorted_points2_idxs[p2_idx];
            float d2 = 0.0f;
            #pragma unroll
            for (int d = 0; d < dim; d++) {
                float diff = points1[d * P1 + p1] - points2[d * P1 + original_idx2];  // SoA
                d2 += diff * diff;
            }
            if (d2 >= max_dist_sq) continue;
            insert_neighbor(local_dists, local_idxs, K, d2, original_idx2);
            max_dist_sq = local_dists[0];
        }
    }

    // 3. Write back to Global Memory (SoA: coalesced warp stores)
    #pragma unroll 16
    for (int k = 0; k < K; ++k) {
        dists[k * P1 + p1] = local_dists[k];
        idxs[k * P1 + p1]  = local_idxs[k];
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