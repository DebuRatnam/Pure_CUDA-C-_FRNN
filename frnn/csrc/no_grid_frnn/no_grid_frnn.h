#ifndef NO_GRID_FRNN_H
#define NO_GRID_FRNN_H

/*
 * =============================================================================
 * no_grid_frnn.h — Header for the brute-force FRNN fallback path
 * =============================================================================
 *
 * Declares run_bruteforce, the O(N^2) fallback used by FRNNEngine when the
 * uniform-grid path is infeasible:
 *   - total_cells > 1,000,000 (grid too large to fit in pre-allocated device memory)
 *   - 3^D >= total_cells (the 3^D neighbor shell covers the entire grid, so the
 *     grid's per-cell overhead is pure waste)
 *   - res <= 1 (single-cell grid; prefix scan degenerates)
 *
 * The implementation in no_grid_frnn.cu uses shared-memory tiling to amortize
 * global memory reads across a block of query threads, and dispatches a D=16
 * specialization with fully register-resident query coords and unrolled float4
 * distance computation for the most common high-D case.
 */

#include <cuda_runtime.h>

extern "C" {
    void run_bruteforce(
        const float* d_p1, 
        const float* d_p2, 
        int P1, int P2, int K, int dim, float r,
        float* d_dists, int* d_idxs
    );
}

#endif