#ifndef BRUTEFORCE_H
#define BRUTEFORCE_H

#include <cuda_runtime.h>

extern "C" {
    /**
     * run_bruteforce: Standard O(N^2) search.
     * Every point in d_p1 checks every point in d_p2.
     * Useful for verifying the Grid-based search results.
     */
    void run_bruteforce(
        const float3* d_p1, 
        const float3* d_p2, 
        int P1, int P2, int K, float r,
        float* d_dists, int* d_idxs
    );
}

#endif