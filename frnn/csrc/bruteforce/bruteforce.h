#ifndef BRUTEFORCE_H
#define BRUTEFORCE_H

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