#include "prefix_sum.h"
#include <cuda_runtime.h>

// This must match the name inside parallel-scan/kernels.cu
__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo);
extern "C" void scanLargeDeviceArray(int *d_out, int *d_in, int n, bool bcao);

void run_prefix_sum(int* d_out, int* d_in, int n) {
    const int THREADS_PER_BLOCK = 512;
    const int ELEMENTS_PER_BLOCK = 2 * THREADS_PER_BLOCK;

    if (n <= ELEMENTS_PER_BLOCK) {
        int threads = n / 2;
        // Adjust shared memory calculation to avoid bank conflicts if your kernel supports it
        size_t shared_mem = (n + (n / 32)) * sizeof(int); 
        prescan_arbitrary<<<1, threads, shared_mem>>>(d_out, d_in, n, n);
    } else {
        scanLargeDeviceArray(d_out, d_in, n, true);
    }
    cudaDeviceSynchronize();
}

