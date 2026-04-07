#include "cuda_runtime.h"

void _checkCudaError(const char *message, cudaError_t err, const char *caller);

// Move these OUTSIDE of the extern "C" block
void printResult(const char* prefix, int result, long nanoseconds);
void printResult(const char* prefix, int result, float milliseconds);

// ONLY keep the specific helpers the CUDA/Scan code needs here
extern "C" {
    int nextPowerOfTwo(int x);
    long get_nanos();
}

bool isPowerOfTwo(int x);