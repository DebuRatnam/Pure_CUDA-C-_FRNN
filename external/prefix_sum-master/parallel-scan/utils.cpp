#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <ctime>
#include <iostream>

#include "cuda_runtime.h"

#include "utils.h"

void _checkCudaError(const char *message, cudaError_t err, const char *caller) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Error in: %s\n", caller);
		fprintf(stderr, message);
		fprintf(stderr, ": %s\n", cudaGetErrorString(err));
		exit(0);
	}
}

void printResult(const char* prefix, int result, long nanoseconds) {
	printf("  ");
	printf(prefix);
	printf(" : %i in %ld ms \n", result, nanoseconds / 1000);
}

void printResult(const char* prefix, int result, float milliseconds) {
	printf("  ");
	printf(prefix);
	printf(" : %i in %f ms \n", result, milliseconds);
}


// from https://stackoverflow.com/a/3638454
bool isPowerOfTwo(int x) {
	return x && !(x & (x - 1));
}

// WRAP THESE TWO IN EXTERN "C"
extern "C" {
    // from https://stackoverflow.com/a/12506181
    int nextPowerOfTwo(int x) {
        int power = 1;
        while (power < x) {
            power *= 2;
        }
        return power;
    }

    // from https://stackoverflow.com/a/36095407
    long get_nanos() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
    }
}
