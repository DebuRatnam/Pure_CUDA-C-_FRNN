#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>

extern "C" void run_prefix_sum(int* d_out, int* d_in, int n) {
    // Wrap raw pointers so Thrust can handle them on the GPU
    thrust::device_ptr<int> dev_in(d_in);
    thrust::device_ptr<int> dev_out(d_out);
    
    // This performs the exact same parallel prefix sum as your other repo
    thrust::exclusive_scan(dev_in, dev_in + n, dev_out);
    
    cudaDeviceSynchronize();
}
