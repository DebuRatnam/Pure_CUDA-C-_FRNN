#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

template <typename T>
class DeviceBuffer {
private:
    T* d_ptr = nullptr;
    size_t _count = 0;
public:
    DeviceBuffer(size_t count) : _count(count) {
        if (cudaMalloc(&d_ptr, _count * sizeof(T)) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory.");
        }
    }
    ~DeviceBuffer() { if (d_ptr) cudaFree(d_ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete; // No accidental copies
    T* data() { return d_ptr; }
    size_t count() const { return _count; }
    void upload(const T* h_ptr) { cudaMemcpy(d_ptr, h_ptr, _count * sizeof(T), cudaMemcpyHostToDevice); }
    void download(T* h_ptr) const { cudaMemcpy(h_ptr, d_ptr, _count * sizeof(T), cudaMemcpyDeviceToHost); }
};
