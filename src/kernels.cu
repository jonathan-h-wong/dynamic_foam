#include <cuda_runtime.h>
#include <stdio.h>

// Simple example kernel: add two arrays
__global__ void addKernel(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Wrapper function to call the kernel from C++
void addArrays(float *d_c, const float *d_a, const float *d_b, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    addKernel<<<gridSize, blockSize>>>(d_c, d_a, d_b, n);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// Example: allocate device memory
float* allocateDeviceMemory(size_t size) {
    float *d_ptr;
    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA allocation error: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return d_ptr;
}

// Example: free device memory
void freeDeviceMemory(float *d_ptr) {
    if (d_ptr != nullptr) {
        cudaFree(d_ptr);
    }
}
