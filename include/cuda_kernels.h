#pragma once

// CUDA kernel wrapper functions
float* allocateDeviceMemory(size_t size);
void freeDeviceMemory(float *d_ptr);
void addArrays(float *d_c, const float *d_a, const float *d_b, int n);
