// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "pair_transform.h"

#include <cuda_runtime.h>

#include <cstdlib>

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::abort();
    }
}

__global__ void pair_transform_kernel(const float* left,
                                      const float* right,
                                      float* sums,
                                      float* differences,
                                      int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        sums[index] = left[index] + right[index];
        differences[index] = left[index] - right[index];
    }
}

}  // namespace

void pair_transform(const float* left,
                    const float* right,
                    float* sums,
                    float* differences,
                    int count) {
    if (count <= 0) {
        return;
    }
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    float* device_left = nullptr;
    float* device_right = nullptr;
    float* device_sums = nullptr;
    float* device_differences = nullptr;
    check(cudaMalloc(&device_left, bytes));
    check(cudaMalloc(&device_right, bytes));
    check(cudaMalloc(&device_sums, bytes));
    check(cudaMalloc(&device_differences, bytes));
    check(cudaMemcpy(device_left, left, bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_right, right, bytes, cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    pair_transform_kernel<<<blocks, threads>>>(
        device_left, device_right, device_sums, device_differences, count);
    check(cudaMemcpy(sums, device_sums, bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(differences, device_differences, bytes, cudaMemcpyDeviceToHost));

    check(cudaFree(device_left));
    check(cudaFree(device_right));
    check(cudaFree(device_sums));
    check(cudaFree(device_differences));
}
