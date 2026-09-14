// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "masked_affine.h"

__global__ void masked_affine_kernel(const float* input,
                                     const unsigned char* mask,
                                     float* output,
                                     int count,
                                     float scale,
                                     float bias) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += blockDim.x * gridDim.x) {
        output[index] = mask[index] ? scale * input[index] + bias : input[index];
    }
}

void masked_affine(const float* input,
                   const unsigned char* mask,
                   float* output,
                   int count,
                   float scale,
                   float bias,
                   cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    constexpr int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    masked_affine_kernel<<<blocks, threads, 0, stream>>>(input, mask, output, count, scale, bias);
}
