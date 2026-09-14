// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "masked_affine.h"

#include <cassert>
#include <cmath>
#include <vector>

void run_case(int count, float scale, float bias) {
    if (count == 0) {
        masked_affine(nullptr, nullptr, nullptr, 0, scale, bias, nullptr);
        return;
    }
    std::vector<float> input(count);
    std::vector<unsigned char> mask(count);
    for (int index = 0; index < count; ++index) {
        input[index] = static_cast<float>((index * 11) % 31 - 15) / 4.0F;
        mask[index] = static_cast<unsigned char>((index % 3) != 0);
    }
    float* device_input = nullptr;
    unsigned char* device_mask = nullptr;
    float* device_output = nullptr;
    const size_t float_bytes = static_cast<size_t>(count) * sizeof(float);
    assert(cudaMalloc(&device_input, float_bytes) == cudaSuccess);
    assert(cudaMalloc(&device_mask, static_cast<size_t>(count)) == cudaSuccess);
    assert(cudaMalloc(&device_output, float_bytes) == cudaSuccess);
    assert(cudaMemcpy(device_input, input.data(), float_bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(device_mask, mask.data(), count, cudaMemcpyHostToDevice) == cudaSuccess);
    masked_affine(device_input, device_mask, device_output, count, scale, bias, nullptr);

    std::vector<float> output(count);
    assert(cudaMemcpy(output.data(), device_output, float_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int index = 0; index < count; ++index) {
        const float expected = mask[index] ? scale * input[index] + bias : input[index];
        assert(std::fabs(output[index] - expected) <= 1.0e-6F);
    }
    assert(cudaFree(device_input) == cudaSuccess);
    assert(cudaFree(device_mask) == cudaSuccess);
    assert(cudaFree(device_output) == cudaSuccess);
}

int main() {
    run_case(0, 2.0F, -1.0F);
    run_case(17, -0.5F, 3.0F);
    run_case(1025, 1.25F, -2.5F);
    return 0;
}
