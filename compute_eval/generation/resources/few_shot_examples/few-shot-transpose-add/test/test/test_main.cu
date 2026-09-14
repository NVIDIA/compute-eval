// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_add.h"

#include <cassert>
#include <cmath>
#include <vector>

void run_case(int rows, int columns) {
    if (rows == 0 || columns == 0) {
        transpose_add(nullptr, nullptr, rows, columns, 0.5F, -0.25F, nullptr);
        return;
    }
    const int count = rows * columns;
    std::vector<float> input(count);
    for (int index = 0; index < count; ++index) {
        input[index] = static_cast<float>((index * 7) % 23 - 11);
    }
    float* device_input = nullptr;
    float* device_output = nullptr;
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    assert(cudaMalloc(&device_input, bytes) == cudaSuccess);
    assert(cudaMalloc(&device_output, bytes) == cudaSuccess);
    assert(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    transpose_add(device_input, device_output, rows, columns, 0.5F, -0.25F, nullptr);

    std::vector<float> output(count);
    assert(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const float expected = input[row * columns + column] + 0.5F * row - 0.25F * column;
            assert(std::fabs(output[column * rows + row] - expected) <= 1.0e-6F);
        }
    }
    assert(cudaFree(device_input) == cudaSuccess);
    assert(cudaFree(device_output) == cudaSuccess);
}

int main() {
    run_case(0, 4);
    run_case(1, 1);
    run_case(7, 19);
    run_case(33, 18);
    return 0;
}
