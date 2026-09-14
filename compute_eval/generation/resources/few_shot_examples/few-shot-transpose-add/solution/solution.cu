// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_add.h"

__global__ void transpose_add_kernel(const float* input,
                                     float* output,
                                     int rows,
                                     int columns,
                                     float row_scale,
                                     float column_scale) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && column < columns) {
        output[column * rows + row] =
            input[row * columns + column] + row * row_scale + column * column_scale;
    }
}

void transpose_add(const float* input,
                   float* output,
                   int rows,
                   int columns,
                   float row_scale,
                   float column_scale,
                   cudaStream_t stream) {
    if (rows <= 0 || columns <= 0) {
        return;
    }
    const dim3 threads(16, 16);
    const dim3 blocks((columns + 15) / 16, (rows + 15) / 16);
    transpose_add_kernel<<<blocks, threads, 0, stream>>>(
        input, output, rows, columns, row_scale, column_scale);
}
