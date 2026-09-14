// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

void transpose_add(const float* input,
                   float* output,
                   int rows,
                   int columns,
                   float row_scale,
                   float column_scale,
                   cudaStream_t stream);
