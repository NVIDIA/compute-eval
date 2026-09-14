// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

void masked_affine(const float* input,
                   const unsigned char* mask,
                   float* output,
                   int count,
                   float scale,
                   float bias,
                   cudaStream_t stream);
