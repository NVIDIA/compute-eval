// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "pair_transform.h"

#include <cassert>
#include <cmath>
#include <vector>

void run_case(int count) {
    std::vector<float> left(count);
    std::vector<float> right(count);
    for (int index = 0; index < count; ++index) {
        left[index] = static_cast<float>((index * 5) % 17 - 8) / 3.0F;
        right[index] = static_cast<float>((index * 13) % 29 - 14) / 6.0F;
    }
    std::vector<float> sums(count);
    std::vector<float> differences(count);
    pair_transform(left.data(), right.data(), sums.data(), differences.data(), count);
    for (int index = 0; index < count; ++index) {
        assert(std::fabs(sums[index] - (left[index] + right[index])) <= 1.0e-6F);
        assert(std::fabs(differences[index] - (left[index] - right[index])) <= 1.0e-6F);
    }
}

int main() {
    run_case(0);
    run_case(1);
    run_case(257);
    return 0;
}
