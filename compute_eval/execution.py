# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file from human-eval (https://github.com/openai/human-eval/).
#
# The MIT License
#
# Copyright (c) OpenAI (https://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from compute_eval.data.data_model import (
    GradedSolution,
    Problem,
    Solution,
)
from compute_eval.evaluators import create_evaluator


def evaluate_solutions(
    problem: Problem,
    solutions: list[Solution],
    eval_mode: str,
    profile_mode: str | None = None,
) -> list[GradedSolution]:
    """
    Evaluates a list of solutions for a given problem.

    Args:
        problem: The problem specification
        solutions: List of solutions to evaluate
        eval_mode: Docker or local mode
        profile_mode: Optional profiling mode for performance analysis of problems that support it

    Returns:
        List of graded solutions with evaluation results

    Raises:
        ValueError: If cuda_toolkit version is invalid or mode is unknown
        RuntimeError: If no GPUs are available
    """
    evaluator = create_evaluator(problem=problem, eval_mode=eval_mode, profile_mode=profile_mode)
    return evaluator.evaluate_solutions(problem, solutions)
