# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from compute_eval.evaluation.evaluate import (
        _empty_solution,
        estimate_pass_at_k,
        evaluate_functional_correctness,
    )
    from compute_eval.evaluation.evaluators import SolutionEvaluator, create_evaluator
    from compute_eval.evaluation.profilers import PerformanceProfiler, create_profiler
except ImportError as e:
    raise ImportError(
        f"compute_eval.evaluation requires optional dependencies: {e}. "
        "Install them with: pip install compute-eval[eval]"
    ) from e

__all__ = [
    "_empty_solution",
    "evaluate_functional_correctness",
    "estimate_pass_at_k",
    "create_evaluator",
    "SolutionEvaluator",
    "create_profiler",
    "PerformanceProfiler",
]
