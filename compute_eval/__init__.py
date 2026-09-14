# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""compute-eval public API.

Generator authors and other library consumers should import the schema types
from this package directly (`from compute_eval import Problem`) or from the
explicit `compute_eval.models` namespace.

Internal modules continue to import from `compute_eval.data.*` — that is the
implementation location, not the public surface.
"""

from collections.abc import Callable

from compute_eval.models import *  # noqa: F401, F403  — re-export public schema
from compute_eval.models import __all__ as _MODELS_ALL

# Execution result type.
#
# Tuple of (exit_code, timed_out, output)
EvaluatorRuntimeResult = tuple[int, bool, str]

# Evaluator runtime function type.
#
# Callable that takes (command: str, timeout_seconds: float, task_tag: str | None)
# Returns ExecutionResult
EvaluatorRuntime = Callable[[str, float, str | None], EvaluatorRuntimeResult]

__all__ = [
    *_MODELS_ALL,
    "EvaluatorRuntime",
    "EvaluatorRuntimeResult",
]
