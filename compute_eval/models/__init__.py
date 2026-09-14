# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public schema namespace for compute-eval.

Re-exports the Problem/Solution/Datapack data model so external consumers
(generator authors, dataset tooling) have a single, stable import path.

Internal code should continue to import from `compute_eval.data.*` — this
module is purely for the public API surface.
"""

from compute_eval.data.data_model import (
    PROBLEM_SCHEMA_VERSION,
    SOLUTION_SCHEMA_VERSION,
    CudaCppProblem,
    CudaPythonProblem,
    FileSolution,
    GradedSolution,
    Metadata,
    PatchSolution,
    Problem,
    ReleaseVersion,
    SanitizerResult,
    SanitizerStatus,
    SanitizerTool,
    SanitizerToolResult,
    Solution,
    SourceFile,
    SourceReferences,
    ValidGroup,
)
from compute_eval.data.data_pack import (
    DatapackMetadata,
    ProblemDatapack,
    ProblemDatapackMetadata,
    SolutionDatapack,
    SolutionDatapackMetadata,
)
from compute_eval.data.metrics_data_model import (
    CustomTimingMode,
    PerformanceMetrics,
    ProcessTimingMode,
    TimingModes,
)

__all__ = [
    # Problem types
    "Problem",
    "CudaCppProblem",
    "CudaPythonProblem",
    "SourceFile",
    "Metadata",
    "SourceReferences",
    "ValidGroup",
    "ReleaseVersion",
    "TimingModes",
    "ProcessTimingMode",
    "CustomTimingMode",
    # Solution types
    "Solution",
    "FileSolution",
    "PatchSolution",
    # Datapack types
    "ProblemDatapack",
    "SolutionDatapack",
    "DatapackMetadata",
    "ProblemDatapackMetadata",
    "SolutionDatapackMetadata",
    # Evaluation results
    "GradedSolution",
    "SanitizerTool",
    "SanitizerStatus",
    "SanitizerToolResult",
    "SanitizerResult",
    "PerformanceMetrics",
    # Schema version constants
    "PROBLEM_SCHEMA_VERSION",
    "SOLUTION_SCHEMA_VERSION",
]
