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

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_serializer,
    model_validator,
)

from compute_eval.data.metrics_data_model import PerformanceMetrics, ProcessTimingMode, TimingModes
from compute_eval.utils.parsing import get_most_likely_language

PROBLEM_SCHEMA_VERSION = 2
SOLUTION_SCHEMA_VERSION = 1
SOURCE_ENCODING = "utf-8"


class ReleaseVersion(str, Enum):
    INTERNAL = "internal"
    V2025_1 = "2025-1"
    V2025_2 = "2025-2"
    V2025_3 = "2025-3"
    V2026_1 = "2026-1"


class SourceFile(BaseModel):
    path: str
    content: str


#: Valid group names for categorizing problems by topic.
#:
#: - cuda-runtime: CUDA Runtime & Execution Model (kernel launch, memory management,
#:   streams, events, CUDA Graphs, cluster launch, occupancy)
#: - cuda-kernels: CUDA Kernel Programming (shared memory, warp intrinsics, reductions,
#:   stencils, tensor cores, cooperative groups, applied GPU computation)
#: - cccl: CCCL (CUDA C++ Core Libraries) - Thrust, CUB, and libcu++
#: - cublas: cuBLAS - Dense linear algebra (BLAS levels 1-3, extensions)
#: - mathlibs: Math Libraries - cuSPARSE, cuSOLVER, cuFFT, cuRAND
#: - cutile: cuTile - Tile-based programming with cuTile kernels and patterns
#: - cudnn: cuDNN - Deep Neural Network library (convolutions, pooling,
#:   normalization, activations using cuDNN Graph API)
ValidGroup = Literal[
    "cuda-runtime",
    "cuda-kernels",
    "cccl",
    "cublas",
    "mathlibs",
    "cutile",
    "cudnn",
]


class Metadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    difficulty: str | None = None
    tags: list[str] = Field(default_factory=list)
    releases: list[ReleaseVersion] = Field(default_factory=list)

    do_not_release: bool = Field(default=False)


class SourceReferences(BaseModel):
    """Model for source references with any/all semantics.

    Can be:
    - {'any': ['ref1', 'ref2']} - at least one reference must be present
    - {'all': ['ref1', 'ref2']} - all references must be present
    - {'any': ['ref1', 'ref2'], 'all': ['ref3', 'ref4']} - combined logic:
        ALL references in 'all' must be present AND at least ONE from 'any' must be present
    """

    any: list[str] | None = None
    all: list[str] | None = None

    @model_validator(mode="after")
    def validate_at_least_one_key(self):
        if self.any is None and self.all is None:
            raise ValueError("Must specify at least one of 'any' or 'all' in source_references")
        return self


class Problem(BaseModel, ABC):
    type: Literal["cuda_cpp", "cuda_python"]
    schema_version: int = Field(default=PROBLEM_SCHEMA_VERSION)

    task_id: str
    date: str
    prompt: str
    metadata: Metadata
    group: ValidGroup

    context_files: list[SourceFile] = Field(default_factory=list)
    test_files: list[SourceFile] = Field(default_factory=list)

    source_references: str | list[str] | SourceReferences | None = None

    build_command: str | None = None
    test_command: str
    benchmark_command: str | None = None

    timing_mode: TimingModes = Field(
        default_factory=ProcessTimingMode, description="Method for extracting performance timing from metrics"
    )

    min_cuda_toolkit: str | None = None
    compute_capability: str | None = Field(default="8.0")
    requires_datacenter_gpu: bool = Field(default=False)
    timeout_seconds: float | None = None

    baseline_solution: Annotated["FileSolution | PatchSolution", Field(discriminator="type")] | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_to_concrete(cls, data):
        if isinstance(data, cls) or cls is not Problem:
            return data

        adapter = TypeAdapter(Annotated[CudaCppProblem | CudaPythonProblem, Field(discriminator="type")])
        return adapter.validate_python(data)

    @model_serializer(mode="wrap", when_used="json")
    def _serialize_model(self, serializer, info):
        """Normalize source_references field for consistent serialization to HuggingFace datasets."""
        data = serializer(self)

        # Normalize source_references to a consistent format for HuggingFace dataset compatibility
        if "source_references" in data and data["source_references"] is not None:
            source_ref = self.source_references

            # Convert to normalized dict format with 'any' and 'all' keys
            normalized = {"any": None, "all": None}

            if isinstance(source_ref, str):
                # Single string -> treat as 'all' (must be present)
                normalized["all"] = [source_ref]
            elif isinstance(source_ref, list):
                # List of strings -> treat as 'all' (all must be present)
                normalized["all"] = source_ref
            elif isinstance(source_ref, SourceReferences):
                # Already in SourceReferences format
                normalized["any"] = source_ref.any
                normalized["all"] = source_ref.all

            data["source_references"] = normalized

        return data


class CudaCppProblem(Problem):
    type: Literal["cuda_cpp"] = "cuda_cpp"


class CudaPythonProblem(Problem):
    type: Literal["cuda_python"] = "cuda_python"


class Solution(BaseModel, ABC):
    model_config = ConfigDict(extra="allow")
    schema_version: int = Field(default=SOLUTION_SCHEMA_VERSION)

    type: Literal["file", "patch"]
    task_id: str

    @abstractmethod
    def validate(self, problem: Problem) -> bool:
        pass

    @abstractmethod
    def setup_workspace(self, work_dir: Path):
        pass

    @abstractmethod
    def verify_source_references(self, source_references: str | list[str] | SourceReferences | None) -> bool:
        pass

    @model_validator(mode="before")
    @classmethod
    def _upgrade_to_concrete(cls, data):
        if isinstance(data, cls) or cls is not Solution:
            return data

        adapter = TypeAdapter(Annotated[FileSolution | PatchSolution, Field(discriminator="type")])
        return adapter.validate_python(data)


class FileSolution(Solution):
    type: Literal["file"] = "file"

    files: list[SourceFile] = Field(default_factory=list)

    def validate(self, problem: Problem) -> bool:
        if self.task_id != problem.task_id:
            return False
        if not self.files:
            return False
        return not {f.path for f in self.files} & {tf.path for tf in problem.test_files}

    def setup_workspace(self, work_dir: Path):
        for file in self.files:
            file_path = work_dir / file.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file.content)

    def verify_source_references(self, source_references: str | list[str] | SourceReferences | None) -> bool:
        if source_references is None:
            return True

        # Parse the source_references format
        all_refs: list[str] = []
        any_refs: list[str] = []

        if isinstance(source_references, str):
            all_refs = [source_references]
        elif isinstance(source_references, list):
            all_refs = source_references
        elif isinstance(source_references, SourceReferences):
            if source_references.all is not None:
                all_refs = source_references.all
            if source_references.any is not None:
                any_refs = source_references.any

        if not all_refs and not any_refs:
            return True

        all_remaining = {ref.encode(SOURCE_ENCODING) for ref in all_refs}
        any_remaining = {ref.encode(SOURCE_ENCODING) for ref in any_refs}

        for file in self.files:
            encoded_content = file.content.encode(SOURCE_ENCODING)
            language = get_most_likely_language(file.path, encoded_content)
            if language is not None:
                for ref, _ in language.find_matching_subtrees(encoded_content, all_remaining | any_remaining):
                    all_remaining.discard(ref)
                    if ref in any_remaining:
                        any_remaining.clear()

                    if not all_remaining and not any_remaining:
                        return True

        return not all_remaining and not any_remaining


class PatchSolution(Solution):
    type: Literal["patch"] = "patch"

    patch: str

    def validate(self, problem: Problem) -> bool:
        if self.task_id != problem.task_id:
            return False
        return self.patch is not None

    def setup_workspace(self, work_dir: Path):
        # TODO: Apply the patch to the files in work_dir
        pass

    def verify_source_references(self, source_references: str | list[str] | SourceReferences | None) -> bool:
        # TODO: Need to fully implement patch solutions.
        return True


class GradedSolution(BaseModel):
    task_id: str
    passed: bool
    skipped: bool
    solution: Solution
    problem: Problem
    build_output: str | None = None
    test_output: str | None = None
    benchmark_output: str | None = None
    solution_metrics: PerformanceMetrics | None = None
    solution_time: float | None = None
    baseline_metrics: PerformanceMetrics | None = None
    baseline_time: float | None = None
    speedup: float | None = None

    @property
    def sm_throughput(self) -> float | None:
        """Get average SM throughput from performance metrics."""
        if self.solution_metrics is None:
            return None
        return self.solution_metrics.get_average_sm_throughput()

    @property
    def dram_throughput(self) -> float | None:
        """Get average DRAM throughput from performance metrics."""
        if self.solution_metrics is None:
            return None
        return self.solution_metrics.get_average_dram_throughput()

    @property
    def baseline_sm_throughput(self) -> float | None:
        """Get average SM throughput from baseline performance metrics."""
        if self.baseline_metrics is None:
            return None
        return self.baseline_metrics.get_average_sm_throughput()

    @property
    def baseline_dram_throughput(self) -> float | None:
        """Get average DRAM throughput from baseline performance metrics."""
        if self.baseline_metrics is None:
            return None
        return self.baseline_metrics.get_average_dram_throughput()

    @model_validator(mode="before")
    @classmethod
    def _upgrade_nested_union(cls, data):
        if "problem" in data and data["problem"] is not None:
            problem_adapter = TypeAdapter(Annotated[CudaCppProblem | CudaPythonProblem, Field(discriminator="type")])
            data["problem"] = problem_adapter.validate_python(data["problem"])
        if "solution" in data and data["solution"] is not None:
            solution_adapter = TypeAdapter(Annotated[FileSolution | PatchSolution, Field(discriminator="type")])
            data["solution"] = solution_adapter.validate_python(data["solution"])
        return data


Problem.model_rebuild()
CudaCppProblem.model_rebuild()
CudaPythonProblem.model_rebuild()
