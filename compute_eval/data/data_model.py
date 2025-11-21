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
    model_validator,
)

from compute_eval.utils.parsing import get_most_likely_language

PROBLEM_SCHEMA_VERSION = 2
SOLUTION_SCHEMA_VERSION = 1
SOURCE_ENCODING = "utf-8"


class ReleaseVersion(str, Enum):
    INTERNAL = "internal"
    V2025_1 = "2025-1"
    V2025_2 = "2025-2"
    V2025_3 = "2025-3"


class SourceFile(BaseModel):
    path: str
    content: str


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

    context_files: list[SourceFile] = Field(default_factory=list)
    test_files: list[SourceFile] = Field(default_factory=list)

    source_references: str | list[str] | SourceReferences | None = None

    build_command: str | None = None
    test_command: str

    min_cuda_toolkit: str | None = None
    timeout_seconds: float | None = None
    metadata: Metadata | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_to_concrete(cls, data):
        if isinstance(data, cls) or cls is not Problem:
            return data

        adapter = TypeAdapter(Annotated[CudaCppProblem | CudaPythonProblem, Field(discriminator="type")])
        return adapter.validate_python(data)


class CudaCppProblem(Problem):
    type: Literal["cuda_cpp"] = "cuda_cpp"

    arch_list: list[str] = Field(default_factory=list)


class CudaPythonProblem(Problem):
    type: Literal["cuda_python"] = "cuda_python"

    python_version: str | None = None


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
    elapsed_time: float
    solution: Solution
    problem: Problem
    build_output: str | None = None
    test_output: str | None = None

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
