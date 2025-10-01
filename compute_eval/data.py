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

import gzip
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, fields


def _construct(instance, **kwargs):
    for key, value in kwargs.items():
        if key in {f.name for f in fields(instance)}:
            setattr(instance, key, value)


@dataclass
class Problem(ABC):
    task_id: str
    prompt: str
    declaration: str
    test: str
    example_test: str
    solution: str | None = None

    @abstractmethod
    def problem_type(self) -> str:
        pass


@dataclass
class CProblem(Problem):
    cc_flags: str | None = None
    ld_flags: str | None = None
    cuda_toolkit: str | None = None

    def __init__(self, **kwargs):
        _construct(self, **kwargs)

    def problem_type(self) -> str:
        return "CUDA"

    def cli_args(self) -> str:
        cli_args = ""
        if self.cc_flags is not None:
            cli_args += " " + self.cc_flags
        if self.ld_flags is not None:
            cli_args += " " + self.ld_flags

        return cli_args

@dataclass
class Sample:
    task_id: str
    sample_idx: int
    compilable_code: str

    prompt: str = ""
    generated_completion: str = ""

    def __init__(self, **kwargs):
        _construct(self, **kwargs)


@dataclass
class EvaluatedSample(Sample):
    skipped: bool = True
    passed: bool = False
    elapsed_time: float = -1.0
    result: str = ""

    def __init__(
        self,
        sample: Sample,
        skipped: bool,
        passed: bool,
        elapsed_time: float = -1.0,
        result: str = "",
    ):
        super().__init__(**sample.__dict__)
        self.skipped = skipped
        self.passed = passed
        self.elapsed_time = elapsed_time
        self.result = result


def read_problems(file_path: str) -> list[Problem]:
    problems = []
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            problem_type = data.get("type", "c_problem")
            if problem_type == "c_problem":
                problems.append(CProblem(**data))

    return problems


def stream_samples(filename: str) -> Iterable[Sample]:
    for index, item in enumerate(stream_jsonl(filename)):
        yield Sample(sample_idx=index, **item)


def stream_jsonl(filename: str) -> Iterable[dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp, gzip.open(gzfp, "rt") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line, strict=False)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line, strict=False)


def write_jsonl(filename: str, data: Iterable[dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    mode = "ab" if append else "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp, gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
            for x in data:
                if x:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                if x:
                    fp.write((json.dumps(x) + "\n").encode("utf-8"))
