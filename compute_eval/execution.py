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
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from compute_eval.data.data_model import (
    SOLUTION_SCHEMA_VERSION,
    GradedSolution,
    Problem,
    Solution,
)
from compute_eval.utils.eval_utils import parse_semver


@contextmanager
def _work_dir_context(task_id: str):
    """
    Context manager for a work directory that in DEBUG mode is persistent
    and named exactly as `identifier` in the current working directory.

    Non-DEBUG mode uses a TemporaryDirectory with automatic cleanup.

    Args:
        task_id (str): Identifier for the work directory, typically the task ID.
    """
    debug_mode = os.environ.get("DEBUG", "0") == "1"

    if debug_mode:
        base_path = Path.cwd()
        tmpdir_path = base_path / task_id

        if tmpdir_path.exists():
            print(f"ERROR: debug temp directory already exists: {tmpdir_path}\n")
            raise FileExistsError(f"Refusing to overwrite existing debug temp directory: {tmpdir_path}")

        tmpdir_path.mkdir(parents=True, exist_ok=False)
        yield tmpdir_path
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)


def evaluate_solution(
    installed_ctk_major: int,
    installed_ctk_minor: int,
    problem: Problem,
    solution: Solution,
) -> GradedSolution:
    start_time = time.time()
    # Verify that the Solution is generated with the current schema version
    if solution.schema_version != SOLUTION_SCHEMA_VERSION:
        elapsed = time.time() - start_time
        return GradedSolution(
            task_id=solution.task_id,
            solution=solution,
            problem=problem,
            passed=False,
            skipped=True,
            elapsed_time=elapsed,
            build_output=f"[SCHEMA VERSION MISMATCH] Solution schema version {solution.schema_version} does not match expected version {SOLUTION_SCHEMA_VERSION}.",
        )

    # Check CUDA toolkit version compatibility (if applicable)
    if (required_ctk := parse_semver(problem.min_cuda_toolkit)) is not None:
        required_ctk_major, required_ctk_minor, _ = required_ctk
        if (installed_ctk_major, installed_ctk_minor) < (required_ctk_major, required_ctk_minor):
            elapsed = time.time() - start_time
            return GradedSolution(
                task_id=solution.task_id,
                solution=solution,
                problem=problem,
                passed=False,
                skipped=True,
                elapsed_time=elapsed,
            )

    # Check preconditions -- task ids must match, the solution must have source files, and the solution must not
    # attempt to overwrite or modify the unseen test files.
    if not solution.validate(problem):
        elapsed = time.time() - start_time
        return GradedSolution(
            task_id=solution.task_id,
            solution=solution,
            problem=problem,
            passed=False,
            skipped=False,
            elapsed_time=elapsed,
            build_output="[VALIDATION ERROR] Solution failed validation checks.",
        )

    with _work_dir_context(problem.task_id.replace("/", "-")) as workdir_path:
        # Write context files (public) from Problem to workdir
        for cf in problem.context_files:
            file_path = workdir_path / cf.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(cf.content)

        # Write test files (private) from Problem to workdir
        for tf in problem.test_files:
            file_path = workdir_path / tf.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(tf.content)

        # Apply the Solution to the workdir.  Note that these may intentionally overwrite context files.
        solution.setup_workspace(workdir_path)

        build_output = None
        # Run build command (if set)
        if problem.build_command:
            try:
                result = subprocess.run(
                    problem.build_command,
                    shell=True,
                    cwd=workdir_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                build_output = result.stdout + "\n" + result.stderr
            except subprocess.CalledProcessError as e:
                elapsed = time.time() - start_time
                return GradedSolution(
                    task_id=solution.task_id,
                    solution=solution,
                    problem=problem,
                    passed=False,
                    skipped=False,
                    elapsed_time=elapsed,
                    build_output=f"[BUILD ERROR]\n{e.stdout}\n{e.stderr}",
                )

        # Validate the Solution passes the Problem's source_references (if any)
        if not solution.verify_source_references(problem.source_references):
            elapsed = time.time() - start_time
            return GradedSolution(
                task_id=solution.task_id,
                solution=solution,
                problem=problem,
                passed=False,
                skipped=False,
                elapsed_time=elapsed,
                build_output=build_output,
                test_output="[VALIDATION ERROR] Solution does not include required source references.",
            )

        # Run test command
        try:
            result = subprocess.run(
                problem.test_command,
                shell=True,
                cwd=workdir_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=problem.timeout_seconds,
            )
            passed = True
            test_output = result.stdout + "\n" + result.stderr
        except subprocess.CalledProcessError as e:
            passed = False
            test_output = e.stdout + "\n" + e.stderr
        except subprocess.TimeoutExpired as e:
            passed = False
            test_output = f"[TIMEOUT EXPIRED after {e.timeout} seconds]\n{e.stdout}\n{e.stderr}"

        elapsed_time = time.time() - start_time

        # Return graded result
        return GradedSolution(
            task_id=solution.task_id,
            solution=solution,
            problem=problem,
            passed=passed,
            skipped=False,
            elapsed_time=elapsed_time,
            build_output=build_output,
            test_output=test_output,
        )
