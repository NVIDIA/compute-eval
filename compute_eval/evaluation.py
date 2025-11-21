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

import itertools
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tqdm

from compute_eval.data.data_model import (
    PROBLEM_SCHEMA_VERSION,
    SOLUTION_SCHEMA_VERSION,
    GradedSolution,
    Problem,
)
from compute_eval.data.data_pack import ProblemDatapack, SolutionDatapack
from compute_eval.data.utils import write_graded_solutions
from compute_eval.execution import evaluate_solution
from compute_eval.utils.eval_utils import get_nvcc_version, parse_semver

WARNING_MSG = """===================
     WARNING
===================

Evaluation of correctness or performance will execute untrusted model-generated
code.

Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.

Users are strongly encouraged to sandbox this evaluation suite so that it does
not perform destructive actions on their host or network.

In order to execute this code you must explicitly pass the --allow-execution flag.
"""


def estimate_pass_at_k(
    num_samples: int | list[int] | np.ndarray,
    num_correct: list[int] | np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.

    Args:
        num_samples: Number of samples for each problem
        num_correct: Number of correct samples for each problem
        k: The k value for pass@k calculation

    Returns:
        Array of pass@k estimates for each problem
    """
    if k <= 0:
        raise ValueError("k must be positive")

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct, strict=False)])


def evaluate_functional_correctness(
    solutions_datapack: str,
    problems_datapack_dir: str,
    allow_execution: bool,
    k: tuple[int] | int,
    n_workers: int,
    results_file: str | None,
):
    """
    Evaluates the functional correctness of generated solutions and writes results.

    Args:
        solutions_datapack (str): Path to the solution datapack.
        problems_datapack_dir (str): Directory containing problem datapacks.
        allow_execution (bool): Whether to allow execution of untrusted code.
        k (Tuple[int] | int): Tuple of k values for evaluation or single k value (default: 1).
        n_workers (int): Number of worker threads.
        results_file (str | None): Path to output results file.

    Returns:
        None
    """
    if not allow_execution:
        raise RuntimeError(WARNING_MSG)

    if (installed_ctk_version := parse_semver(get_nvcc_version())) is None:
        raise RuntimeError("Could not determine CUDA toolkit version from nvcc.")

    installed_ctk_major, installed_ctk_minor, _ = installed_ctk_version

    # Check if only one k value was passed in (as an integer)
    # Multiple k values (tuple) is converted to a list of int
    k_vals = [k] if isinstance(k, int) else list(k)

    with SolutionDatapack(solutions_datapack) as datapack:
        release = datapack.metadata.release

        print("Reading solutions...")
        solutions = list(datapack.read_items())

        # Verify that all solutions are for the current schema version
        if any(s.schema_version != SOLUTION_SCHEMA_VERSION for s in solutions):
            raise ValueError(
                f"One or more solutions in {solutions_datapack} do not match the expected schema version {SOLUTION_SCHEMA_VERSION}."
            )

    problems_file = os.path.join(problems_datapack_dir, f"{release.value}-problems.tar.gz")
    with ProblemDatapack(problems_file) as datapack:
        # Sanity check: ensure the problems datapack matches the solutions datapack release
        if datapack.metadata.release != release:
            raise ValueError(
                f"Problems datapack release {datapack.metadata.release} does not match solutions datapack release {release}."
            )

        print("Reading problems...")
        problems = list(datapack.read_items())
        keyed_problems: dict[str, Problem] = {p.task_id: p for p in problems}

        # Verify that all problems are for the current schema version
        if any(p.schema_version != PROBLEM_SCHEMA_VERSION for p in keyed_problems.values()):
            raise ValueError(
                f"One or more problems in {problems_file} do not match the expected schema version {PROBLEM_SCHEMA_VERSION}."
            )

    # Verify that each problem is attempted at least once
    task_ids = set(p.task_id for p in problems)
    test_ids = set(s.task_id for s in solutions)

    missing_ids = task_ids - test_ids
    if missing_ids:
        raise ValueError(f"The following task_ids are missing in the solutions: {missing_ids}")

    # Check the generated solutions against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        results: list[GradedSolution] = []

        for solution in tqdm.tqdm(solutions):
            task_id = solution.task_id
            problem = keyed_problems.get(task_id)

            args = (installed_ctk_major, installed_ctk_minor, problem, solution)
            future = executor.submit(evaluate_solution, *args)
            futures.append(future)

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    pass_at_k = estimate_metrics(results, k_vals)
    write_metrics(
        results,
        pass_at_k,
        results_file or f"{release.value}-graded-solutions.jsonl",
    )


def estimate_metrics(results: list[GradedSolution], k_vals: list[int]) -> dict[str, float]:
    """
    Estimates the metrics for the given solutions.

    Args:
        results (List[GradedSolution]): List of graded solutions
        k_vals (List[int]): List of k values for evaluation.

    Returns:
        Dict[str, float]: A dictionary containing the estimated metrics.
    """
    # Calculate pass@k.
    total, correct = [], []

    # Group results by task_id
    results_by_task = defaultdict(list)
    for result in results:
        results_by_task[result.solution.task_id].append(result)

    skipped = 0
    for _, results in results_by_task.items():
        total.append(len(results))
        correct.append(sum(r.passed for r in results))
        skipped += all(r.skipped for r in results)
    total = np.array(total)
    correct = np.array(correct)

    return {
        "skipped": float(skipped),
        **{f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in k_vals if (total >= k).all()},
    }


def write_metrics(
    results: list[GradedSolution],
    pass_at_k: dict[str, float],
    results_file: str,
) -> None:
    """
    Writes the metrics to a file and prints consolidated output.

    Args:
        results (list[EvaluatedSample]): List of evaluated samples.
        pass_at_k (Dict[str, float]): Pass@k metrics.
        results_file (str): Path to the output results file.

    Returns:
        None
    """
    # Finally, save the results in one file:
    print(f"Writing results to {results_file}...")
    write_graded_solutions(results_file, results)

    # Output structured JSON to stdout
    output = {
        "pass_at_k": {k: float(v) for k, v in pass_at_k.items()},
        "problem_count": len(set(r.solution.task_id for r in results)),
    }
    print(json.dumps(output, indent=2))
