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
from typing import Dict, List, Tuple

import numpy as np
import tqdm

from compute_eval.data import (
    EvaluatedSample,
    read_problems,
    stream_samples,
    write_jsonl,
)
from compute_eval.execution import check_correctness

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
    sample_file: str,
    problem_file: str,
    allow_execution: bool = False,
    k: tuple[int] = (1, 10, 100),
    n_workers: int = 4,
    timeout: float = 60.0,
    results_file: str | None = None,
):
    """
    Evaluates the functional correctness of generated samples and writes results.

    Args:
        sample_file (str): Path to the sample file.
        problem_file (str): Path to the problem file.
        allow_execution (bool): Whether to allow execution of untrusted code.
        k (Tuple[int]): Tuple of k values for evaluation.
        n_workers (int): Number of worker threads.
        timeout (float): Timeout for each task in seconds.
        results_file (str | None): Path to the output results file. If None, defaults to sample_file with '_correctness_results.jsonl' suffix.

    Returns:
        None
    """

    if not allow_execution:
        raise RuntimeError(WARNING_MSG)

    # Check if only one k value was passed in (as an integer)
    # Multiple k values (tuple) is converted to a list of int
    k_vals = [k] if isinstance(k, int) else list(k)

    problems = read_problems(problem_file)
    keyed_problems = {p.task_id: p for p in problems}

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        results: list[EvaluatedSample] = []

        print("Reading samples...")
        samples = list(stream_samples(sample_file))

        # Verify that each problem is attempted at least once
        task_ids = set(p.task_id for p in problems)
        test_ids = set(s.task_id for s in samples)
        missing_ids = task_ids - test_ids
        if missing_ids:
            raise ValueError(f"The following task_ids are missing in the samples: {missing_ids}")

        for sample in tqdm.tqdm(samples):
            task_id = sample.task_id
            problem = keyed_problems.get(task_id)

            args = (problem, sample, timeout)
            future = executor.submit(check_correctness, *args)
            futures.append(future)

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    pass_at_k = estimate_metrics(results, k_vals)
    write_metrics(
        results,
        pass_at_k,
        results_file or os.path.splitext(sample_file)[0] + "_correctness_results.jsonl",
    )


def estimate_metrics(results: list[EvaluatedSample], k_vals: list[int]) -> dict[str, float]:
    """
    Estimates the metrics for the given problem.

    Args:
        results (List[EvaluatedSample]): List of evaluated samples.
        k_vals (List[int]): List of k values for evaluation.

    Returns:
        Dict[str, float]: A dictionary containing the estimated metrics.
    """
    # Calculate pass@k.
    total, correct = [], []

    # Group results by task_id
    results_by_task = defaultdict(list)
    for result in results:
        results_by_task[result.task_id].append(result)

    for task_id, results in results_by_task.items():
        passed = [r.passed for r in results if not r.skipped]

        # If all test cases are skipped, we skip the problem.
        if len(passed) == 0:
            print(
                f"Skipping problem {task_id}, it would be ignored while calculating pass@k. Possible reasons maybe incompatible GPU architecture."
            )
            continue
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    return {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in k_vals if (total >= k).all()}


def write_metrics(
    results: list[EvaluatedSample],
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
    write_jsonl(results_file, (r.__dict__ for r in results))

    # Output structured JSON to stdout
    output = {
        "pass_at_k": {k: float(v) for k, v in pass_at_k.items()},
        "problem_count": len(set(r.task_id for r in results)),
    }
    print(json.dumps(output, indent=2))
