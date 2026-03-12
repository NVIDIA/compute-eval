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
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

import numpy as np
import tqdm

from compute_eval.data.data_model import (
    PROBLEM_SCHEMA_VERSION,
    SOLUTION_SCHEMA_VERSION,
    GradedSolution,
    Problem,
    ReleaseVersion,
    Solution,
    ValidGroup,
)
from compute_eval.data.data_pack import ProblemDatapack, SolutionDatapack
from compute_eval.data.utils import write_graded_solutions
from compute_eval.execution import evaluate_solutions

WARNING_MSG = """===================
     WARNING
===================

You are about to execute untrusted code. This is a security risk.
Please ensure you understand the implications before proceeding.

To proceed, you must explicitly set mode to either 'docker' or 'local'.
"""


def geometric_mean(values: list[float]) -> float:
    """
    Calculate geometric mean using log-space to avoid overflow.

    Args:
        values: List of positive float values

    Returns:
        Geometric mean of the values, or 0.0 if empty list
    """
    if not values:
        return 0.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


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
    release: ReleaseVersion,
    solutions_datapack: str,
    problems_datapack_dir: str,
    mode: Literal["docker", "local"] | None,
    profile_mode: str | None,
    k: tuple[int] | int,
    n_workers: int,
):
    """
    Evaluates the functional correctness of generated solutions and writes results.

    Args:
        release (ReleaseVersion): The release version to evaluate solutions for.
        solutions_datapack (str): Path to a directory holding solutions datapacks or a single solutions datapack to evaluate.
        problems_datapack_dir (str): Directory containing problem datapacks.
        mode (Literal["docker", "local"] | None): Evaluation execution mode. Must be set to 'docker' or 'local' to allow execution.
        profile_mode (str | None): Profiling mode for performance analysis of problems that support it. Can be 'cupti', 'ncu', or none.
        k (Tuple[int] | int): Tuple of k values for evaluation or single k value (default: 1).
        n_workers (int): Number of worker threads.

    Raises:
        RuntimeError: If mode is not specified.
        ValueError: If schema versions do not match or required data is missing.
    """
    if not mode:
        raise RuntimeError(WARNING_MSG)

    k_vals = [k] if isinstance(k, int) else list(k)

    if os.path.isdir(solutions_datapack):
        # If a directory is provided for datapacks, find all files that match the *-solutions.tar.gz pattern
        datapacks = [
            f"{solutions_datapack}/{f}" for f in os.listdir(solutions_datapack) if f.endswith("-solutions.tar.gz")
        ]
        if not datapacks:
            raise ValueError(f"No solutions datapacks found in directory {solutions_datapack}.")
    else:
        datapacks = [solutions_datapack]

    solutions = []
    groups = None
    for path in datapacks:
        with SolutionDatapack(path) as datapack:
            if datapack.metadata.release != release:
                raise ValueError(
                    f"Solutions datapack release {datapack.metadata.release} does not match expected release {release} for datapack {path}."
                )

            print(f"Reading solutions from {path}...")
            solutions.extend(datapack.read_items())

            if groups is None:
                groups = datapack.metadata.groups
            elif set(groups) != set(datapack.metadata.groups):
                raise ValueError(
                    f"Solutions datapack {path} has groups {datapack.metadata.groups} which do not match groups {groups} from previous datapacks. Ensure all datapacks have the same groups or specify a single datapack to evaluate."
                )

            # Verify that all solutions are for the current schema version
            if any(s.schema_version != SOLUTION_SCHEMA_VERSION for s in solutions):
                raise ValueError(
                    f"One or more solutions in {path} do not match the expected schema version {SOLUTION_SCHEMA_VERSION}."
                )

    problems_file = os.path.join(problems_datapack_dir, f"{release.value}-problems.tar.gz")

    # Load the ProblemDatapack
    with ProblemDatapack(problems_file, include=groups, exclude=None) as datapack:
        # Sanity check: ensure the problems datapack matches the solutions datapack release
        if datapack.metadata.release != release:
            raise ValueError(
                f"Problems datapack release {datapack.metadata.release} does not match solutions datapack release {release}."
            )

        print("Reading problems...")
        problems = list(datapack.read_items())

        if len(problems) == 0:
            raise ValueError(
                f"No valid problems found using groups '{groups}'. "
                "Check that any group names are correct and that the problems datapack contains problems for these groups."
            )

        keyed_problems: dict[str, Problem] = {p.task_id: p for p in problems}

        # Verify that all problems are for the current schema version
        if any(p.schema_version != PROBLEM_SCHEMA_VERSION for p in keyed_problems.values()):
            raise ValueError(
                f"One or more problems in {problems_file} do not match the expected schema version {PROBLEM_SCHEMA_VERSION}."
            )

    task_ids = set(p.task_id for p in problems)

    # Collate the solution task_ids by datapack
    solutions_by_datapack: dict[str, set[str]] = defaultdict(set)
    for solution in solutions:
        solutions_by_datapack[solution.datapack_name].add(solution.task_id)

    # Verify that each solutions datapack contains at least one solution for each problem
    for datapack_name, solution_task_ids in solutions_by_datapack.items():
        missing_ids = task_ids - solution_task_ids
        if missing_ids:
            raise ValueError(
                f"The following task_ids are missing in the solutions datapack {datapack_name}: {missing_ids}"
            )

    # Collate the solutions by task_id for evaluation.
    solutions_by_task_id: dict[str, list[Solution]] = defaultdict(list)
    for solution in solutions:
        solutions_by_task_id[solution.task_id].append(solution)

    # Check the generated solutions against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        results: list[GradedSolution] = []

        for task_id, solutions in tqdm.tqdm(solutions_by_task_id.items()):
            problem = keyed_problems.get(task_id)

            future = executor.submit(
                evaluate_solutions,
                problem=problem,
                solutions=solutions,
                eval_mode=mode,
                profile_mode=profile_mode,
            )
            futures.append(future)

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            results.extend(future.result())

    # Now that we've graded all solutions we want to break back out to report on each solution datapack.
    graded_by_datapack: dict[str, list[GradedSolution]] = defaultdict(list)
    for result in results:
        graded_by_datapack[result.solution.datapack_name].append(result)

    for name, results in graded_by_datapack.items():
        pass_at_k = estimate_metrics(results, k_vals)

        # Calculate per-group metrics if groups are specified
        metrics_by_group = None
        if groups:
            metrics_by_group = estimate_metrics_by_group(results, k_vals)

        # Calculate performance metrics if profiling was enabled
        performance_metrics = estimate_performance_metrics(results) if profile_mode else None

        write_metrics(
            results,
            pass_at_k,
            f"{name}-graded-solutions.jsonl",
            performance_metrics,
            groups,
            metrics_by_group,
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
    for _, task_results in results_by_task.items():
        if all(r.skipped for r in task_results):
            skipped += 1
            continue
        total.append(len(task_results))
        correct.append(sum(r.passed for r in task_results))
    total = np.array(total)
    correct = np.array(correct)

    # Count of problems actually evaluated (not skipped)
    evaluated_problem_count = len(total)

    return {
        "skipped": float(skipped),
        "evaluated_problem_count": evaluated_problem_count,
        **{f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in k_vals if (total >= k).all()},
    }


def estimate_metrics_by_group(
    results: list[GradedSolution],
    k_vals: list[int],
) -> dict[str, dict[str, float]]:
    """
    Estimates metrics grouped by problem category.

    Args:
        results: List of graded solutions
        k_vals: List of k values for pass@k calculation

    Returns:
        Dictionary mapping group name to metrics dict
    """
    # Group by problem.group
    results_by_group = defaultdict(list)
    for result in results:
        results_by_group[result.problem.group].append(result)

    group_metrics = {}
    for group, group_results in results_by_group.items():
        # Group results by task_id within this group
        results_by_task = defaultdict(list)
        for result in group_results:
            results_by_task[result.solution.task_id].append(result)

        skipped = 0
        total, correct = [], []

        for _, task_results in results_by_task.items():
            if all(r.skipped for r in task_results):
                skipped += 1
                continue
            total.append(len(task_results))
            correct.append(sum(r.passed for r in task_results))

        total = np.array(total)
        correct = np.array(correct)
        evaluated_problem_count = len(total)

        group_metrics[group] = {
            "skipped": float(skipped),
            "evaluated_problem_count": evaluated_problem_count,
            **{
                f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                for k in k_vals
                if len(total) > 0 and (total >= k).all()
            },
        }

    return group_metrics


def write_metrics(
    results: list[GradedSolution],
    pass_at_k: dict[str, float],
    results_file: str,
    performance_metrics: dict[str, Any] | None = None,
    groups: list[ValidGroup] | None = None,
    metrics_by_group: dict[str, dict[str, float]] | None = None,
) -> None:
    """
    Writes the metrics to a file and prints consolidated output.

    Args:
        results (list[EvaluatedSample]): List of evaluated samples.
        pass_at_k (Dict[str, float]): Pass@k metrics.
        results_file (str): Path to the output results file.
        performance_metrics (dict[str, Any] | None): Performance metrics.
        groups (list[ValidGroup] | None): List of groups evaluated or None.

    Returns:
        None
    """
    # Finally, save the results in one file:
    print(f"Writing results to {results_file}...")
    write_graded_solutions(results_file, results)

    # Total problems attempted (including skipped)
    total_problem_count = len(set(r.solution.task_id for r in results))

    # Extract evaluated_problem_count from pass_at_k metrics (not skipped problems)
    evaluated_problem_count = int(pass_at_k.pop("evaluated_problem_count", total_problem_count))

    # Output structured JSON to stdout
    output: dict[str, Any] = {
        "pass_at_k": {k: float(v) for k, v in pass_at_k.items()},
        "total_problem_count": total_problem_count,
        "evaluated_problem_count": evaluated_problem_count,
    }

    if groups is not None:
        output["groups"] = groups

    if metrics_by_group is not None:
        output["metrics_by_group"] = metrics_by_group

    if performance_metrics:
        output.update(performance_metrics)

    print(json.dumps(output, indent=2))


def estimate_performance_metrics(results: list[GradedSolution]) -> dict[str, dict[str, float | int]]:
    """
    Estimates performance metrics for the given solutions.

    Args:
        results (list[GradedSolution]): List of graded solutions.

    Returns:
        dict[str, dict[str, float | int]]: A dictionary containing performance metrics.
    """

    def get_valid_values(field_getter) -> list[float]:
        """Extract valid (passed, non-null, non-NaN) values from results."""
        # noinspection PyUnboundLocalVariable
        return [value for r in results if r.passed and (value := field_getter(r)) is not None and not math.isnan(value)]

    # Collect valid metrics
    valid_solution_times = get_valid_values(lambda r: r.solution_time)
    valid_sm_throughputs = get_valid_values(lambda r: r.sm_throughput)
    valid_dram_throughputs = get_valid_values(lambda r: r.dram_throughput)
    valid_speedups = get_valid_values(lambda r: r.speedup)

    # Count invalid/skipped
    invalid_count = sum(1 for r in results if not r.passed or r.solution_time is None)

    metrics: dict[str, Any] = {
        "performance_analysis": {
            "invalid_skipped": invalid_count,
            "avg_solution_time_ms": (
                sum(valid_solution_times) / len(valid_solution_times) if valid_solution_times else 0.0
            ),
            "avg_sm_throughput_pct": (
                sum(valid_sm_throughputs) / len(valid_sm_throughputs) if valid_sm_throughputs else 0.0
            ),
            "avg_dram_throughput_pct": (
                sum(valid_dram_throughputs) / len(valid_dram_throughputs) if valid_dram_throughputs else 0.0
            ),
        }
    }

    # Add baseline comparison metrics if speedups are available
    if valid_speedups:
        metrics["baseline_comparison"] = {
            "baseline_available_count": len(valid_speedups),
            "baseline_coverage_pct": (len(valid_speedups) / len(results) * 100) if results else 0.0,
            "avg_speedup": geometric_mean(valid_speedups),
            "median_speedup": float(np.median(valid_speedups)),
            "min_speedup": float(min(valid_speedups)),
            "max_speedup": float(max(valid_speedups)),
            "regressions_count": sum(1 for s in valid_speedups if s < 1.0),
            "improvements_count": sum(1 for s in valid_speedups if s > 1.0),
            "speedup_p25": float(np.percentile(valid_speedups, 25)),
            "speedup_p75": float(np.percentile(valid_speedups, 75)),
        }

    return metrics
