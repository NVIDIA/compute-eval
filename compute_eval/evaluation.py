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
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Union

import numpy as np
import tqdm
from tabulate import tabulate

from compute_eval.data import (
    read_problems,
    stream_jsonl,
    write_completions_to_dir,
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
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

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

    return np.array([
        estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)
    ])


def get_cli_args(problem: Dict[str, Dict]):
    cc_flags = problem.get("cc_flags")
    ld_flags = problem.get("ld_flags")

    cli_args = ""
    if cc_flags is not None:
        cli_args += " " + cc_flags
    if ld_flags is not None:
        cli_args += " " + ld_flags

    return cli_args


def evaluate_functional_correctness(
    sample_file: str,
    problem_file: str,
    allow_execution: bool = False,
    k: Tuple[int] = (1, 10, 100),
    n_workers: int = 4,
    timeout: float = 60.0,
    save_completions_dir: str = "",
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_correctness_results.jsonl".
    """

    if not allow_execution:
        print(WARNING_MSG)
        sys.exit(1)

    # Check if only one k value was passed in (as an integer)
    if isinstance(k, int):
        k_vals = [k]
    else:
        # Multiple k values (tuple) is converted to a list of int
        k_vals = list(k)

    # If the user wants to save completions, check that the directory exists
    if save_completions_dir != "":
        assert os.path.exists(os.path.abspath(save_completions_dir)), (
            "You must have created the directory where the temporary completions will go"
        )

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            if "compilable_code" in sample:
                compilable_code = sample["compilable_code"]
            else:
                assert "completion" in sample, (
                    "Each sample must have either 'compilable_code' or 'completion' field."
                )
                completion = sample["completion"]
                compilable_code = problems[task_id]["declaration"] + "\n\n"
                compilable_code += "// completion-begin \n"
                compilable_code += completion + "\n"
                compilable_code += "// completion-end \n\n"
                compilable_code += problems[task_id]["test"]

            problem = problems[task_id]

            cli_args = get_cli_args(problem)

            cuda_version = problem.get("cuda_version")

            args = (
                problem,
                compilable_code,
                timeout,
                completion_id[task_id],
                cli_args,
                cuda_version,
            )
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        # make sure that solved all the problems (at least once)
        assert len(completion_id) == len(problems), "Some problems are not attempted."

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result if not r[1]["skipped"]]

        # If all test cases are skipped, we skip the problem.
        if len(passed) == 0:
            print(
                f"Skipping problem {result[0][1]['task_id']}, it would be ignored while calculating pass@k. Possible reasons maybe incompatible GPU architecture."
            )
            continue
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in k_vals
        if (total >= k).all()
    }

    # Finally, save the results in one file:
    sample_results = []
    for sample in stream_jsonl(sample_file):
        task_id = sample["task_id"]
        result = results[task_id].pop(0)
        sample["result"] = result[1]["result"]
        sample["skipped"] = result[1]["skipped"]
        sample["passed"] = result[1]["passed"]
        sample["completion_id"] = result[1]["completion_id"]
        sample_results.append(sample)

    out_file = (
        os.path.splitext(os.path.basename(sample_file))[0]
        + "_correctness_results.jsonl"
    )
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, sample_results)

    if save_completions_dir != "":
        print(f"Saving the completions to {os.path.abspath(save_completions_dir)}...")
        write_completions_to_dir(save_completions_dir, sample_results)
    print(pass_at_k)
