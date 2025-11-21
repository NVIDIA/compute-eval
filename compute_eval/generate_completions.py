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
import gzip
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm

from compute_eval.data.data_model import FileSolution, Problem, ReleaseVersion, Solution, SourceFile
from compute_eval.data.utils import read_solutions, write_solutions

from . import get_model_class
from .data.data_pack import ProblemDatapack, SolutionDatapack
from .models.model_interface import ModelInterface
from .prompts import to_user_message


def generate_model_completions(
    system_prompt: str,
    problem: Problem,
    model: str,
    base_url: str | None = None,
    reasoning: str | None = None,
    params: dict | None = None,
    debug: bool = False,
) -> Solution:
    """
    Orchestrate the generation of code completions using the specified model.

    Args:
        system_prompt (str): The system prompt to use for generating completions.
        problem (Problem): The problem object containing the task details.
        model (str): The name of the model to use for generating completions.
        base_url (str, optional): The base URL for the custom model API endpoint.
        reasoning (str, optional): Reasoning mode for the model (e.g., 'low', 'medium', 'high' for GPT models, or any value for Claude models to enable extended thinking).
        params (dict, optional): Additional parameters to pass to the model invocation.
        debug (bool, optional): Whether to include the system prompt, prompt, and generated completion in the output solution for debugging.

    Returns:
        solution (Solution): The generated solution object containing the completions.
    """

    model_class = get_model_class(model)
    model_instance: ModelInterface = model_class(
        model_name=model,
        base_url=base_url,
        reasoning=reasoning,
    )

    if params is None:
        params = {}

    prompt = to_user_message(problem)

    completion = model_instance.generate_response(system_prompt, prompt, params)

    debug_info = (
        {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "generated_completion": completion,
        }
        if debug
        else {}
    )

    return FileSolution(
        task_id=problem.task_id,
        files=_parse_solution(completion),
        **debug_info,
    )


def generate_samples(
    release: ReleaseVersion,
    problems_datapack_dir: str,
    solutions_per_problem: int,
    n_workers: int,
    system_prompt: str,
    model: str,
    base_url: str | None,
    reasoning: str | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    temp_dir: str | None,
    debug: bool,
):
    """
    Generates code completions for a set of problems using a specified model and writes them to a solutions datapack.
    Args:
        release (ReleaseVersion): The release version to generate solutions for.
        problems_datapack_dir (str): Directory where released problem datapacks are stored.
        solutions_per_problem (int): Number of solutions to generate per problem.
        n_workers (int): Number of worker threads to use for parallel generation.
        system_prompt (str): The system prompt to use for generating completions.
        model (str): The name of the model to use for generating completions.
        base_url (str | None): Base URL for the custom model API endpoint.
        reasoning (str | None): Reasoning mode for the model (e.g., 'low', 'medium', 'high' for GPT models, or any value for Claude models to enable extended thinking).
        temperature (float | None): Temperature for generation.
        top_p (float | None): Top-p for generation.
        max_tokens (int | None): Maximum tokens for generation.
        temp_dir (str | None): Temporary directory to store intermediate results.
        debug (bool): Whether to include the system prompt, prompt, and generated completion in the output solution for debugging.
    """

    def _task_id_to_filename(directory: str, _id: str) -> str:
        return f"{directory}/{_id.replace('/', '_')}.jsonl"

    if temp_dir is None:
        temp_dir = model if model else "temp_results"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    problem_file = os.path.join(problems_datapack_dir, f"{release.value}-problems.tar.gz")
    with ProblemDatapack(problem_file) as datapack:
        if datapack.metadata.release != release:
            raise ValueError(
                f"Problems datapack release {datapack.metadata.release} does not match expected release {release}."
            )
        problems = list(datapack.read_items())

    task_count = {p.task_id: _count_lines(_task_id_to_filename(temp_dir, p.task_id)) for p in problems}

    print("Started generating the model completions")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for problem in problems:
            existing_sample_count = task_count.get(problem.task_id, 0)
            solutions_to_generate = solutions_per_problem - existing_sample_count

            if solutions_to_generate <= 0:
                print(f"Skipping {problem.task_id}, already have {existing_sample_count} solutions")
                continue

            for _ in range(solutions_to_generate):
                params = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                args = {
                    "system_prompt": system_prompt,
                    "problem": problem,
                    "model": model,
                    "base_url": base_url,
                    "reasoning": reasoning,
                    "params": params,
                    "debug": debug,
                }
                future = executor.submit(generate_model_completions, **args)
                futures.append(future)

        print("Waiting for all the model completions")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                solution = future.result()
                write_solutions(
                    file_path=_task_id_to_filename(temp_dir, solution.task_id),
                    solutions=[solution],
                    append=True,
                )
            except Exception as e:
                print(f"Error processing future: {e}")

    all_results = []
    for task_file in sorted(os.listdir(temp_dir)):
        task_file_path = os.path.join(temp_dir, task_file)
        all_results.extend(read_solutions(task_file_path))

    if len(all_results) != len(problems) * solutions_per_problem:
        print(f"Error: Expected {len(problems) * solutions_per_problem} samples, but got {len(all_results)}")
        raise ValueError("Sample generation incomplete")

    model = model.replace("/", "-") if model else None
    SolutionDatapack.create(
        file_path=f"{release.value}-{model}-solutions.tar.gz" if model else f"{release.value}-solutions.tar.gz",
        items=all_results,
        release=release,
    )

    # Clean up temporary files
    for task_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, task_file))
    os.rmdir(temp_dir)

    print("Completed generating all the samples for the problems. Written to the samples JSONL file")


def _count_lines(filename: str) -> int:
    """
    Counts the number of lines in a file
    """
    if not os.path.exists(filename):
        return 0

    count = 0
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp, gzip.open(gzfp, "rt") as fp:
            for _ in fp:
                count += 1
    else:
        with open(filename, "r") as fp:
            for _ in fp:
                count += 1
    return count


_CODE_BLOCK_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL | re.MULTILINE)
_FIRST_LINE_PATH_RE = re.compile(r"^(?://|#|;)\s*file:\s*([A-Za-z0-9._/\-]+)\s*$", re.IGNORECASE)
_FIRST_LINE_BLOCK_COMMENT_PATH_RE = re.compile(r"^\s*/\*\s*file:\s*(.+?)\s*\*/\s*$", re.IGNORECASE)


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _guess_ext_from_lang(lang: str) -> str:
    lang = (lang or "").strip().lower()
    if lang == "cuda":
        return ".cu"
    if lang in ("cpp", "c++"):
        return ".cc"
    if lang == "c":
        return ".c"
    if lang in ("h", "hpp", "header"):
        return ".h"
    return ".txt"


def _parse_solution(response: str) -> list[SourceFile]:
    if not isinstance(response, str):
        raise TypeError("response must be a string")

    text = _normalize_newlines(response)
    matches = list(_CODE_BLOCK_RE.finditer(text))

    # If no fenced code blocks found, treat entire response as raw code
    if not matches:
        source_file = _process_code_block(text)
        return [source_file] if source_file else []

    # Process each fenced code block
    files: list[SourceFile] = []
    for m in matches:
        block = _normalize_newlines(m.group(2))
        source_file = _process_code_block(block)
        if source_file:
            files.append(source_file)

    return files


def _process_code_block(block: str) -> SourceFile | None:
    """Process a single code block and extract path + content."""
    block_stripped = block.lstrip("\n")
    lines = block_stripped.split("\n")
    if not lines:
        return None

    first_line = lines[0].strip("\ufeff").strip()
    path = _extract_path_from_line(first_line)

    if not path:
        return None

    return SourceFile(
        path="solution.cu",
        content="\n".join(lines[1:]),
    )


def _extract_path_from_line(line: str) -> str | None:
    """Extract file path from a line using various comment formats."""
    # Try regular path format first
    m1 = _FIRST_LINE_PATH_RE.match(line)
    if m1:
        return m1.group(1).strip()

    # Try block comment format
    m2 = _FIRST_LINE_BLOCK_COMMENT_PATH_RE.match(line)
    if m2:
        return m2.group(1).strip()

    return None
