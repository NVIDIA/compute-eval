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

from compute_eval.models.claude import ClaudeModel
from compute_eval.models.nim_model import NimModel
from compute_eval.models.openAI_model import OpenAIModel

from .data import Problem, read_problems, stream_jsonl, write_jsonl
from .prompts import SYSTEM_PROMPT, generate_user_prompt


def generate_model_completions(
    system_prompt: str,
    problem: Problem,
    print_completions: bool,
    include_header_files: bool,
    model: str | None,
    model_type: str | None,
    custom_model: dict | None = None,
    params: dict | None = None,
):
    """
    Orchestrate the generation of code completions using the specified model.

    Args:
        system_prompt (str): The system prompt to use for generating completions.
        problem (Problem): The problem object containing the task details.
        model (str): The name of the model to use for generating completions.
        model_type (str): The type of the model ("instruct" or "base").
        print_completions (bool): Whether to print the completions.
        include_header_files (bool): Whether to include header files in the prompt.
        custom_model (dict, optional): Custom model object to use for generating completions.
        params (dict, optional): Additional parameters to pass to the model.

    Returns:
        str: runnable code completion, including declaration, completion, and test code.
    """

    # Means we are invoking a model from the preset list of models

    if custom_model is not None:
        model_instance = OpenAIModel(base_url=custom_model["api_endpoint"], model_name=custom_model["model_id"])
    else:
        model_map = {
            "mixtral-8x22b-v0.1": lambda: NimModel(
                "mistralai/mixtral-8x22b-instruct-v0.1"
            ),
            "gemma-2-2b-it": lambda: NimModel("google/gemma-2-2b-it"),
            "llama-3.1-8b-instruct": lambda: NimModel("meta/llama-3.1-8b-instruct"),
            "llama-3.1-70b-instruct": lambda: NimModel("meta/llama-3.1-70b-instruct"),
            "llama-3.1-405b-instruct": lambda: NimModel("meta/llama-3.1-405b-instruct"),
            "llama-3.2-1b-instruct": lambda: NimModel("meta/llama-3.2-1b-instruct"),
            "llama-3.2-3b-instruct": lambda: NimModel("meta/llama-3.2-3b-instruct"),
            "llama-3.1-nemotron-70b-instruct": lambda: NimModel(
                "nvidia/llama-3.1-nemotron-70b-instruct"
            ),
            "nemotron-mini-4b-instruct": lambda: NimModel(
                "nvidia/nemotron-mini-4b-instruct"
            ),
            "starcoder2-7b": lambda: NimModel("bigcode/starcoder2-7b"),
            "mistral-nemo-12b-instruct": lambda: NimModel(
                "nv-mistralai/mistral-nemo-12b-instruct"
            ),
            "claude-sonnet-3.5": lambda: ClaudeModel("claude-3-5-sonnet-20241022"),
        }

        assert model in model_map, f"Unsupported model: {model}"

        model_instance_factory = model_map.get(model)
        if model_instance_factory is None:
            raise ValueError(f"Unsupported model: {model}")

        model_instance = model_instance_factory()

    prompt = generate_user_prompt(problem, include_header_files=include_header_files)
    completion = model_instance.generate_response(system_prompt, prompt, params)

    if print_completions:
        if hasattr(problem, "cuda_toolkit"):
            print(f"CUDA toolkit: {problem.cuda_toolkit}")
        print("=" * 30)

        print(problem.task_id + "\n")
        print(f"=== Prompt ===\n{prompt}\n")

    if model_type == "instruct":
        # we need to parse the completion to get the code
        # first, check whether the declaration provides the function signature
        drop_signature = False
        # TODO: TBD on improving this logic
        declaration = problem.declaration
        if declaration.strip().endswith("{"):
            drop_signature = True

        function_body = parse_function_body(completion, drop_signature=drop_signature)
    else:
        function_body = completion

    if print_completions:
        print(f"=== Completion ===\n{completion}\n")

    result = problem.declaration + "\n\n"
    result = result + "// completion-begin \n"
    result = result + function_body + "\n"
    result = result + "// completion-end \n\n"
    result = result + problem.test

    return problem.task_id, result, completion, prompt


def parse_function_body(input_string, drop_signature: bool = True):
    """
    Extract function body from the response of the model.

    Args:
        input_string (str): The response string from the model.
        drop_signature (bool): Whether to remove the function signature. Default is True.

    Returns:
        str: The extracted function body, or the original string if no code fences are found.
    """

    # Regular expression to find code fences and extract the code between them
    fence_pattern = re.compile(r"```\s*[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)
    fence_matches = fence_pattern.findall(input_string)

    if len(fence_matches) == 0:
        return input_string.strip()  # Return the original input if no code fences are found

    # Use the code block from the first code fence found
    code_block = fence_matches[0].strip()

    if not drop_signature:
        return code_block

    # Regular expression to match the function body
    body_pattern = re.compile(r"{(.*)}", re.DOTALL)
    body_match = body_pattern.search(code_block)

    if body_match:
        function_body = body_match.group(1)
        return function_body + "\n}"
    else:
        return code_block


def generate_samples(
    problem_file: str,
    sample_file: str = "generated_samples.jsonl",
    num_samples_per_problem: int = 100,
    n_workers: int = 20,
    system_prompt: str | None = SYSTEM_PROMPT,
    print_completions: bool = False,
    include_header_files: bool = False,
    model: str | None = "llama3.1-70b",
    model_type: str | None = "instruct",
    custom_model: dict | None = None,
    params: dict | None = None,
    temp_dir: str | None = None,
):
    """Generates `n_samples_per_problem` number of completions for each of the problems in the
    problem file and then writes them out to the samples.jsonl file provided.
    """

    def _task_id_to_filename(directory: str, task_id: str) -> str:
        return f"{directory}/{task_id.replace('/', '_')}.jsonl"

    if temp_dir is None:
        temp_dir = model if model else "temp_results"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    problems = read_problems(problem_file)
    task_count = {p.task_id: _count_lines(_task_id_to_filename(temp_dir, p.task_id)) for p in problems}

    print("Started generating the model completions")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for problem in problems:
            existing_sample_count = task_count.get(problem.task_id, 0)
            samples_to_generate = num_samples_per_problem - existing_sample_count

            if samples_to_generate <= 0:
                print(f"Skipping {problem.task_id}, already have {existing_sample_count} samples")
                continue

            for _ in range(samples_to_generate):
                args = (
                    system_prompt,
                    problem,
                    print_completions,
                    include_header_files,
                    model,
                    model_type,
                    custom_model,
                    params,
                )
                future = executor.submit(generate_model_completions, *args)
                futures.append(future)

        print("Waiting for all the model completions")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                task_id = result[0]
                result_data = {
                    "task_id": result[0],
                    "compilable_code": result[1],
                    "generated_completion": result[2],
                    "prompt": result[3],
                }
                write_jsonl(
                    filename=_task_id_to_filename(temp_dir, task_id),
                    data=[result_data],
                    append=True,
                )
            except Exception as e:
                print(f"Error processing future: {e}")

    all_results = []
    for task_file in sorted(os.listdir(temp_dir)):
        task_file_path = os.path.join(temp_dir, task_file)
        all_results.extend(stream_jsonl(task_file_path))

    if len(all_results) != len(problems) * num_samples_per_problem:
        print(f"Error: Expected {len(problems) * num_samples_per_problem} samples, but got {len(all_results)}")
        raise ValueError("Sample generation incomplete")

    write_jsonl(sample_file, all_results)

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
