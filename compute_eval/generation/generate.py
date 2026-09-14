# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from random import Random
from typing import Literal, Self, get_args

import tqdm
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from compute_eval.data.data_model import FileSolution, Problem, ReleaseVersion, ValidGroup
from compute_eval.data.data_pack import ProblemDatapack, SolutionDatapack, open_problem_datapack
from compute_eval.data.utils import read_solutions, write_solutions
from compute_eval.generation.prompting import (
    DEFAULT_STOP_SEQUENCES,
    CompletionPromptConfig,
    build_completion_task,
    load_demonstrations,
    target_file,
)

from . import ModelInterface, get_model_class

VALID_GROUP_CHOICES = list(get_args(ValidGroup))


class CompletionConfig(BaseModel):
    """Configure a completion run."""

    model_config = ConfigDict(extra="forbid")

    release: ReleaseVersion = ReleaseVersion.V2026_2
    include: list[ValidGroup] | None = Field(
        default=None,
        description=f"Groups to include. Valid values: {VALID_GROUP_CHOICES}",
    )
    exclude: list[ValidGroup] | None = Field(
        default=None,
        description=f"Groups to exclude. Valid values: {VALID_GROUP_CHOICES}",
    )
    problems_datapack_dir: str = "data/releases/"
    problems_source: str | None = Field(
        default=None,
        description=(
            "Where to load problems from. None loads the conventional "
            "'{release}-problems.tar.gz' under problems_datapack_dir if it exists, "
            "otherwise falls back to HuggingFace (nvidia/compute-eval). A .tar.gz path "
            "opens that datapack directly; any other path is treated as a directory of "
            "YAML problem specs."
        ),
    )
    demonstrations_dir: str | None = None
    solutions_per_problem: int = Field(default=1, gt=0)
    n_workers: int = Field(default=10, gt=0)
    system_prompt: str | None = None
    model: str = Field(description="Model to use for generation")
    base_url: str | None = None
    reasoning: str | None = None
    thinking: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    temp_dir: str | None = None
    output_dir: str | None = None
    api_mode: Literal["chat", "completion"] = "chat"
    debug: bool = False
    num_shots: int = Field(default=0, ge=0)
    seed: int = 0
    limit: int | None = Field(default=None, gt=0)
    stop: list[str] | None = None

    @field_validator("include", "exclude", mode="before")
    @classmethod
    def coerce_to_list(cls, value):
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        raise ValueError(f"Expected a comma-separated string or list, got {type(value).__name__}")

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if self.include is not None and self.exclude is not None:
            raise ValueError("include and exclude are mutually exclusive")
        if self.reasoning is not None and self.thinking:
            raise ValueError("reasoning and thinking are mutually exclusive")
        if self.api_mode == "completion" and (self.reasoning is not None or self.thinking):
            raise ValueError("Raw completion mode does not support reasoning or thinking")
        if self.num_shots and self.demonstrations_dir is None:
            raise ValueError("demonstrations_dir is required when num_shots is positive")
        return self


def _complete_problem(
    problem: Problem,
    config: CompletionConfig,
    prompt_config: CompletionPromptConfig,
    request_options: dict,
) -> FileSolution:
    model_class = get_model_class(config.model)
    model_instance: ModelInterface = model_class(model_name=config.model, base_url=config.base_url)
    task = build_completion_task(problem, prompt_config)
    if config.api_mode == "completion":
        request_prompt = task.text_prompt
        completion = model_instance.generate_text_completion(request_prompt, request_options)
    else:
        request_prompt = task.user_prompt
        completion = model_instance.generate_response(task.system_prompt, task.user_prompt, request_options)

    debug_info = (
        {
            "system_prompt": task.system_prompt,
            "prompt": request_prompt,
            "generated_completion": completion,
            "demonstration_task_ids": list(task.demonstration_task_ids),
        }
        if config.debug
        else {}
    )
    solution = task.build_solution(completion)
    if not debug_info:
        return solution
    return FileSolution(task_id=solution.task_id, files=solution.files, **debug_info)


def _resolve_problems_source(config: CompletionConfig) -> str | None:
    """Pick the problems source for open_problem_datapack.

    An explicit ``problems_source`` wins. Otherwise fall back to the
    conventional ``{release}-problems.tar.gz`` under ``problems_datapack_dir``
    when it exists; if it does not, return None so the datapack loads from
    HuggingFace.
    """
    if config.problems_source is not None:
        return config.problems_source
    conventional = Path(config.problems_datapack_dir) / f"{config.release.value}-problems.tar.gz"
    return str(conventional) if conventional.exists() else None


def generate_samples(config: CompletionConfig) -> tuple[Path, Path]:
    source = _resolve_problems_source(config)
    with open_problem_datapack(config.release, source, include=config.include, exclude=config.exclude) as datapack:
        if datapack.metadata.release != config.release:
            raise ValueError(
                f"Problems datapack release {datapack.metadata.release} does not match expected release {config.release}"
            )
        problems: list[Problem] = list(datapack.read_items())
        # datapack.groups is a set; sort for a deterministic solutions manifest.
        groups = sorted(datapack.groups)

    demonstrations = (
        load_demonstrations(config.demonstrations_dir, config.num_shots)
        if config.demonstrations_dir is not None
        else ()
    )
    prompt_config = CompletionPromptConfig(
        demonstrations=demonstrations,
        system_prompt=config.system_prompt,
        stop_sequences=tuple(config.stop) if config.stop is not None else DEFAULT_STOP_SEQUENCES,
    )
    targets = _select_targets(
        problems,
        list(demonstrations),
        seed=config.seed,
        limit=config.limit,
    )
    print(f"Using demonstration tasks: {[problem.task_id for problem in demonstrations]}")
    safe_model = config.model.replace("/", "-")
    temp_path = Path(config.temp_dir or f"{safe_model}-{config.num_shots}shot-temp")
    temp_path.mkdir(parents=True, exist_ok=True)
    result_paths = {problem.task_id: _result_path(temp_path, problem.task_id) for problem in targets}
    task_counts = {task_id: _count_lines(path) for task_id, path in result_paths.items()}
    request_options = {
        "reasoning": config.reasoning,
        "thinking": config.thinking,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stop": list(prompt_config.stop_sequences),
    }

    futures = []
    with ThreadPoolExecutor(max_workers=config.n_workers) as executor:
        for problem in targets:
            remaining = config.solutions_per_problem - task_counts[problem.task_id]
            for _ in range(max(remaining, 0)):
                futures.append(
                    executor.submit(
                        _complete_problem,
                        problem,
                        config,
                        prompt_config,
                        request_options,
                    )
                )

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Generating solutions"):
            solution = future.result()
            write_solutions(result_paths[solution.task_id], [solution], append=True)

    all_results = [solution for path in result_paths.values() for solution in read_solutions(path)]
    expected = len(targets) * config.solutions_per_problem
    if len(all_results) != expected:
        raise ValueError(f"Sample generation incomplete: expected {expected}, found {len(all_results)}")

    output_path = Path(config.output_dir or ".")
    output_path.mkdir(parents=True, exist_ok=True)
    solutions_file = output_path / f"{config.release.value}-{safe_model}-{config.num_shots}shot-solutions.tar.gz"
    problems_file = output_path / f"{config.release.value}-seed{config.seed}-n{len(targets)}-problems.tar.gz"
    SolutionDatapack.create(solutions_file, all_results, config.release, groups=groups)
    ProblemDatapack.create(
        problems_file,
        targets,
        config.release,
        description="Scored target subset for full-solution generation",
    )

    for path in result_paths.values():
        path.unlink()
    try:
        temp_path.rmdir()
    except OSError:
        print(f"Kept non-empty temporary directory: {temp_path}")

    print(f"Wrote solutions to {solutions_file}")
    print(f"Wrote scored problem subset to {problems_file}")
    return solutions_file, problems_file


def _select_targets(
    problems: list[Problem],
    demonstrations: list[Problem],
    seed: int,
    limit: int | None,
) -> list[Problem]:
    for problem in [*problems, *demonstrations]:
        target_file(problem)

    targets = sorted(problems, key=lambda problem: problem.task_id)
    Random(seed).shuffle(targets)

    overlapping_ids = {problem.task_id for problem in targets} & {problem.task_id for problem in demonstrations}
    if overlapping_ids:
        raise ValueError(f"Demonstrations overlap scored targets: {sorted(overlapping_ids)}")
    return targets[:limit]


def _result_path(directory: Path, task_id: str) -> Path:
    return directory / f"{task_id.replace('/', '_')}.jsonl"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as result_file:
        return sum(1 for _ in result_file)
