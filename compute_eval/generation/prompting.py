# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

from compute_eval.data.data_model import FileSolution, Problem, SourceFile
from compute_eval.data.loaders import load_problem_tree

PROMPT_FORMAT_VERSION = "compute-eval-full-file-v1"
SOLUTION_END = "<|END_SOLUTION|>"
DEFAULT_STOP_SEQUENCES = (SOLUTION_END, "\n## Example", "\n## Task")
_SUPPORTED_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".py")

_ZERO_SHOT_SYSTEM_PROMPT = (
    "You are a senior CUDA engineer. Write the complete requested source file. "
    "Preserve interfaces and return only that file."
)
_FEW_SHOT_SYSTEM_PROMPT = (
    "You are a senior CUDA engineer. Study the completed examples, then write the complete requested source "
    "file for the final task. Preserve interfaces and return only that file."
)

__all__ = [
    "CompletionPromptConfig",
    "CompletionTask",
    "build_completion_task",
    "load_demonstrations",
    "packaged_demonstrations_dir",
]


@dataclass(frozen=True)
class CompletionPromptConfig:
    """Configure a transport-neutral full-file completion prompt."""

    demonstrations: tuple[Problem, ...] = ()
    system_prompt: str | None = None
    stop_sequences: tuple[str, ...] = DEFAULT_STOP_SEQUENCES


@dataclass(frozen=True)
class CompletionTask:
    """Exact prompt and parser contract for one model completion."""

    task_id: str
    target_path: str
    system_prompt: str
    user_prompt: str
    demonstration_task_ids: tuple[str, ...]
    stop_sequences: tuple[str, ...]
    prompt_format_version: str = PROMPT_FORMAT_VERSION

    @property
    def text_prompt(self) -> str:
        return _to_text_prompt(self.system_prompt, self.user_prompt)

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.text_prompt.encode()).hexdigest()

    def build_solution(
        self,
        completion: str,
    ) -> FileSolution:
        generated_file = parse_generated_file(completion, self.target_path)
        files = [generated_file] if generated_file is not None else []
        return FileSolution(task_id=self.task_id, files=files)

    def to_manifest(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "target_path": self.target_path,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "text_prompt": self.text_prompt,
            "prompt_sha256": self.prompt_sha256,
            "prompt_format_version": self.prompt_format_version,
            "demonstration_task_ids": list(self.demonstration_task_ids),
            "stop_sequences": list(self.stop_sequences),
        }


@cache
def _load_demonstrations(directory: Path) -> tuple[Problem, ...]:
    demonstrations = [
        problem.model_copy(update={"baseline_solution": solution}) for problem, solution in load_problem_tree(directory)
    ]
    return tuple(sorted(demonstrations, key=lambda problem: problem.task_id))


def load_demonstrations(directory: str | Path, num_shots: int) -> tuple[Problem, ...]:
    """Load a deterministic prefix of demonstrations from a directory."""
    if num_shots < 0:
        raise ValueError("num_shots must be non-negative")
    available = _load_demonstrations(Path(directory).expanduser().resolve())
    if len(available) < num_shots:
        raise ValueError(f"Requested {num_shots} demonstrations, but found {len(available)}")
    return available[:num_shots]


def packaged_demonstrations_dir() -> Path:
    """Return the installed filesystem path for the reviewed demonstrations."""
    packaged = resources.files("compute_eval.generation").joinpath("resources/few_shot_examples")
    return Path(packaged)


def build_completion_task(
    problem: Problem,
    config: CompletionPromptConfig,
) -> CompletionTask:
    """Build one full-file completion task from a problem and prompt configuration."""
    demonstrations = config.demonstrations

    source = target_file(problem)
    for demonstration in demonstrations:
        target_file(demonstration)
    if problem.task_id in {demonstration.task_id for demonstration in demonstrations}:
        raise ValueError(f"Demonstrations overlap scored target: {problem.task_id}")

    system_prompt = config.system_prompt
    if system_prompt is None:
        system_prompt = default_system_prompt(bool(demonstrations))
    return CompletionTask(
        task_id=problem.task_id,
        target_path=source.path,
        system_prompt=system_prompt,
        user_prompt=_build_user_prompt(problem, demonstrations),
        demonstration_task_ids=tuple(demonstration.task_id for demonstration in demonstrations),
        stop_sequences=config.stop_sequences,
    )


def default_system_prompt(has_demonstrations: bool) -> str:
    return _FEW_SHOT_SYSTEM_PROMPT if has_demonstrations else _ZERO_SHOT_SYSTEM_PROMPT


def target_file(problem: Problem) -> SourceFile:
    solution = problem.baseline_solution
    if not isinstance(solution, FileSolution) or len(solution.files) != 1:
        raise ValueError(f"Problem {problem.task_id} needs one solution file")
    source = solution.files[0]
    if not source.path.lower().endswith(_SUPPORTED_SUFFIXES):
        raise ValueError(f"Problem {problem.task_id} has an unsupported solution file: {source.path}")
    return source


def _fence_for_path(path: str) -> str:
    suffix = path.lower()
    if suffix.endswith((".cu", ".cuh")):
        return "cuda"
    if suffix.endswith((".cc", ".cpp", ".cxx")):
        return "cpp"
    if suffix.endswith(".c"):
        return "c"
    if suffix.endswith((".h", ".hpp")):
        return "h"
    if suffix.endswith(".py"):
        return "python"
    return ""


def _comment_prefix(path: str) -> str:
    return "#" if path.lower().endswith(".py") else "//"


def _format_file(source: SourceFile) -> str:
    return (
        f"--- file: {source.path}\n"
        f"```{_fence_for_path(source.path)}\n"
        f"{_comment_prefix(source.path)} file: {source.path}\n"
        f"{source.content}\n"
        "```\n"
    )


def _format_problem(problem: Problem) -> str:
    source = target_file(problem)
    context = "".join(_format_file(context_file) for context_file in problem.context_files)
    marker = "# COMPUTE_EVAL_COMPLETION\n" if source.path.lower().endswith(".py") else "/* COMPUTE_EVAL_COMPLETION */\n"
    scaffold = SourceFile(path=source.path, content=marker)
    return (
        f"Problem:\n{problem.prompt}\n\n"
        f"Build command:\n{problem.build_command}\n\n"
        f"Context files:\n{context or '(none)'}\n"
        f"Solution scaffold:\n{_format_file(scaffold)}"
    )


def _build_user_prompt(target: Problem, demonstrations: Sequence[Problem]) -> str:
    examples = ""
    for index, demonstration in enumerate(demonstrations, start=1):
        examples += (
            f"\n## Example {index}\n\n"
            f"{_format_problem(demonstration)}\n"
            f"Completed solution:\n{_format_file(target_file(demonstration))}"
            f"{SOLUTION_END}\n"
        )

    source = target_file(target)
    path_comment = f"{_comment_prefix(source.path)} file: {source.path}"
    fence = _fence_for_path(source.path)
    return (
        f"Write the entire target file in place of the marker. Begin the response immediately with one `{fence}` "
        f"fenced code block containing the full completed target file. Put the path comment `{path_comment}` on "
        f"the first line inside the fence. After the closing fence, output `{SOLUTION_END}` and stop. Return no "
        "explanation, tests, additional files, or benchmark records.\n"
        f"{examples}\n"
        "## Task\n\n"
        f"{_format_problem(target)}\n"
        "Completed solution:\n"
    )


def _to_text_prompt(system_prompt: str, user_prompt: str) -> str:
    return f"{system_prompt.rstrip()}\n\n{user_prompt}" if system_prompt else user_prompt


_CODE_BLOCK_RE = re.compile(r"```[^\n`]*\n(?P<body>.*?)(?:```|\Z)", re.DOTALL)
_PATH_RE = re.compile(
    r"^\s*(?://|#|;|/\*)\s*(?:file:\s*)?(?P<path>[A-Za-z0-9._/\-]+)\s*(?:\*/)?\s*$",
    re.IGNORECASE,
)
_EXTERNAL_PATH_RE = re.compile(r"(?:^|\n)---\s*file:\s*(?P<path>[A-Za-z0-9._/\-]+)\s*\Z")


def parse_generated_file(completion: str, target_path: str) -> SourceFile | None:
    text = completion.replace("\r\n", "\n").replace("\r", "\n")
    blocks = list(_CODE_BLOCK_RE.finditer(text))
    for block in blocks:
        body = block.group("body")
        lines = body.lstrip("\n").split("\n")
        path = _path_from_line(lines[0].lstrip("﻿").strip()) if lines else None
        if path == target_path:
            return SourceFile(path=target_path, content="\n".join(lines[1:]))
        if path is not None:
            continue
        if len(blocks) == 1:
            external_path = _EXTERNAL_PATH_RE.search(text[: block.start()].rstrip())
            if external_path is None or external_path.group("path") == target_path:
                return SourceFile(path=target_path, content=body)
    return None


def _path_from_line(line: str) -> str | None:
    match = _PATH_RE.match(line)
    return match.group("path") if match else None
