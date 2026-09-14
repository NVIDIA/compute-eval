import os
from collections.abc import Generator
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, TypeAdapter

from compute_eval.data.data_model import (
    CudaCppProblem,
    CudaPythonProblem,
    FileSolution,
    Metadata,
    PatchSolution,
    Problem,
    Solution,
    SourceFile,
    SourceReferences,
    ValidGroup,
)
from compute_eval.data.metrics_data_model import ProcessTimingMode, TimingModes


class ProblemSpecBase(BaseModel):
    task_id: str
    date: str
    prompt: str
    group: ValidGroup
    build_command: str | None = None
    test_command: str | None = None
    benchmark_command: str | None = None
    base_image: str | None = None
    timeout_seconds: float | None = None
    source_references: str | list[str] | SourceReferences | None = None
    min_cuda_toolkit: str | None = None
    metadata: Metadata
    compute_capability: str | None = None
    requires_datacenter_gpu: bool = Field(default=False)
    timing_mode: TimingModes = Field(
        default_factory=ProcessTimingMode, description="Method for extracting performance timing from metrics"
    )
    problem_type: Literal["cuda_cpp", "cuda_python"]


class CudaCppProblemSpec(ProblemSpecBase):
    arch_list: list[str] = Field(default_factory=list)


class CudaPythonProblemSpec(ProblemSpecBase):
    pass


def _load_yaml(path: Path) -> CudaCppProblemSpec | CudaPythonProblemSpec:
    raw = yaml.safe_load(path.read_text())
    adapter = TypeAdapter(CudaCppProblemSpec | CudaPythonProblemSpec)
    return adapter.validate_python(raw)


def _load_source_files(dir_path: Path) -> list[SourceFile]:
    if not dir_path.exists():
        return []
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path} exists but is not a directory.")

    out: list[SourceFile] = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            p = Path(root) / name
            rel = p.relative_to(dir_path).as_posix()
            try:
                content = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = p.read_text(encoding="utf-8", errors="replace")
            out.append(SourceFile(path=rel, content=content))
    return out


def _detect_solution_type(solution_dir: Path) -> tuple[list[Path], list[Path]]:
    all_files = [p for p in solution_dir.rglob("*") if p.is_file()]
    patch_files = [p for p in all_files if p.suffix.lower() in {".patch", ".diff"}]
    regular_files = [p for p in all_files if p not in patch_files]
    return patch_files, regular_files


def load_problem_dir(path: Path) -> tuple[Problem, Solution]:
    """
    Load a Problem + Solution from a directory structured in human-friendly format.
    Returns a tuple of (Problem, Solution).
    """
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Problem directory '{path}' not found.")

    spec = _load_yaml(path / "problem-spec.yaml")

    folder_task_id = path.name
    normalized_spec_id = spec.task_id.replace("/", "-")
    normalized_folder_id = folder_task_id.replace("/", "-")

    if normalized_spec_id != normalized_folder_id:
        raise ValueError(
            f"task_id mismatch: folder name '{folder_task_id}' does not match "
            f"task_id '{spec.task_id}' in problem-spec.yaml"
        )

    if spec.problem_type == "cuda_cpp":
        problem = CudaCppProblem(
            context_files=_load_source_files(path / "context"),
            test_files=_load_source_files(path / "test"),
            **spec.model_dump(exclude={"problem_type"}),
        )
    elif spec.problem_type == "cuda_python":
        problem = CudaPythonProblem(
            context_files=_load_source_files(path / "context"),
            test_files=_load_source_files(path / "test"),
            **spec.model_dump(exclude={"problem_type"}),
        )
    else:
        raise ValueError(f"Unknown problem type '{spec.problem_type}' in {path}.")

    solution_dir = path / "solution"
    if not solution_dir.exists() or not solution_dir.is_dir():
        raise ValueError(f"Missing solution directory: {solution_dir}")

    patch_files, regular_files = _detect_solution_type(solution_dir)

    if not patch_files and not regular_files:
        raise ValueError(f"No files found in solution directory: {solution_dir}")

    if len(patch_files) == 1 and not regular_files:
        patch_content = patch_files[0].read_text(encoding="utf-8")
        return problem, PatchSolution(task_id=spec.task_id, patch=patch_content)

    if not patch_files and regular_files:
        solution_files = _load_source_files(solution_dir)
        overlap = {f.path for f in solution_files} & {f.path for f in problem.test_files}
        if overlap:
            raise ValueError(f"Solution files overlap with test files in {path}: {overlap}")
        return problem, FileSolution(task_id=spec.task_id, files=solution_files)

    raise ValueError(
        "Invalid solution directory contents in "
        f"{solution_dir}: patch files={[p.name for p in patch_files]}, "
        f"regular files count={len(regular_files)}. "
        "Must be either exactly one .patch/.diff with no other files, or no .patch/.diff and at least one regular file."
    )


def _is_problem_dir(path: Path) -> bool:
    return (path / "problem-spec.yaml").exists()


def _find_problem_dirs(root_path: Path) -> Generator[Path, None, None]:
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir():
            continue
        if _is_problem_dir(subdir):
            yield subdir
        else:
            yield from _find_problem_dirs(subdir)


def load_problem_tree(root_path: Path) -> Generator[tuple[Problem, Solution], None, None]:
    """
    Recursively load all problems and solutions from a directory tree.
    Problem directories can be nested at any depth - they are identified by
    containing a problem-spec.yaml file.
    Returns a generator of (Problem, Solution) tuples.
    """
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Root directory '{root_path}' not found.")

    def _safe_load(path: Path) -> tuple[Problem, Solution]:
        try:
            return load_problem_dir(path)
        except Exception as e:
            raise RuntimeError(f"Error loading problem from '{path}': {e}") from e

    yield from (_safe_load(problem_dir) for problem_dir in _find_problem_dirs(root_path))
