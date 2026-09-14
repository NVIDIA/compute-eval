import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from compute_eval.data.data_model import FileSolution, Problem, SourceFile

STEP_NAME = "generate"
WORKDIR = "/workspace"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_.") or "item"


def write_problem_workspace(problem: Problem, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for source_file in problem.context_files:
        relative_path = PurePosixPath(source_file.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe relative path: {source_file.path}")
        path = workspace.joinpath(*relative_path.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source_file.content, encoding="utf-8")


def write_optimize_workspace(problem: Problem, workspace: Path) -> None:
    """Write workspace for optimize mode: context files + test files + baseline solution.

    Loosens the normal rules of engagement so the agent can see the full problem
    (test files and baseline implementation) rather than only context files.
    Precondition: problem.baseline_solution is not None.
    """
    if not isinstance(problem.baseline_solution, FileSolution):
        raise ValueError(
            f"write_optimize_workspace requires a FileSolution baseline for {problem.task_id}; "
            f"got {type(problem.baseline_solution).__name__}"
        )
    write_problem_workspace(problem, workspace)
    for source_file in problem.test_files:
        relative_path = PurePosixPath(source_file.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe relative path: {source_file.path}")
        path = workspace.joinpath(*relative_path.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source_file.content, encoding="utf-8")
    _write_baseline_solution(problem.baseline_solution, workspace)


def _write_baseline_solution(baseline: FileSolution, workspace: Path) -> None:
    """Write a FileSolution into the workspace.

    Mirrors the compute-eval evaluator's _setup_workspace logic: write each
    file directly, allowing solution files to intentionally overwrite context
    files. Only FileSolution is supported — PatchSolution baselines are not
    handled by the compute-eval evaluator either.
    """
    for source_file in baseline.files:
        relative_path = PurePosixPath(source_file.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe relative path in baseline solution: {source_file.path}")
        path = workspace.joinpath(*relative_path.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source_file.content, encoding="utf-8")


def _original_file_map(sources: list[SourceFile]) -> dict[str, str]:
    return {PurePosixPath(s.path).as_posix(): s.content for s in sources}


def collect_solution(
    problem: Problem,
    workspace: Path,
    *,
    extra_originals: Sequence[SourceFile] = (),
) -> FileSolution:
    """Collect changed files from an agent workspace back into a FileSolution.

    Test files are excluded by path (both the files themselves and anything under a
    directory that holds a test file), so agent edits to tests never reach the output
    datapack. Remaining files are dropped when byte-identical to a known original:
    context_files plus any extra_originals (optimize mode passes the baseline solution
    here so an unmodified baseline is not re-emitted).
    """
    original_files = _original_file_map(problem.context_files)
    original_files.update(_original_file_map(list(extra_originals)))
    hidden_test_files = {PurePosixPath(source.path) for source in problem.test_files}
    hidden_test_directories = {path.parent for path in hidden_test_files if path.parent != PurePosixPath(".")}
    files: list[SourceFile] = []
    if not workspace.exists():
        return FileSolution(task_id=problem.task_id, files=files)
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        relative_path = PurePosixPath(path.relative_to(workspace).as_posix())
        is_hidden_test = relative_path in hidden_test_files
        is_in_hidden_test_directory = any(
            relative_path.is_relative_to(directory) for directory in hidden_test_directories
        )
        if is_hidden_test or is_in_hidden_test_directory:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative_path_string = relative_path.as_posix()
        if original_files.get(relative_path_string) == content:
            continue
        files.append(SourceFile(path=relative_path_string, content=content))
    return FileSolution(task_id=problem.task_id, files=files)


def collect_optimize_solution(problem: Problem, workspace: Path) -> FileSolution:
    """Collect changed files from an optimize workspace.

    Same collection rules as generate mode (test files excluded by path, unchanged
    context files dropped), with one addition: every baseline solution file is
    always included, whether or not the agent modified it.

    The grader rebuilds from context files + test files + the graded solution
    only — it never re-provides the baseline. So a baseline solution file the
    agent left unchanged (e.g. solution.cu when only a header was optimized) must
    still be emitted, or grading fails with a missing-source error. Baseline
    files are still kept out of the unchanged-context drop; they are re-added here
    from the workspace copy (the agent's edits) or the original baseline content.
    """
    baseline_files = problem.baseline_solution.files if isinstance(problem.baseline_solution, FileSolution) else []
    collected = collect_solution(problem, workspace, extra_originals=baseline_files)

    present = {source.path for source in collected.files}
    baseline_paths = {PurePosixPath(source.path).as_posix() for source in baseline_files}
    missing = baseline_paths - present
    if not missing:
        return collected

    # Re-add baseline files the drop logic removed (agent left them unchanged).
    # Prefer the workspace copy so any agent edit still wins; fall back to the
    # original baseline content when the file is absent from the workspace.
    baseline_content = _original_file_map(list(baseline_files))
    files = list(collected.files)
    for path in sorted(missing):
        workspace_path = workspace.joinpath(*PurePosixPath(path).parts)
        try:
            content = workspace_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            content = baseline_content[path]
        files.append(SourceFile(path=path, content=content))
    return FileSolution(task_id=problem.task_id, files=files)


def raise_on_empty_solution(problem: Problem, solution: FileSolution, allow_empty: bool) -> None:
    if allow_empty or solution.files:
        return
    raise RuntimeError(
        f"no solution files produced for {problem.task_id}; use --allow-empty to write an empty FileSolution."
    )
