"""Export ComputeEval problems as single-step, self-grading Harbor tasks.

NeMo Gym's Harbor server and the Harbor CLI run the task folder as is: the agent works in /workspace,
then a fresh GPU sandbox of the same image gets only /workspace and tests/, and runs tests/test.sh.
tests/ is only uploaded then, so the agent sees neither the hidden tests nor the grader.

Each task folder holds:
  task.toml            single step, separate verifier, no network, /workspace kept as an artifact
  instruction.md       the problem prompt and build command
  environment/         context files, uploaded to the workdir
  tests/test.sh        unpacks and runs the grader
  tests/grader.tar.gz  the compute_eval package that wrote the task (compute_eval.harbor.grade)
  tests/problem.json   the whole problem, hidden tests and baseline included
  solution/solve.sh    writes the baseline solution into /workspace (used by oracle runs)

This module must not import the harbor library, so it runs with the eval extra alone.
"""

import argparse
import base64
import functools
import gzip
import io
import json
import logging
import shlex
import shutil
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import compute_eval
from compute_eval.data.data_model import FileSolution, Problem, ReleaseVersion
from compute_eval.data.data_pack import open_problem_datapack
from compute_eval.harbor.workspace import (
    WORKDIR,
    compute_eval_instruction,
    safe_name,
    write_problem_workspace,
)
from compute_eval.utils.eval_utils import GpuInfo, parse_semver

LOGGER = logging.getLogger(__name__)

# The image's venv with the grader's third-party packages.
GRADER_PYTHON = "/opt/grader-venv/bin/python"
GRADER_ARCHIVE = "grader.tar.gz"
# Package data the grader reads at run time.
GRADER_DATA_DIRS = ("utils/language_data",)
# Run as `python <unpacked folder>`: the folder goes on sys.path without a PYTHONPATH the problem's
# own build and test commands would inherit.
GRADER_MAIN = "from compute_eval.harbor.grade import main\n\nmain()\n"
DEFAULT_AGENT_TIMEOUT_SEC = 1800.0
DEFAULT_PROBLEM_TIMEOUT_SEC = 60.0
# The evaluator's benchmark timeout when a problem sets none.
DEFAULT_BENCHMARK_TIMEOUT_SEC = 300.0
# Time for the grader to start and assemble the workspace, on top of the problem's own commands.
VERIFIER_OVERHEAD_SEC = 60.0

CPUS = 4
MEMORY_MB = 8192
STORAGE_MB = 20480


@dataclass(frozen=True)
class TaskOptions:
    """Settings shared by every task of an export."""

    # Image the agent works in and the grader runs in; it must hold the grader's third-party packages.
    image: str
    agent_timeout_sec: float = DEFAULT_AGENT_TIMEOUT_SEC
    # No network for the agent's sandbox and the grading sandbox. False writes no network setting,
    # for a runner that sets the network itself. Not a CLI option: a shared export is restricted.
    restrict_network: bool = True
    # Profiler the grader times the solution and the baseline with, for problems that have a benchmark
    # command. The image must hold it (Dockerfile.eval builds the CUPTI one). None turns timing off.
    profile_mode: str | None = "cupti"


def timed(problem: Problem, options: TaskOptions) -> bool:
    return options.profile_mode is not None and problem.benchmark_command is not None


def verifier_timeout_sec(problem: Problem, options: TaskOptions) -> float:
    """Build and test each get the problem timeout.

    A timed problem also benchmarks the solution, then builds, tests and benchmarks the baseline.
    """
    step = problem.timeout_seconds or DEFAULT_PROBLEM_TIMEOUT_SEC
    timeout = VERIFIER_OVERHEAD_SEC + 2 * step
    if timed(problem, options):
        timeout += 2 * step + 2 * (problem.timeout_seconds or DEFAULT_BENCHMARK_TIMEOUT_SEC)
    return timeout


def task_toml(problem: Problem, options: TaskOptions) -> str:
    # No GPU names: [metadata] says what the problem needs (compute_capability, requires_datacenter_gpu)
    # and the evaluator decides on the GPU it finds. A deployment that must route by GPU type adds it.
    metadata = {
        "compute_eval_task_id": problem.task_id,
        "group": problem.group,
        "type": problem.type,
        "min_cuda_toolkit": problem.min_cuda_toolkit,
        "compute_capability": problem.compute_capability,
    }
    metadata_lines = "".join(f"{key} = {json.dumps(value)}\n" for key, value in metadata.items() if value is not None)
    resources = f"cpus = {CPUS}\nmemory_mb = {MEMORY_MB}\nstorage_mb = {STORAGE_MB}\ngpus = 1\n"
    # Set for both: the grading sandbox is a separate one and would otherwise only inherit it.
    network = 'network_mode = "no-network"\n' if options.restrict_network else ""
    return f"""artifacts = [{json.dumps(WORKDIR)}]

[task]
name = {json.dumps(f"compute-eval/{safe_name(problem.task_id)}")}

[metadata]
{metadata_lines}requires_datacenter_gpu = {json.dumps(problem.requires_datacenter_gpu)}

[agent]
timeout_sec = {options.agent_timeout_sec}

[environment]
docker_image = {json.dumps(options.image)}
{resources}workdir = {json.dumps(WORKDIR)}
{network}
[verifier]
timeout_sec = {verifier_timeout_sec(problem, options)}
environment_mode = "separate"
{network}"""


def _grader_files() -> list[tuple[str, bytes]]:
    """The installed compute_eval package as (archive path, content): its code and the data the grader reads.

    It packs the package as installed, so export tasks that will be shared from an installed release,
    not from a development checkout.
    """
    package = Path(compute_eval.__file__).parent
    paths = set(package.rglob("*.py"))
    for data_dir in GRADER_DATA_DIRS:
        paths.update(path for path in (package / data_dir).rglob("*") if path.is_file())
    files = [("__main__.py", GRADER_MAIN.encode("utf-8"))]
    files += [(f"compute_eval/{path.relative_to(package).as_posix()}", path.read_bytes()) for path in sorted(paths)]
    return files


@functools.cache
def grader_archive() -> bytes:
    """tests/grader.tar.gz. Byte-identical for the same compute_eval code, so exports are reproducible."""
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, content in _grader_files():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_sh(problem: Problem, options: TaskOptions) -> str:
    # The grader writes /logs/verifier/reward.json only when grading finished, so a grader crash
    # (or a failed unpack) shows up as a missing reward instead of a silent 0.
    profile = f" \\\n    --profile-mode {shlex.quote(options.profile_mode)}" if timed(problem, options) else ""
    return f"""#!/bin/bash
set -euo pipefail
grader=$(mktemp -d)
tar -xzf /tests/{GRADER_ARCHIVE} -C "$grader"
exec {GRADER_PYTHON} "$grader" \\
    --problem /tests/problem.json \\
    --workspace {WORKDIR} \\
    --out /logs/verifier{profile}
"""


def solve_sh(solution: FileSolution) -> str:
    """Write the solution files byte for byte; base64 avoids any quoting or delimiter issues."""
    lines = ["#!/bin/bash", "set -euo pipefail", f"cd {WORKDIR}"]
    for source_file in solution.files:
        path = _safe_relative_path(source_file.path)
        encoded = base64.b64encode(source_file.content.encode("utf-8")).decode("ascii")
        lines.append(f"mkdir -p {shlex.quote(path.parent.as_posix())}")
        lines.append(f"echo {encoded} | base64 -d > {shlex.quote(path.as_posix())}")
    return "\n".join(lines) + "\n"


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    return path


def _write_script(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    # Harbor runs these directly, so they must be executable.
    path.chmod(0o755)


def write_task(problem: Problem, task_dir: Path, options: TaskOptions) -> None:
    if not isinstance(problem.baseline_solution, FileSolution):
        raise ValueError(f"Problem {problem.task_id} needs a FileSolution baseline for solution/solve.sh.")
    if task_dir.exists():
        shutil.rmtree(task_dir)

    environment = task_dir / "environment"
    environment.mkdir(parents=True)
    write_problem_workspace(problem, environment)
    (task_dir / "instruction.md").write_text(compute_eval_instruction(problem), encoding="utf-8")
    (task_dir / "task.toml").write_text(task_toml(problem, options), encoding="utf-8")
    _write_script(task_dir / "tests" / "test.sh", test_sh(problem, options))
    (task_dir / "tests" / GRADER_ARCHIVE).write_bytes(grader_archive())
    (task_dir / "tests" / "problem.json").write_text(problem.model_dump_json(), encoding="utf-8")
    _write_script(task_dir / "solution" / "solve.sh", solve_sh(problem.baseline_solution))


def runs_on(problem: Problem, gpu: GpuInfo | None) -> bool:
    """Whether the evaluator would run a problem on a GPU. No GPU means no filter.

    This is the evaluator's own rule (BaseEvaluator._validate_cc): the GPU's compute capability must
    be the problem's or newer, and a datacenter GPU where the problem asks for one.
    """
    required = parse_semver(problem.compute_capability)
    if gpu is None or required is None:
        return True
    if (gpu.compute_capability or (0, 0)) < required[:2]:
        return False
    return not problem.requires_datacenter_gpu or gpu.is_datacenter_gpu


def select_problems(
    problems: Sequence[Problem],
    *,
    task_ids: Sequence[str] = (),
    gpu: GpuInfo | None = None,
) -> list[Problem]:
    selected = [problem for problem in problems if not task_ids or problem.task_id in task_ids]
    missing = set(task_ids) - {problem.task_id for problem in selected}
    if missing:
        raise ValueError(f"Unknown task ids: {sorted(missing)}")

    runnable: list[Problem] = []
    for problem in selected:
        if runs_on(problem, gpu):
            runnable.append(problem)
        else:
            datacenter = ", datacenter GPU" if problem.requires_datacenter_gpu else ""
            LOGGER.info(
                "Skipping %s: needs compute capability %s%s", problem.task_id, problem.compute_capability, datacenter
            )
    return runnable


def write_dataset(problems: Sequence[Problem], dataset_dir: Path, options: TaskOptions) -> list[Path]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if any(dataset_dir.iterdir()):
        raise ValueError(f"Output directory {dataset_dir} is not empty. Choose a new or empty directory.")

    task_dirs: list[Path] = []
    for problem in problems:
        task_dir = dataset_dir / safe_name(problem.task_id)
        if task_dir in task_dirs:
            raise ValueError(f"Two problems map to the same task folder: {task_dir.name}")
        write_task(problem, task_dir, options)
        task_dirs.append(task_dir)
    return task_dirs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export ComputeEval problems as single-step, self-grading Harbor tasks.",
    )
    parser.add_argument(
        "--release",
        default=ReleaseVersion.V2026_2.value,
        choices=[release.value for release in ReleaseVersion if release is not ReleaseVersion.INTERNAL],
        help="The published release to export, downloaded from Hugging Face.",
    )
    parser.add_argument("--output-dir", required=True, help="New or empty directory for the exported task folders.")
    parser.add_argument(
        "--image",
        required=True,
        help="Image the agent works in and the grader runs in; it must hold the grader's third-party "
        "packages (docker/Dockerfile.eval --target grader). Pin it by digest.",
    )
    parser.add_argument("--agent-timeout", type=float, default=DEFAULT_AGENT_TIMEOUT_SEC)
    parser.add_argument("--group", action="append", help="Only export this group. Repeat for several.")
    parser.add_argument("--task-id", action="append", help="Only export this task id. Repeat for several.")
    parser.add_argument(
        "--gpu-compute-capability",
        help="Compute capability of the GPUs the tasks will run on, such as 9.0 for H100. Problems the evaluator "
        "would skip there are left out, so they are never graded as skipped. Leave it out to export "
        "every problem.",
    )
    parser.add_argument(
        "--datacenter-gpu",
        action="store_true",
        help="Those GPUs are datacenter GPUs. Goes with --gpu-compute-capability.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    gpu = None
    if args.gpu_compute_capability is not None:
        capability = parse_semver(args.gpu_compute_capability)
        if capability is None:
            parser.error(f"--gpu-compute-capability must look like 9.0, got {args.gpu_compute_capability!r}")
        gpu = GpuInfo(num_gpus=1, compute_capability=capability[:2], is_datacenter_gpu=args.datacenter_gpu)
    elif args.datacenter_gpu:
        parser.error("--datacenter-gpu needs --gpu-compute-capability")

    with open_problem_datapack(ReleaseVersion(args.release), include=args.group) as datapack:
        problems = list(datapack.read_items())
    selected = select_problems(problems, task_ids=args.task_id or (), gpu=gpu)
    options = TaskOptions(
        image=args.image,
        agent_timeout_sec=args.agent_timeout,
    )
    task_dirs = write_dataset(selected, Path(args.output_dir).expanduser(), options)
    LOGGER.info("Wrote %d of %d tasks to %s", len(task_dirs), len(problems), args.output_dir)


if __name__ == "__main__":
    main()
