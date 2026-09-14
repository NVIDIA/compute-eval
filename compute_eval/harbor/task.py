import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment

from compute_eval.data.data_model import Problem
from compute_eval.gpu_lock.scheduler import (
    CONTAINER_CUDA_CHECKPOINT_PATH,
    CUDA_CHECKPOINT_PATH,
    GPU_TASK_LABEL,
)
from compute_eval.harbor.workspace import (
    STEP_NAME,
    WORKDIR,
    safe_name,
    write_optimize_workspace,
    write_problem_workspace,
)

LOGGER = logging.getLogger(__name__)

NVIDIA_DOCKER_ENVIRONMENT_IMPORT_PATH = "compute_eval.harbor.task:NvidiaDockerEnvironment"
SKILLS_DIR = "/tmp/compute-eval-harbor-skills"


class NvidiaDockerEnvironment(DockerEnvironment):
    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return super().capabilities.model_copy(update={"gpus": True})


def compute_eval_instruction(problem: Problem) -> str:
    return f"{problem.prompt}\n\nBuild command:\n{problem.build_command}\n"


def docker_compose_yaml(
    gpus: int,
    skills_dir: Path | None = None,
    gpu_scheduler_id: str | None = None,
) -> str:
    main: dict[str, Any] = {}
    if skills_dir is not None:
        main["volumes"] = [f"{skills_dir}:{SKILLS_DIR}:ro"]
    if gpus > 0:
        if gpu_scheduler_id is not None:
            main["labels"] = {GPU_TASK_LABEL: gpu_scheduler_id}
            main.setdefault("volumes", []).append(f"{CUDA_CHECKPOINT_PATH}:{CONTAINER_CUDA_CHECKPOINT_PATH}:ro")
        main.update(
            {
                "shm_size": "2g",
                "deploy": {
                    "resources": {
                        "reservations": {"devices": [{"driver": "nvidia", "count": gpus, "capabilities": ["gpu"]}]}
                    }
                },
            }
        )

    return yaml.safe_dump({"services": {"main": main}}, sort_keys=False)


def write_skills_dir(source: Path, destination: Path) -> None:
    if (source / "SKILL.md").is_file():
        shutil.copytree(source, destination / source.name)
    else:
        shutil.copytree(source, destination)


def skill_names(source: Path) -> list[str]:
    if (source / "SKILL.md").is_file():
        return [source.name]
    return sorted(child.name for child in source.iterdir() if child.is_dir() and (child / "SKILL.md").is_file())


def task_toml(
    problem: Problem,
    docker_image: str,
    timeout_sec: float,
    gpus: int,
    skills_dir: str | None,
    allowed_hosts: list[str],
) -> str:
    # [agent] timeout_sec is the single source of truth for the agent timeout
    # (resolve_agent_config deliberately leaves AgentConfig.override_timeout_sec unset).
    # [agent] user pins the UID Harbor uses for exec_as_agent calls. Without this,
    # Harbor creates an internal agent user with a fixed UID that may not match the
    # invoking user on the host, making trial directories impossible to delete without
    # sudo and blocking native resume after a hard kill.
    task_name = f"compute-eval/{safe_name(problem.task_id)}"
    skills_line = f"skills_dir = {json.dumps(skills_dir)}\n" if skills_dir is not None else ""
    network_lines = ""
    if allowed_hosts:
        network_lines = f'network_mode = "allowlist"\nallowed_hosts = {json.dumps(allowed_hosts)}\n'
    return f"""[task]
name = {json.dumps(task_name)}

[agent]
timeout_sec = {timeout_sec}
user = {os.getuid()}
{network_lines}

[environment]
docker_image = {json.dumps(docker_image)}
cpus = 4
memory_mb = 8192
storage_mb = 20480
gpus = {gpus}
workdir = {json.dumps(WORKDIR)}
{skills_line}

[[steps]]
name = {json.dumps(STEP_NAME)}
artifacts = [{{ source = {json.dumps(WORKDIR)}, destination = "workspace" }}]
"""


def write_harbor_task(
    problem: Problem,
    task_dir: Path,
    *,
    timeout_sec: float,
    gpus: int,
    docker_image: str,
    allowed_hosts: list[str] | None = None,
    skills_dir: list[str] | None = None,
    gpu_scheduler_id: str | None = None,
    instruction_builder: Callable[[Problem], str] = compute_eval_instruction,
    workspace_writer: Callable[[Problem, Path], None] = write_problem_workspace,
) -> None:
    if task_dir.exists():
        shutil.rmtree(task_dir)
    # Every problem must ship a baseline solution — it is the invariant the benchmark
    # relies on (generate builds from it as a reference; optimize starts from it).
    if problem.baseline_solution is None:
        raise ValueError(f"Problem {problem.task_id} has no baseline_solution. A baseline is required for every task.")
    host_skills_dirs = [Path(entry).expanduser().resolve() for entry in (skills_dir or [])]
    for host_skills_dir in host_skills_dirs:
        if not host_skills_dir.is_dir():
            raise ValueError(f"--skills-dir must be an existing directory: {host_skills_dir}")

    instruction = instruction_builder(problem)
    workspace = task_dir / "steps" / STEP_NAME / "workdir"
    workspace_writer(problem, workspace)

    (task_dir / "steps" / STEP_NAME / "instruction.md").write_text(instruction, encoding="utf-8")
    environment_dir = task_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    resolved_image = docker_image
    mounted_skills_dir: Path | None = None
    if host_skills_dirs:
        mounted_skills_dir = environment_dir / "skills"
        for host_skills_dir in host_skills_dirs:
            write_skills_dir(host_skills_dir, mounted_skills_dir)
    (environment_dir / "docker-compose.yaml").write_text(
        docker_compose_yaml(
            gpus,
            mounted_skills_dir,
            gpu_scheduler_id,
        ),
        encoding="utf-8",
    )

    container_skills_dir = SKILLS_DIR if host_skills_dirs else None
    (task_dir / "task.toml").write_text(
        task_toml(problem, resolved_image, timeout_sec, gpus, container_skills_dir, allowed_hosts or []),
        encoding="utf-8",
    )


def write_harbor_dataset(
    problems: list[Problem],
    dataset_dir: Path,
    *,
    timeout_sec: float,
    gpus: int,
    docker_image: str,
    allowed_hosts: list[str] | None = None,
    skills_dir: list[str] | None = None,
    gpu_scheduler_id: str | None = None,
) -> dict[Path, Problem]:
    return _write_dataset(
        problems,
        dataset_dir,
        timeout_sec=timeout_sec,
        gpus=gpus,
        allowed_hosts=allowed_hosts,
        docker_image=docker_image,
        skills_dir=skills_dir,
        gpu_scheduler_id=gpu_scheduler_id,
        instruction_builder=compute_eval_instruction,
        workspace_writer=write_problem_workspace,
    )


def _write_dataset(
    problems: list[Problem],
    dataset_dir: Path,
    *,
    timeout_sec: float,
    gpus: int,
    allowed_hosts: list[str] | None,
    docker_image: str,
    skills_dir: list[str] | None,
    gpu_scheduler_id: str | None,
    instruction_builder: Callable[[Problem], str],
    workspace_writer: Callable[[Problem, Path], None],
) -> dict[Path, Problem]:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    problem_by_task_dir: dict[Path, Problem] = {}
    for index, problem in enumerate(problems):
        task_dir = dataset_dir / f"{index:04d}-{safe_name(problem.task_id)}"
        write_harbor_task(
            problem,
            task_dir,
            timeout_sec=timeout_sec,
            gpus=gpus,
            allowed_hosts=allowed_hosts,
            docker_image=docker_image,
            skills_dir=skills_dir,
            gpu_scheduler_id=gpu_scheduler_id,
            instruction_builder=instruction_builder,
            workspace_writer=workspace_writer,
        )
        problem_by_task_dir[task_dir.resolve()] = problem

    return problem_by_task_dir


def optimize_instruction(problem: Problem) -> str:
    """Build the Harbor instruction for optimize mode.

    Unlike generate mode the agent receives the full problem context: it can see
    test files and starts from a known-good baseline. Its goal is performance, not
    initial correctness.
    """
    return (
        f"{problem.prompt}\n\n"
        f"Build command:\n{problem.build_command}\n\n"
        f"Test command:\n{problem.test_command}\n\n"
        f"Benchmark command:\n{problem.benchmark_command}\n\n"
        "A correct baseline implementation is already in your workspace. Optimize it for performance. \n"
        "The test command must still pass. \n"
        "Do not modify test files. \n"
        "You may have skills available that will help you with this task.\n"
    )


def write_optimize_dataset(
    problems: list[Problem],
    dataset_dir: Path,
    *,
    timeout_sec: float,
    gpus: int,
    docker_image: str,
    allowed_hosts: list[str] | None = None,
    skills_dir: list[str] | None = None,
    gpu_scheduler_id: str | None = None,
) -> dict[Path, Problem]:
    """Write an optimize Harbor dataset, filtering problems by benchmark eligibility.

    Problems without a benchmark_command are skipped — there is nothing to measure a
    speedup against. The remaining problems are written as Harbor tasks; the baseline
    invariant (every problem must have a baseline_solution) is enforced by
    write_harbor_task.

    Returns problem_by_task_dir.
    """
    eligible_problems: list[Problem] = []
    for problem in problems:
        if problem.benchmark_command is None:
            LOGGER.info("Problem %s has no benchmark_command, skipping (nothing to optimize against)", problem.task_id)
            continue
        eligible_problems.append(problem)

    return _write_dataset(
        eligible_problems,
        dataset_dir,
        timeout_sec=timeout_sec,
        gpus=gpus,
        allowed_hosts=allowed_hosts,
        docker_image=docker_image,
        skills_dir=skills_dir,
        gpu_scheduler_id=gpu_scheduler_id,
        instruction_builder=optimize_instruction,
        workspace_writer=write_optimize_workspace,
    )
