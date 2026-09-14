import argparse
import asyncio
import logging
import os
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harbor.job import Job
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import EnvironmentConfig, VerifierConfig
from harbor.models.trial.result import ExceptionInfo
from harbor.trial.hooks import TrialHookEvent

from compute_eval.data.data_model import FileSolution, Problem, ReleaseVersion
from compute_eval.data.data_pack import ProblemDatapack, SolutionDatapack
from compute_eval.gpu_lock.scheduler import GpuProcessScheduler
from compute_eval.harbor.agent_config import resolve_agent_config
from compute_eval.harbor.task import (
    NVIDIA_DOCKER_ENVIRONMENT_IMPORT_PATH,
    skill_names,
    write_harbor_dataset,
    write_optimize_dataset,
)
from compute_eval.harbor.telemetry import (
    GenerationArtifacts,
    TelemetryRun,
    TrialAttempt,
)
from compute_eval.harbor.traces import write_trace_archive
from compute_eval.harbor.usage import write_usage_report
from compute_eval.harbor.workspace import (
    STEP_NAME,
    collect_optimize_solution,
    collect_solution,
    safe_name,
)

LOGGER = logging.getLogger(__name__)


def _exception_summary(exception_info: ExceptionInfo) -> str:
    message = str(exception_info.exception_message)
    if len(message) > 2000:
        message = f"{message[:2000]}... <truncated>"
    return f"{exception_info.exception_type}: {message}"


def _load_resume_solutions(
    artifacts: GenerationArtifacts | None,
) -> tuple[list[FileSolution], set[str]]:
    """Load solutions from a previous completed run for resume support.

    This operates at the solution-datapack level: it reads the prior output
    datapack and skips whole task_ids that already produced a solution. It is
    complementary to Harbor's native trial-level resume (see _run_harbor_job),
    which handles interrupted jobs. This handles the case where a job ran to
    completion but some agents failed or timed out — a fresh job is started
    for just the unsolved problems.

    Returns (existing_solutions, task_ids_done).
    """
    if artifacts is None:
        return [], set()

    solutions: list[FileSolution] = []
    with SolutionDatapack(artifacts.solutions) as datapack:
        for item in datapack.read_items():
            if isinstance(item, FileSolution):
                solutions.append(item)
    task_ids = {s.task_id for s in solutions}
    LOGGER.info("Loaded %d existing solutions to resume from (%s)", len(solutions), artifacts.solutions)
    return solutions, task_ids


def _resume_artifacts(
    resume: str | bool | None,
    default_artifacts: GenerationArtifacts,
) -> GenerationArtifacts | None:
    if resume is None:
        return None
    if isinstance(resume, str):
        solutions_path = Path(resume).expanduser()
        if not solutions_path.exists():
            raise FileNotFoundError(f"--resume path does not exist: {solutions_path}")
    elif default_artifacts.solutions.exists():
        solutions_path = default_artifacts.solutions
        LOGGER.info("Resuming from existing output file: %s", solutions_path)
    else:
        raise FileNotFoundError(
            f"--resume specified but no existing output found at {default_artifacts.solutions}. "
            f"If this is a first run, omit --resume."
        )
    return GenerationArtifacts.from_solutions(solutions_path)


def _open_problem_datapack(problems_path: str | None, release: ReleaseVersion):
    """Open a local problem archive or source tree, or download the release."""
    if problems_path is None:
        return ProblemDatapack.from_huggingface(release)
    path = Path(problems_path).expanduser()
    if path.is_dir():
        return ProblemDatapack.from_directory(path, release=release)
    return ProblemDatapack.from_file(path, release)


def _load_problems(
    args: argparse.Namespace,
    skip_task_ids: set[str] | None,
) -> tuple[list[Problem], list[Problem]]:
    """Open the problem datapack and apply --limit and resume-skip filtering.

    Returns (problems_to_run, all_problems): the filtered set to write into the
    dataset, and the full set (pre-skip) used for group metadata and progress totals.

    Selection order: if COMPUTE_EVAL_SAMPLE_SEED is set, the full problem list is
    shuffled with that seed BEFORE --limit is applied, so `--limit N` yields a random
    N-problem sample rather than the first N. Without the env var, order is unchanged
    (deterministic head slice, preserving prior behavior). The seed and selected
    task_ids are logged so a run can be reproduced.
    """
    release = ReleaseVersion(args.release)
    with _open_problem_datapack(args.problems, release) as datapack:
        problems = list(datapack.read_items())

    seed_env = os.environ.get("COMPUTE_EVAL_SAMPLE_SEED")
    if seed_env is not None:
        seed = int(seed_env)
        random.Random(seed).shuffle(problems)
        LOGGER.info("Shuffled %d problems with COMPUTE_EVAL_SAMPLE_SEED=%d", len(problems), seed)

    if args.limit is not None:
        problems = problems[: args.limit]

    if seed_env is not None:
        LOGGER.info(
            "Sampled %d task_ids (seed=%s): %s",
            len(problems),
            seed_env,
            ", ".join(p.task_id for p in problems),
        )

    all_problems = list(problems)

    if skip_task_ids:
        before = len(problems)
        problems = [p for p in problems if p.task_id not in skip_task_ids]
        LOGGER.info("Skipping %d already-solved problems (resume)", before - len(problems))

    return problems, all_problems


def _build_retry_config(args: argparse.Namespace) -> RetryConfig:
    """Build the Harbor RetryConfig from --max-retries / --retry-include.

    With the default --max-retries 0 this is a no-op (behavior unchanged).
    When retries are enabled and no --retry-include is given, only transient
    ApiRateLimitError trials are retried, so genuine code failures still fail
    fast instead of burning the retry budget.
    """
    include = set(args.retry_include) if args.retry_include else {"ApiRateLimitError"}
    return RetryConfig(max_retries=args.max_retries, include_exceptions=include)


async def _run_harbor_job(
    args: argparse.Namespace,
    dataset_dir: Path,
    jobs_dir: Path,
    job_name: str,
    gpu_scheduler_id: str | None,
    completed_trials: list[Any],
    captured_attempts: list[TrialAttempt],
    state: dict[str, Any],
) -> JobResult:
    """Create and run a Harbor job.

    completed_trials, captured_attempts, and state["run_id"] are populated
    incrementally so the caller can build partial telemetry on KeyboardInterrupt.
    """
    agent_config = resolve_agent_config(args)
    job_config = JobConfig(
        jobs_dir=jobs_dir,
        job_name=job_name,
        n_attempts=args.attempts,
        n_concurrent_trials=args.concurrency,
        quiet=args.quiet,
        agents=[agent_config],
        datasets=[DatasetConfig(path=dataset_dir)],
        environment=EnvironmentConfig(
            import_path=NVIDIA_DOCKER_ENVIRONMENT_IMPORT_PATH if args.gpus > 0 else None,
        ),
        verifier=VerifierConfig(disable=True),
        retry=_build_retry_config(args),
    )
    scheduler = GpuProcessScheduler(task_label_value=gpu_scheduler_id) if gpu_scheduler_id is not None else None
    if scheduler is not None:
        await scheduler.start()

    try:
        try:
            job = await Job.create(job_config)
        except FileExistsError as exc:
            raise SystemExit(
                f"Cannot resume the existing Harbor jobs directory because the run "
                f"configuration has changed:\n  {exc}\n\n"
                f"To start fresh, remove the jobs directory:\n"
                f"  rm -rf {jobs_dir}\n\n"
                f"Or use --output-dir to write to a new location."
            ) from exc

        state["run_id"] = job.id
        attempt_counts: Counter[str] = Counter()

        async def record_trial_attempt(event: TrialHookEvent) -> None:
            completed_trials.append(event.result)
            attempt_counts[event.result.trial_name] += 1
            captured_attempts.append(TrialAttempt.capture(event.result, attempt_counts[event.result.trial_name]))

        job.on_trial_ended(record_trial_attempt)
        if scheduler is None:
            job_result = await job.run()
        else:
            job_result = await scheduler.run_guarded(job.run())
    except BaseException:
        if scheduler is not None:
            try:
                await scheduler.stop()
            except BaseException:
                LOGGER.exception("GPU scheduler cleanup failed after the Harbor job failed.")
        raise
    else:
        if scheduler is not None:
            await scheduler.stop()
        return job_result


def _collect_job_solutions(
    trial_results: list[Any],
    problem_by_task_dir: dict[Path, Problem],
    allow_empty: bool,
    collector: Callable[[Problem, Path], FileSolution] = collect_solution,
) -> tuple[list[FileSolution], int]:
    """Collect FileSolution objects from a list of completed Harbor TrialResults.

    All failures (trial crash, step exception, missing artifacts, empty solution)
    are logged as warnings and skipped so partial results are never discarded.
    A re-run with --resume can pick up the skipped problems.

    Returns (solutions, skipped_count).
    """
    solutions: list[FileSolution] = []
    skipped = 0
    for trial_result in trial_results:
        # Trial-level failure (e.g. timeout, container crash).
        if trial_result.exception_info is not None:
            LOGGER.warning(
                "Skipping %s (%s): %s",
                trial_result.trial_name,
                trial_result.task_name,
                _exception_summary(trial_result.exception_info),
            )
            skipped += 1
            continue

        # Step-level exception: log it but fall through to the artifact checks.
        # Harbor may still have collected artifacts even when a step is flagged as
        # failed (e.g. agent timeout after it already wrote a solution), so the
        # artifact existence and workspace content checks below are the authoritative
        # signal — not the step status.
        if trial_result.step_results is not None:
            for step in trial_result.step_results:
                if step.exception_info is not None:
                    LOGGER.warning(
                        "%s (%s): step %s raised %s — checking for artifacts anyway",
                        trial_result.trial_name,
                        trial_result.task_name,
                        step.step_name,
                        _exception_summary(step.exception_info),
                    )

        task_path = trial_result.config.task.path
        if task_path is None:
            raise ValueError(f"Harbor trial {trial_result.trial_name} did not use a local task path.")
        task_dir = task_path.resolve()
        problem = problem_by_task_dir.get(task_dir)
        if problem is None:
            LOGGER.error(
                "Skipping Harbor trial %s because task path %s did not match the materialized dataset.",
                trial_result.trial_name,
                task_dir,
            )
            skipped += 1
            continue

        trial_dir = trial_result.config.trials_dir / trial_result.trial_name
        step_artifacts = trial_dir / "steps" / STEP_NAME / "artifacts"
        if not step_artifacts.exists():
            LOGGER.warning(
                "Skipping %s (%s): Harbor did not collect step artifacts",
                trial_result.trial_name,
                trial_result.task_name,
            )
            skipped += 1
            continue

        solution = collector(problem, step_artifacts / "workspace")
        if not allow_empty and not solution.files:
            LOGGER.warning(
                "Skipping %s (%s): agent produced no output files",
                trial_result.trial_name,
                trial_result.task_name,
            )
            skipped += 1
            continue

        solutions.append(solution)
    return solutions, skipped


def _build_telemetry_run(
    job_result: JobResult | None,
    job_state: dict[str, Any],
    release_value: str,
    problem_by_task_dir: dict[Path, Problem],
    captured_attempts: list[TrialAttempt],
    args: argparse.Namespace,
) -> TelemetryRun | None:
    """Build a TelemetryRun from a completed or interrupted Harbor job.

    Returns None when no job was run (empty dataset) or when no run_id was
    assigned before an interrupt, so callers can guard with `if telemetry_run`.
    """
    skills = [
        name for skills_dir in (args.skills_dir or []) for name in skill_names(Path(skills_dir).expanduser().resolve())
    ]
    if job_result is not None:
        return TelemetryRun.from_job(
            job_result,
            release_value,
            problem_by_task_dir,
            captured_attempts=captured_attempts,
            skills_loaded=skills,
        )
    if job_state.get("run_id") is not None:
        return TelemetryRun.from_attempts(
            job_state["run_id"],
            release_value,
            problem_by_task_dir,
            captured_attempts=captured_attempts,
            skills_loaded=skills,
        )
    return None


def _write_telemetry(
    artifacts: GenerationArtifacts,
    telemetry_run: TelemetryRun,
    resume_artifacts: GenerationArtifacts | None,
) -> None:
    """Write usage report and trajectory archive for a completed run."""
    usage_path = write_usage_report(
        artifacts,
        telemetry_run,
        previous_report=resume_artifacts.usage if resume_artifacts is not None else None,
    )
    LOGGER.info("Wrote Harbor usage report to %s", usage_path)
    trace_path = write_trace_archive(
        artifacts,
        telemetry_run,
        previous_archive=resume_artifacts.trajectories if resume_artifacts is not None else None,
    )
    LOGGER.info("Wrote Harbor trajectory archive to %s", trace_path)


def _run_generation(
    args: argparse.Namespace,
    *,
    name_suffix: str,
    dataset_writer: Callable[..., dict[Path, Problem]],
    collector: Callable[[Problem, Path], FileSolution],
    allow_empty: bool,
) -> None:
    """Shared driver for the generate and optimize subcommands.

    The two modes differ only in: the output/dir name suffix, which dataset builder
    materializes the Harbor tasks, which collector scrapes solutions back out, and
    whether an empty solution is allowed. Everything else — resume setup, stable-vs-temp
    run dirs, GPU scheduler wiring, job run + interrupt handling, telemetry, atomic
    write, and cleanup — is identical.
    """
    release = ReleaseVersion(args.release)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{release.value}-{safe_name(args.model or args.agent or 'harbor')}"
    artifacts = GenerationArtifacts.from_solutions(output_dir / f"{run_name}{name_suffix}-solutions.tar.gz")
    tmp_solutions_path = output_dir / f"{run_name}{name_suffix}-solutions.tar.gz.tmp"

    is_resume = args.resume is not None
    resume_artifacts = _resume_artifacts(args.resume, artifacts)
    resumed_solutions, skip_task_ids = _load_resume_solutions(resume_artifacts)

    # For fresh runs, dataset_dir and jobs_dir are stable paths derived from
    # run_name alongside the output file. Harbor's config equality check
    # includes DatasetConfig.path, so the dataset path must be the same across
    # runs for native trial-level resume to work on an interrupted (hard-killed) job.
    #
    # For --resume runs the problem set is different (already-solved problems
    # are filtered out), so Harbor config matching would fail anyway. Use a
    # temp dir so there's no collision with the original run's stable paths.
    if is_resume:
        tmp_run_dir = Path(tempfile.mkdtemp(prefix=f"compute-eval-harbor-{release.value}-"))
        dataset_dir = tmp_run_dir / "dataset"
        jobs_dir = tmp_run_dir / "jobs"
        cleanup_dir: Path | None = tmp_run_dir
    else:
        dataset_dir = output_dir / f"{run_name}{name_suffix}-dataset"
        jobs_dir = output_dir / f"{run_name}{name_suffix}-jobs"
        cleanup_dir = None

    gpu_scheduler_id = None
    if args.gpu_scheduler and args.gpus > 0 and args.concurrency > 1:
        gpu_scheduler_id = tmp_run_dir.name if is_resume else run_name

    LOGGER.info("Output:   %s", artifacts.solutions)
    LOGGER.info("Jobs dir: %s", jobs_dir)
    problems, all_problems = _load_problems(args, skip_task_ids)
    LOGGER.info("Writing Harbor dataset to %s", dataset_dir)
    problem_by_task_dir = dataset_writer(
        problems,
        dataset_dir,
        timeout_sec=args.timeout,
        gpus=args.gpus,
        allowed_hosts=args.allow_host,
        docker_image=args.docker_image,
        skills_dir=args.skills_dir,
        gpu_scheduler_id=gpu_scheduler_id,
    )
    try:
        solutions: list[FileSolution] = list(resumed_solutions)
        telemetry_run: TelemetryRun | None = None
        skipped = 0
        interrupted = False

        if problem_by_task_dir:
            LOGGER.info("Running Harbor job with %s attempt(s) per problem", args.attempts)
            completed_trials: list[Any] = []
            captured_attempts: list[TrialAttempt] = []
            job_state: dict[str, Any] = {}
            job_result: JobResult | None = None
            try:
                job_result = asyncio.run(
                    _run_harbor_job(
                        args,
                        dataset_dir,
                        jobs_dir,
                        run_name,
                        gpu_scheduler_id,
                        completed_trials,
                        captured_attempts,
                        job_state,
                    )
                )
            except KeyboardInterrupt:
                interrupted = True
                LOGGER.warning(
                    "Interrupted — collecting %d completed trial(s) and writing partial output",
                    len(completed_trials),
                )

            trial_results = completed_trials if interrupted else job_result.trial_results
            telemetry_run = _build_telemetry_run(
                job_result, job_state, release.value, problem_by_task_dir, captured_attempts, args
            )
            new_solutions, skipped = _collect_job_solutions(
                trial_results,
                problem_by_task_dir,
                allow_empty,
                collector=collector,
            )
            solutions.extend(new_solutions)

        # Write to a temp file then rename atomically so a crash mid-write never
        # corrupts the output file. The previous good file survives intact.
        SolutionDatapack.create(
            file_path=tmp_solutions_path,
            items=solutions,
            release=release,
            groups=sorted({problem.group for problem in all_problems}),
        )
        os.replace(tmp_solutions_path, artifacts.solutions)

        if interrupted:
            LOGGER.warning(
                "Wrote %d of %d solutions to %s (partial — job was interrupted; re-run with --resume to continue)",
                len(solutions),
                len(all_problems),
                artifacts.solutions,
            )
        else:
            log = LOGGER.warning if skipped else LOGGER.info
            log(
                "Wrote %d of %d solutions to %s%s",
                len(solutions),
                len(all_problems),
                artifacts.solutions,
                f" ({skipped} skipped — re-run with --resume to retry)" if skipped else "",
            )

        if telemetry_run is not None:
            _write_telemetry(artifacts, telemetry_run, resume_artifacts)

        # Clean up stable dataset dir on success — it's only needed to give
        # Harbor a stable DatasetConfig.path for interrupted-run recovery.
        # The jobs dir is left in place (Harbor's own cleanup handles it).
        if not is_resume and not args.keep_run_dir:
            shutil.rmtree(dataset_dir, ignore_errors=True)

    finally:
        tmp_solutions_path.unlink(missing_ok=True)
        if cleanup_dir is not None:
            if args.keep_run_dir:
                LOGGER.info("Keeping Harbor run directory: %s", cleanup_dir)
            else:
                shutil.rmtree(cleanup_dir, ignore_errors=True)


def _generate(args: argparse.Namespace) -> None:
    _run_generation(
        args,
        name_suffix="",
        dataset_writer=write_harbor_dataset,
        collector=collect_solution,
        allow_empty=args.allow_empty,
    )


def _optimize(args: argparse.Namespace) -> None:
    _run_generation(
        args,
        name_suffix="-optimize",
        dataset_writer=write_optimize_dataset,
        collector=collect_optimize_solution,
        allow_empty=False,
    )


def _add_common_generation_args(subparser: argparse.ArgumentParser) -> None:
    """Add the flags shared by the generate and optimize subcommands.

    Only generate's --allow-empty differs between the two modes; it is added by
    that caller.
    """
    inputs = subparser.add_argument_group("inputs")
    inputs.add_argument("--release", default=ReleaseVersion.V2026_2.value)
    inputs.add_argument(
        "--problems",
        metavar="PATH",
        help="Load problems from a local .tar.gz archive or YAML source directory. Downloads the release if omitted.",
    )
    inputs.add_argument("--output-dir", default=".")

    agent = subparser.add_argument_group("agent")
    agent.add_argument("--agent", default="claude-code")
    agent.add_argument("--model")
    agent.add_argument(
        "--skills-dir",
        action="append",
        metavar="PATH",
        help="Skill directory to mount for the agent. Repeat to load multiple skills.",
    )
    agent.add_argument(
        "--forward-env",
        action="append",
        metavar="VAR",
        help="Copy this environment variable from the launch environment into the agent's "
        "environment when it is set. Repeat for several. Useful for credentials a mounted "
        "skill needs to reach an external service.",
    )
    agent.add_argument(
        "--docker-image",
        required=True,
        help="Task image used by Harbor agents (required). It is used verbatim: the image must ship "
        "the full solve toolchain (compilers, profilers, agent CLI). Build docker/harbor-agent/Dockerfile "
        "for a solve-safe default.",
    )
    agent.add_argument(
        "--opencode-config",
        help="Path to an OpenCode JSON config merged into ~/.config/opencode/opencode.json before each run.",
    )

    execution = subparser.add_argument_group("execution")
    execution.add_argument("--attempts", type=int, default=1, help="Agent attempts per problem. Defaults to 1.")
    execution.add_argument("--concurrency", type=int, default=1, help="Concurrent Harbor trials. Defaults to 1.")
    execution.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retry failed trials up to N times (default 0). Targets transient ApiRateLimitError "
        "unless --retry-include is given.",
    )
    execution.add_argument(
        "--retry-include",
        action="append",
        metavar="EXC",
        help="Exception type name to retry, e.g. ApiRateLimitError. Repeat for several. "
        "Defaults to ApiRateLimitError when --max-retries > 0.",
    )
    execution.add_argument("--timeout", type=float, default=1800.0, help="Agent timeout in seconds. Defaults to 1800.")
    execution.add_argument("--gpus", type=int, default=1, help="GPUs per task. Use 0 for CPU-only debugging.")
    execution.add_argument(
        "--gpu-scheduler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the host-side GPU process scheduler for concurrent GPU trials.",
    )
    execution.add_argument(
        "--allow-host",
        action="append",
        help="Allow agent egress to this hostname, wildcard hostname, IP address, or CIDR. Repeat as needed. "
        "Uses Harbor's native Docker network policy.",
    )
    execution.add_argument("--keep-run-dir", action="store_true", help="Keep the generated Harbor dataset files.")
    execution.add_argument("--quiet", action="store_true")
    execution.add_argument("--limit", type=int, help="Cap the number of problems processed.")
    execution.add_argument(
        "--resume",
        nargs="?",
        const=True,
        metavar="PATH",
        help="Resume from PATH, or from the expected output file when PATH is omitted.",
    )


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate ComputeEval solution datapacks with Harbor agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a ComputeEval solutions datapack.")
    _add_common_generation_args(generate)
    generate.add_argument("--allow-empty", action="store_true")
    generate.set_defaults(func=_generate)

    optimize = subparsers.add_parser(
        "optimize",
        help="Generate performance-optimized ComputeEval solution datapacks with Harbor agents.",
    )
    _add_common_generation_args(optimize)
    optimize.set_defaults(func=_optimize)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
