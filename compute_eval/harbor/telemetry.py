from datetime import datetime
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from harbor.models.job.result import JobResult
from harbor.models.trajectories.trajectory import Trajectory
from harbor.models.trial.result import ModelInfo, TrialResult
from pydantic import BaseModel, ConfigDict

from compute_eval.data.data_model import Problem
from compute_eval.harbor.workspace import STEP_NAME

SOLUTIONS_SUFFIX = "-solutions.tar.gz"
USAGE_SUFFIX = "-harbor-usage.jsonl"
TRACES_SUFFIX = "-harbor-trajectories.tar.gz"
CostSource = Literal["reported", "litellm"]


class TelemetryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerationArtifacts(TelemetryModel):
    solutions: Path
    usage: Path
    trajectories: Path

    @classmethod
    def from_solutions(cls, solutions: Path) -> Self:
        if not solutions.name.endswith(SOLUTIONS_SUFFIX):
            raise ValueError(f"Solutions datapack name must end with {SOLUTIONS_SUFFIX}: {solutions}")
        stem = solutions.name.removesuffix(SOLUTIONS_SUFFIX)
        return cls(
            solutions=solutions,
            usage=solutions.with_name(f"{stem}{USAGE_SUFFIX}"),
            trajectories=solutions.with_name(f"{stem}{TRACES_SUFFIX}"),
        )


class TrialAttempt(TelemetryModel):
    result: TrialResult
    trajectory: Trajectory | None
    trajectory_json: str | None
    attempt: int
    problem: Problem | None = None
    is_final: bool = False

    @classmethod
    def capture(cls, result: TrialResult, attempt: int) -> Self:
        trial_dir = result.config.trials_dir / result.trial_name
        path = trial_dir / "steps" / STEP_NAME / "agent" / "trajectory.json"
        if not path.exists():
            return cls(result=result, trajectory=None, trajectory_json=None, attempt=attempt)

        try:
            trajectory_json = path.read_text(encoding="utf-8")
        except OSError:
            return cls(result=result, trajectory=None, trajectory_json=None, attempt=attempt)

        try:
            trajectory = Trajectory.model_validate_json(trajectory_json)
        except ValueError:
            trajectory = None
        return cls(
            result=result,
            trajectory=trajectory,
            trajectory_json=trajectory_json,
            attempt=attempt,
        )

    @property
    def error_type(self) -> str | None:
        if self.result.exception_info is not None:
            return self.result.exception_info.exception_type
        for step_result in self.result.step_results or []:
            if step_result.exception_info is not None:
                return step_result.exception_info.exception_type
        return None

    @property
    def model_info(self) -> ModelInfo | None:
        return self.result.agent_info.model_info

    def trajectory_path(self, run_id: UUID) -> str | None:
        if self.trajectory_json is None:
            return None
        return f"traces/{run_id}/{self.result.id}/attempt-{self.attempt}/trajectory.json"


class TelemetryRun(TelemetryModel):
    run_id: UUID
    release: str
    attempts: tuple[TrialAttempt, ...]
    skills_loaded: tuple[str, ...]

    @classmethod
    def from_job(
        cls,
        job_result: JobResult,
        release: str,
        problem_by_task_dir: dict[Path, Problem],
        captured_attempts: list[TrialAttempt] | None = None,
        skills_loaded: list[str] | None = None,
    ) -> Self:
        attempts = list(captured_attempts or [])
        captured_ids = {attempt.result.id for attempt in attempts}
        for result in job_result.trial_results:
            if result.id in captured_ids:
                continue
            next_attempt = 1 + max(
                (attempt.attempt for attempt in attempts if attempt.result.trial_name == result.trial_name),
                default=0,
            )
            attempts.append(TrialAttempt.capture(result, next_attempt))

        return cls._build(
            run_id=job_result.id,
            release=release,
            problem_by_task_dir=problem_by_task_dir,
            attempts=attempts,
            skills_loaded=skills_loaded,
        )

    @classmethod
    def from_attempts(
        cls,
        run_id: UUID,
        release: str,
        problem_by_task_dir: dict[Path, Problem],
        captured_attempts: list[TrialAttempt],
        skills_loaded: list[str] | None = None,
    ) -> Self:
        """Build from attempts alone (no JobResult) — for interrupted jobs."""
        return cls._build(
            run_id=run_id,
            release=release,
            problem_by_task_dir=problem_by_task_dir,
            attempts=list(captured_attempts),
            skills_loaded=skills_loaded,
        )

    @classmethod
    def _build(
        cls,
        run_id: UUID,
        release: str,
        problem_by_task_dir: dict[Path, Problem],
        attempts: list[TrialAttempt],
        skills_loaded: list[str] | None,
    ) -> Self:
        final_attempts: dict[str, int] = {}
        for attempt in attempts:
            final_attempts[attempt.result.trial_name] = max(
                attempt.attempt,
                final_attempts.get(attempt.result.trial_name, 0),
            )

        contextualized_attempts = []
        for attempt in attempts:
            task_path = attempt.result.config.task.path
            problem = problem_by_task_dir.get(task_path.resolve()) if task_path is not None else None
            contextualized_attempts.append(
                attempt.model_copy(
                    update={
                        "problem": problem,
                        "is_final": attempt.attempt == final_attempts[attempt.result.trial_name],
                    }
                )
            )

        return cls(
            run_id=run_id,
            release=release,
            attempts=tuple(contextualized_attempts),
            skills_loaded=tuple(skills_loaded or []),
        )


class AttemptMetadata(TelemetryModel):
    run_id: UUID
    release: str
    task_id: str | None
    trial_id: UUID
    trial_name: str
    attempt: int
    is_final_attempt: bool
    error_type: str | None
    agent_name: str
    agent_version: str
    model_name: str | None
    model_provider: str | None

    @classmethod
    def from_attempt(cls, run: TelemetryRun, attempt: TrialAttempt) -> Self:
        model_info = attempt.model_info
        return cls(
            run_id=run.run_id,
            release=run.release,
            task_id=attempt.problem.task_id if attempt.problem is not None else None,
            trial_id=attempt.result.id,
            trial_name=attempt.result.trial_name,
            attempt=attempt.attempt,
            is_final_attempt=attempt.is_final,
            error_type=attempt.error_type,
            agent_name=attempt.result.agent_info.name,
            agent_version=attempt.result.agent_info.version,
            model_name=model_info.name if model_info is not None else None,
            model_provider=model_info.provider if model_info is not None else None,
        )


class StepUsage(TelemetryModel):
    trajectory_id: str | None
    agent_name: str
    step_id: int
    model_name: str | None
    prompt_tokens: int | None
    cached_prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    cost_source: CostSource | None


class UsageRecord(AttemptMetadata):
    schema_version: Literal[2] = 2
    started_at: datetime | None
    duration_seconds: float | None
    prompt_tokens: int | None
    cached_prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    cost_source: CostSource | None
    token_usage_by_step: list[StepUsage] | None
    skills_loaded: list[str]
    tool_counts: dict[str, int] | None


class TraceManifestRecord(AttemptMetadata):
    schema_version: Literal[1] = 1
    trajectory_path: str | None
