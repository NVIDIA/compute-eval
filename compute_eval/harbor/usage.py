import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import litellm
from harbor.models.trajectories.tool_call import ToolCall
from harbor.models.trajectories.trajectory import Trajectory

from compute_eval.harbor.telemetry import (
    AttemptMetadata,
    CostSource,
    GenerationArtifacts,
    StepUsage,
    TelemetryRun,
    TrialAttempt,
    UsageRecord,
)

LOGGER = logging.getLogger(__name__)


def _tool_calls(trajectory: Trajectory) -> list[ToolCall]:
    calls = [call for step in trajectory.steps for call in (step.tool_calls or [])]
    for subagent in trajectory.subagent_trajectories or []:
        calls.extend(_tool_calls(subagent))
    return calls


def _tool_name(call: ToolCall) -> str:
    normalized = call.function_name.lower()
    if normalized.startswith("skill:"):
        return f"skill:{call.function_name.split(':', 1)[1]}"
    if normalized not in {"skill", "skills"}:
        return normalized
    for key in ("skill", "skill_name", "name"):
        value = call.arguments.get(key)
        if isinstance(value, str) and value:
            return f"skill:{value}"
    return "skill:unknown"


def _tool_counts(trajectory: Trajectory | None) -> dict[str, int] | None:
    if trajectory is None:
        return None

    tool_counts: Counter[str] = Counter()
    for call in _tool_calls(trajectory):
        tool_counts[_tool_name(call)] += 1
    return dict(sorted(tool_counts.items()))


def _model_candidates(model_name: str) -> list[str]:
    leaf = model_name.rsplit("/", 1)[-1]
    candidates = [model_name, model_name.split("/", 1)[-1], leaf]
    if leaf.startswith("bedrock-"):
        candidates.append(leaf.removeprefix("bedrock-"))
    return list(dict.fromkeys(candidates))


def _model_pricing(model_name: str) -> dict[str, float] | None:
    for candidate in _model_candidates(model_name):
        pricing = litellm.model_cost.get(candidate)
        if pricing:
            return pricing
    return None


def _estimate_cost(
    model_name: str | None,
    prompt_tokens: int | None,
    cached_prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    if model_name is None or (prompt_tokens is None and completion_tokens is None):
        return None

    pricing = _model_pricing(model_name)
    if pricing is None:
        return None

    input_rate = pricing.get("input_cost_per_token")
    output_rate = pricing.get("output_cost_per_token")
    cached_input_rate = pricing.get("cache_read_input_token_cost")
    if cached_input_rate is None:
        cached_input_rate = input_rate

    cached_tokens = cached_prompt_tokens or 0
    uncached_tokens = max(0, (prompt_tokens or 0) - cached_tokens)
    output_tokens = completion_tokens or 0
    if (uncached_tokens and input_rate is None) or (cached_tokens and cached_input_rate is None):
        return None
    if output_tokens and output_rate is None:
        return None

    return (
        uncached_tokens * (input_rate or 0.0)
        + cached_tokens * (cached_input_rate or 0.0)
        + output_tokens * (output_rate or 0.0)
    )


def _cost_with_source(
    reported_cost: float | None,
    model_name: str | None,
    prompt_tokens: int | None,
    cached_prompt_tokens: int | None,
    completion_tokens: int | None,
) -> tuple[float | None, CostSource | None]:
    if reported_cost is not None:
        return reported_cost, "reported"
    estimated_cost = _estimate_cost(model_name, prompt_tokens, cached_prompt_tokens, completion_tokens)
    if estimated_cost is not None:
        return estimated_cost, "litellm"
    return None, None


def _token_usage_by_step(
    trajectory: Trajectory | None,
    default_model_name: str | None,
) -> list[StepUsage] | None:
    if trajectory is None:
        return None

    usage: list[StepUsage] = []
    model_name = trajectory.agent.model_name or default_model_name
    for step in trajectory.steps:
        if step.metrics is None:
            continue
        step_model_name = step.model_name or model_name
        cost_usd, cost_source = _cost_with_source(
            step.metrics.cost_usd,
            step_model_name,
            step.metrics.prompt_tokens,
            step.metrics.cached_tokens,
            step.metrics.completion_tokens,
        )
        usage.append(
            StepUsage(
                trajectory_id=trajectory.trajectory_id,
                agent_name=trajectory.agent.name,
                step_id=step.step_id,
                model_name=step_model_name,
                prompt_tokens=step.metrics.prompt_tokens,
                cached_prompt_tokens=step.metrics.cached_tokens,
                completion_tokens=step.metrics.completion_tokens,
                cost_usd=cost_usd,
                cost_source=cost_source,
            )
        )
    for subagent in trajectory.subagent_trajectories or []:
        subagent_usage = _token_usage_by_step(subagent, model_name)
        if subagent_usage is not None:
            usage.extend(subagent_usage)
    return usage


def _duration_seconds(started_at: datetime | None, finished_at: datetime | None) -> float | None:
    if started_at is None or finished_at is None:
        return None
    return (finished_at - started_at).total_seconds()


def _trial_usage_record(
    run: TelemetryRun,
    attempt: TrialAttempt,
) -> UsageRecord | None:
    trial_result = attempt.result
    task_path = trial_result.config.task.path
    if task_path is None:
        LOGGER.warning("Skipping usage for trial %s: no local task path.", trial_result.trial_name)
        return None
    if attempt.problem is None:
        LOGGER.warning("Skipping usage for trial %s: task path %s not in dataset.", trial_result.trial_name, task_path)
        return None

    prompt_tokens, cached_prompt_tokens, completion_tokens, cost_usd = trial_result.compute_token_cost_totals()
    model_info = trial_result.agent_info.model_info
    model_name = model_info.name if model_info is not None else None
    cost_usd, cost_source = _cost_with_source(
        cost_usd,
        model_name,
        prompt_tokens,
        cached_prompt_tokens,
        completion_tokens,
    )
    metadata = AttemptMetadata.from_attempt(run, attempt)
    return UsageRecord(
        **metadata.model_dump(),
        started_at=trial_result.started_at,
        duration_seconds=_duration_seconds(trial_result.started_at, trial_result.finished_at),
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        cost_source=cost_source,
        token_usage_by_step=_token_usage_by_step(attempt.trajectory, model_name),
        skills_loaded=list(run.skills_loaded),
        tool_counts=_tool_counts(attempt.trajectory),
    )


def write_usage_report(
    artifacts: GenerationArtifacts,
    run: TelemetryRun | None,
    previous_report: Path | None = None,
) -> Path:
    output_path = artifacts.usage
    output_path.parent.mkdir(parents=True, exist_ok=True)

    previous_records = ""
    if previous_report is not None:
        if previous_report.exists():
            previous_records = previous_report.read_text(encoding="utf-8")
        else:
            LOGGER.warning(
                "No prior Harbor usage report found at %s; resumed solutions will have no usage rows.", previous_report
            )

    with output_path.open("w", encoding="utf-8") as handle:
        if previous_records:
            handle.write(previous_records)
            if not previous_records.endswith("\n"):
                handle.write("\n")
        if run is not None:
            for attempt in run.attempts:
                try:
                    record = _trial_usage_record(run, attempt)
                except Exception:
                    LOGGER.exception("Skipping usage for trial %s due to an error.", attempt.result.trial_name)
                    continue
                if record is not None:
                    handle.write(record.model_dump_json() + "\n")
    return output_path
