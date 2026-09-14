# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute Sanitizer selection, execution, and fail-closed result handling."""

import hashlib
import json
import re
import secrets
import shlex
import time
from collections.abc import Sequence
from pathlib import Path

from compute_eval import EvaluatorRuntime
from compute_eval.data.data_model import (
    SanitizerResult,
    SanitizerStatus,
    SanitizerTool,
    SanitizerToolResult,
)

SANITIZER_TOOL_ORDER = (
    SanitizerTool.MEMCHECK,
    SanitizerTool.RACECHECK,
    SanitizerTool.SYNCCHECK,
    SanitizerTool.INITCHECK,
)
DEFAULT_SANITIZER_TIMEOUT_SECONDS = 300.0

_ERROR_SUMMARY_RE = re.compile(
    r"^=+\s*ERROR SUMMARY:\s*(\d+) errors?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_RACECHECK_SUMMARY_RE = re.compile(
    r"RACECHECK SUMMARY:.*?\((\d+) errors?,\s*(\d+) warnings?\)",
    re.IGNORECASE,
)

_COVERAGE_REPORT_ENV = "COMPUTE_EVAL_SANITIZER_REPORT"
_COVERAGE_TOKEN_ENV = "COMPUTE_EVAL_SANITIZER_TOKEN"


def parse_sanitizer_mode(mode: str | Sequence[str] | None) -> tuple[SanitizerTool, ...]:
    """Expand a sanitizer selector into deterministic execution order."""
    if mode is None:
        return ()

    raw_values = [mode] if isinstance(mode, str) else list(mode)
    names = [name.strip().lower() for value in raw_values for name in value.split(",") if name.strip()]

    if not names or names == ["none"]:
        return ()
    if "none" in names:
        raise ValueError("sanitizer mode 'none' cannot be combined with other tools")
    if "all" in names:
        if len(names) != 1:
            raise ValueError("sanitizer mode 'all' cannot be combined with explicit tools")
        return SANITIZER_TOOL_ORDER

    valid_names = {tool.value for tool in SANITIZER_TOOL_ORDER}
    unknown = sorted(set(names) - valid_names)
    if unknown:
        choices = ", ".join(["none", "all", *[tool.value for tool in SANITIZER_TOOL_ORDER]])
        raise ValueError(f"unknown sanitizer tool(s): {', '.join(unknown)}; choose from {choices}")

    selected = set(names)
    return tuple(tool for tool in SANITIZER_TOOL_ORDER if tool.value in selected)


def _sanitizer_command(
    tool: SanitizerTool,
    test_command: str,
    log_pattern: Path,
    coverage_report: Path,
    coverage_token: str,
) -> str:
    artifact_dir = shlex.quote(log_pattern.parent.as_posix())
    report_env = f"{_COVERAGE_REPORT_ENV}={shlex.quote(coverage_report.as_posix())}"
    token_env = f"{_COVERAGE_TOKEN_ENV}={shlex.quote(coverage_token)}"
    sanitizer = " ".join(
        [
            report_env,
            token_env,
            "compute-sanitizer",
            "--tool",
            tool.value,
            "--error-exitcode",
            "99",
            "--target-processes",
            "all",
            "--log-file",
            shlex.quote(log_pattern.as_posix()),
            "--",
            "bash",
            "-c",
            shlex.quote(test_command),
        ]
    )
    return f"mkdir -p {artifact_dir} && {sanitizer}"


def _error_count(tool: SanitizerTool, raw_log: str) -> int | None:
    if tool == SanitizerTool.RACECHECK:
        summaries = _RACECHECK_SUMMARY_RE.findall(raw_log)
        if not summaries:
            return None
        return sum(int(errors) + int(warnings) for errors, warnings in summaries)

    summaries = _ERROR_SUMMARY_RE.findall(raw_log)
    return sum(int(errors) for errors in summaries) if summaries else None


def _read_raw_logs(artifact_dir: Path, tool: SanitizerTool) -> tuple[str | None, str | None, str | None]:
    try:
        paths = sorted(artifact_dir.glob(f"{tool.value}.*.log"))
        if not paths:
            return None, None, "sanitizer did not produce a raw log"
        parts = [f"===== {path.name} =====\n{path.read_text(errors='replace')}" for path in paths]
        raw_log = "\n".join(parts)
        return raw_log, hashlib.sha256(raw_log.encode()).hexdigest(), None
    except OSError as error:
        return None, None, f"could not read sanitizer log: {error}"


def _read_coverage_report(report_path: Path, expected_token: str) -> tuple[bool, str | None]:
    try:
        data = json.loads(report_path.read_text())
    except FileNotFoundError:
        return False, "hidden test did not produce a target-coverage report"
    except (OSError, json.JSONDecodeError) as error:
        return False, f"target-coverage report is unreadable: {error}"

    if data.get("token") != expected_token:
        return False, "target-coverage report token does not match"
    target_launches = data.get("target_launches")
    if isinstance(target_launches, bool) or not isinstance(target_launches, int) or target_launches <= 0:
        return False, "target-coverage report does not contain a positive target_launches count"
    return True, None


def _classify_tool_result(
    tool: SanitizerTool,
    return_code: int,
    timed_out: bool,
    raw_log: str | None,
    raw_log_error: str | None,
    coverage_confirmed: bool,
    coverage_error: str | None,
) -> tuple[SanitizerStatus, str | None]:
    error_count = _error_count(tool, raw_log) if raw_log is not None else None
    if error_count is not None and error_count > 0:
        return SanitizerStatus.FAILED, f"sanitizer reported {error_count} error(s) or warning(s)"
    if timed_out:
        return SanitizerStatus.INVALID, "sanitizer run timed out"
    if raw_log_error is not None:
        return SanitizerStatus.INVALID, raw_log_error
    if error_count is None:
        return SanitizerStatus.INVALID, "sanitizer log has no recognized completion summary"
    if return_code != 0:
        return SanitizerStatus.INVALID, f"sanitized workload exited with code {return_code}"
    if not coverage_confirmed:
        return SanitizerStatus.INVALID, coverage_error or "target coverage was not confirmed"
    return SanitizerStatus.PASSED, None


def _preparation_failure(tools: Sequence[SanitizerTool], reason: str) -> SanitizerResult:
    return SanitizerResult(
        status=SanitizerStatus.INVALID,
        selected_tools=list(tools),
        tool_results=[],
        reason=reason,
    )


def run_compute_sanitizer(
    *,
    tools: Sequence[SanitizerTool],
    test_command: str,
    workdir: Path,
    timeout_seconds: float,
    exec_fn: EvaluatorRuntime,
) -> SanitizerResult:
    """Run selected tools and re-run the direct correctness test after clean results."""
    selected_tools = list(tools)
    artifact_dir = workdir / ".compute-eval-sanitizer"
    try:
        artifact_dir.mkdir(exist_ok=True)
    except OSError as error:
        return _preparation_failure(selected_tools, f"could not prepare sanitizer artifacts: {error}")

    tool_results: list[SanitizerToolResult] = []
    for tool in selected_tools:
        log_pattern = Path(".compute-eval-sanitizer") / f"{tool.value}.%p.log"
        coverage_token = secrets.token_hex(16)
        coverage_report = Path(".compute-eval-sanitizer") / f"coverage-{tool.value}-{secrets.token_hex(8)}.json"
        host_coverage_report = workdir / coverage_report
        command = _sanitizer_command(tool, test_command, log_pattern, coverage_report, coverage_token)

        try:
            for old_log in artifact_dir.glob(f"{tool.value}.*.log"):
                old_log.unlink()
            host_coverage_report.unlink(missing_ok=True)
        except OSError as error:
            tool_results.append(
                SanitizerToolResult(
                    tool=tool,
                    status=SanitizerStatus.INVALID,
                    command=command,
                    return_code=-1,
                    timed_out=False,
                    wall_seconds=0.0,
                    target_coverage_confirmed=False,
                    reason=f"could not prepare tool artifacts: {error}",
                    target_output="",
                )
            )
            continue

        started = time.monotonic()
        try:
            return_code, timed_out, target_output = exec_fn(command, timeout_seconds, f"sanitizer-{tool.value}")
        except Exception as error:
            return_code, timed_out, target_output = -1, False, f"[SANITIZER EXECUTION ERROR] {error}"
        wall_seconds = time.monotonic() - started

        raw_log, raw_log_sha256, raw_log_error = _read_raw_logs(artifact_dir, tool)
        coverage_confirmed, coverage_error = _read_coverage_report(host_coverage_report, coverage_token)
        status, reason = _classify_tool_result(
            tool,
            return_code,
            timed_out,
            raw_log,
            raw_log_error,
            coverage_confirmed,
            coverage_error,
        )
        tool_results.append(
            SanitizerToolResult(
                tool=tool,
                status=status,
                command=command,
                return_code=return_code,
                timed_out=timed_out,
                wall_seconds=wall_seconds,
                target_coverage_confirmed=coverage_confirmed,
                reason=reason,
                target_output=target_output,
                raw_log=raw_log,
                raw_log_sha256=raw_log_sha256,
            )
        )

    failed = next((result for result in tool_results if result.status == SanitizerStatus.FAILED), None)
    invalid = next((result for result in tool_results if result.status == SanitizerStatus.INVALID), None)
    if failed is not None:
        return SanitizerResult(
            status=SanitizerStatus.FAILED,
            selected_tools=selected_tools,
            tool_results=tool_results,
            reason=f"{failed.tool.value}: {failed.reason}",
        )
    if invalid is not None:
        return SanitizerResult(
            status=SanitizerStatus.INVALID,
            selected_tools=selected_tools,
            tool_results=tool_results,
            reason=f"{invalid.tool.value}: {invalid.reason}",
        )

    try:
        post_return_code, post_timed_out, post_output = exec_fn(
            test_command,
            timeout_seconds,
            "post-sanitizer-test",
        )
    except Exception as error:
        post_return_code, post_timed_out, post_output = -1, False, f"[POST-SANITIZER TEST ERROR] {error}"

    if post_timed_out:
        return SanitizerResult(
            status=SanitizerStatus.INVALID,
            selected_tools=selected_tools,
            tool_results=tool_results,
            reason="direct correctness re-check timed out",
            post_test_output=post_output,
            post_test_return_code=post_return_code,
            post_test_timed_out=True,
        )
    if post_return_code != 0:
        return SanitizerResult(
            status=SanitizerStatus.FAILED,
            selected_tools=selected_tools,
            tool_results=tool_results,
            reason=f"direct correctness re-check exited with code {post_return_code}",
            post_test_output=post_output,
            post_test_return_code=post_return_code,
            post_test_timed_out=False,
        )

    return SanitizerResult(
        status=SanitizerStatus.PASSED,
        selected_tools=selected_tools,
        tool_results=tool_results,
        post_test_output=post_output,
        post_test_return_code=post_return_code,
        post_test_timed_out=False,
    )
