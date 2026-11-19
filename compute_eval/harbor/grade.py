"""Grade one Harbor task's /workspace inside the verifier image.

tests/test.sh of an exported task (compute_eval.harbor.export) runs this. It grades as
`compute_eval evaluate --mode local` does and, with --profile-mode, times the solution and the
baseline as `--profile_mode` does. Timing never changes pass or fail.

Outputs in --out:
  reward.json            "reward": 1.0 for a pass, else 0.0 (a skipped problem too, so exporters
                         should leave those out). Also the evaluator's passed and skipped as
                         1.0 or 0.0 and, when timing ran, solution_time, baseline_time, speedup.
  graded_solution.json   the full GradedSolution

The submitted code runs during grading, as the grader's own user. So main() deletes problem.json,
which holds the baseline solution, before any of it runs.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import Field, TypeAdapter

from compute_eval.data.data_model import CudaCppProblem, CudaPythonProblem, GradedSolution, Problem
from compute_eval.evaluation.evaluators.local_evaluator import LocalEvaluator
from compute_eval.evaluation.profilers import create_profiler
from compute_eval.harbor.workspace import collect_solution

PROBLEM_ADAPTER = TypeAdapter(Annotated[CudaCppProblem | CudaPythonProblem, Field(discriminator="type")])
# GradedSolution fields that go into reward.json next to "reward". The timing ones are unset
# unless timing ran.
REWARD_FIELDS = {"passed", "skipped", "solution_time", "baseline_time", "speedup"}


def without_deleted_context_files(problem: Problem, workspace: Path) -> Problem:
    """The problem without the context files the agent deleted.

    The evaluator rebuilds the workspace from the problem's context files, so a file the agent
    deleted would come back and the graded workspace would not be the submitted one.
    """
    kept = [
        source for source in problem.context_files if workspace.joinpath(*PurePosixPath(source.path).parts).is_file()
    ]
    return problem.model_copy(update={"context_files": kept})


def grade(problem: Problem, workspace: Path, profile_mode: str | None = None) -> GradedSolution:
    solution = collect_solution(problem, workspace)
    problem = without_deleted_context_files(problem, workspace)
    # With a profiler the evaluator also times the baseline and fills in the speedup.
    evaluator = LocalEvaluator.from_config(problem=problem, profiler=create_profiler(profile_mode))
    return evaluator.evaluate_solutions(problem, [solution], check_source_references=True)[0]


def reward_fields(graded: GradedSolution) -> dict[str, float]:
    fields = graded.model_dump(include=REWARD_FIELDS, exclude_none=True)
    # Harbor takes numbers only, so the two flags become 1.0 or 0.0.
    return {"reward": float(graded.passed and not graded.skipped)} | {k: float(v) for k, v in fields.items()}


def write_results(graded: GradedSolution, out: Path) -> dict[str, float]:
    out.mkdir(parents=True, exist_ok=True)
    # serialize_as_any keeps the concrete solution's files; the field is typed as the Solution base.
    (out / "graded_solution.json").write_text(graded.model_dump_json(indent=2, serialize_as_any=True), encoding="utf-8")
    fields = reward_fields(graded)
    # Written last: a reward file means grading finished.
    (out / "reward.json").write_text(json.dumps(fields), encoding="utf-8")
    return fields


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Grade a Harbor task workspace with compute-eval's evaluator.")
    parser.add_argument("--problem", type=Path, required=True, help="The task's problem.json; deleted once read.")
    parser.add_argument("--workspace", type=Path, required=True, help="What the agent left in its workdir.")
    parser.add_argument("--out", type=Path, required=True, help="Folder for reward.json and graded_solution.json.")
    parser.add_argument("--profile-mode", help="Time the solution and the baseline with this profiler, such as cupti.")
    args = parser.parse_args(argv)

    problem = PROBLEM_ADAPTER.validate_json(args.problem.read_text(encoding="utf-8"))
    # The submitted code runs next and could read the baseline solution from this file. If it cannot
    # be deleted, this raises and no reward is written, rather than grading with the answers in reach.
    args.problem.unlink()
    graded = grade(problem, args.workspace, args.profile_mode)
    fields = write_results(graded, args.out)

    # Shown in the verifier's stdout log.
    print(json.dumps(fields))


if __name__ == "__main__":
    main()
