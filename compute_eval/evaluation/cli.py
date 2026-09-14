import yaml
from pydantic import BaseModel, ConfigDict, Field

from compute_eval.data.data_model import ReleaseVersion
from compute_eval.evaluation import evaluate_functional_correctness


class EvaluateConfig(BaseModel):
    """Configuration for functional correctness evaluation."""

    model_config = ConfigDict(extra="forbid")

    release: ReleaseVersion = Field(
        default=ReleaseVersion.V2026_2,
        description="Release version to evaluate solutions for",
    )
    solutions_datapack: str = Field(
        default=...,
        description="Path to a directory holding solutions datapacks or a single solutions datapack to evaluate.",
    )
    problems_source: str | None = Field(
        default=None,
        description="Where to load problems from. None loads from HuggingFace (nvidia/compute-eval). A .tar.gz path opens that datapack directly. Any other path is treated as a directory of YAML problem specs.",
    )
    mode: str | None = Field(
        default=None,
        description="Evaluation execution mode. Must be set to 'podman' or 'local' to allow execution.",
    )
    profile_mode: str | None = Field(
        default=None,
        description="Profiling mode for performance analysis of problems that support it.  Can be 'cupti', 'ncu', or none.",
    )
    sanitizer_mode: str | list[str] = Field(
        default="none",
        description="Compute Sanitizer tools to run before benchmarking: none, all, or a comma-separated/list subset of memcheck, racecheck, synccheck, and initcheck.",
    )
    k: int | tuple[int, ...] = Field(
        default=1,
        description="K value(s) for pass@k evaluation",
    )
    n_workers: int = Field(
        default=4,
        description="Number of worker threads",
    )
    check_source_references: bool = Field(
        default=True,
        description="If False, skip source reference validation when grading solutions.",
    )
    skip_missing: bool = Field(
        default=False,
        description="If True, grade only problems that have a solution in the datapack. By default, a "
        "problem with no solution is graded as a failure (an empty solution). Use this when the datapack "
        "covers a sampled subset of its groups, so unattempted problems are not counted against pass@k.",
    )


def _build_config(
    config_file: str | None,
    model_class: type[BaseModel],
    cli_kwargs: dict,
) -> BaseModel:
    """Merge config file and CLI args, with CLI taking precedence."""
    if config_file:
        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file) or {}
    else:
        config_data = {}

    config_data.update({k: v for k, v in cli_kwargs.items() if v is not None})
    return model_class(**config_data)


def evaluate_functional_correctness_with_config(config_file: str | None = None, **cli_kwargs):
    config = _build_config(config_file, EvaluateConfig, cli_kwargs)
    evaluate_functional_correctness(**config.model_dump())


def main():
    import fire

    fire.Fire(
        {
            "evaluate_functional_correctness": evaluate_functional_correctness_with_config,
        }
    )


if __name__ == "__main__":
    main()
