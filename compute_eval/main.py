from typing import get_args

import fire
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from compute_eval.data.data_model import ReleaseVersion, ValidGroup
from compute_eval.evaluation import evaluate_functional_correctness
from compute_eval.generate_completions import generate_samples
from compute_eval.prompts import SYSTEM_PROMPT

VALID_GROUP_CHOICES = list(get_args(ValidGroup))


class EvaluateConfig(BaseModel):
    """Configuration for functional correctness evaluation."""

    model_config = ConfigDict(extra="forbid")

    release: ReleaseVersion = Field(
        default=ReleaseVersion.V2026_1,
        description="Release version to evaluate solutions for",
    )
    solutions_datapack: str = Field(
        default=...,
        description="Path to a directory holding solutions datapacks or a single solutions datapack to evaluate.",
    )
    problems_datapack_dir: str = Field(
        default="data/releases/",
        description="Directory where released problem datapacks are stored",
    )
    mode: str | None = Field(
        default=None,
        description="Evaluation execution mode. Must be set to 'docker' or 'local' to allow execution.",
    )
    profile_mode: str | None = Field(
        default=None,
        description="Profiling mode for performance analysis of problems that support it.  Can be 'cupti', 'ncu', or none.",
    )
    k: int | tuple[int, ...] = Field(
        default=1,
        description="K value(s) for pass@k evaluation",
    )
    n_workers: int = Field(
        default=4,
        description="Number of worker threads",
    )


class GenerateConfig(BaseModel):
    """Configuration for solution generation."""

    model_config = ConfigDict(extra="forbid")

    release: ReleaseVersion = Field(
        default=ReleaseVersion.V2026_1,
        description="Release version to generate solutions for",
    )
    include: list[ValidGroup] | None = Field(
        default=None,
        description=f"Comma separated list of groups to include when generating solutions. Valid values: {VALID_GROUP_CHOICES}. Mutually exclusive with 'exclude'. If not set, all groups are included.",
    )
    exclude: list[ValidGroup] | None = Field(
        default=None,
        description=f"Comma separated list of groups to exclude when generating solutions. Valid values: {VALID_GROUP_CHOICES}. Mutually exclusive with 'include'. If not set, no groups are excluded.",
    )
    problems_datapack_dir: str = Field(
        default="data/releases/",
        description="Directory where released problem datapacks are stored",
    )
    solutions_per_problem: int = Field(
        default=1,
        description="Number of solutions to generate per problem",
    )
    n_workers: int = Field(
        default=10,
        description="Number of worker threads",
    )
    system_prompt: str = Field(
        default=SYSTEM_PROMPT,
        description="System prompt for the model",
    )
    model: str = Field(
        default=...,
        description="Model to use for generation",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for the model API",
    )
    reasoning: str | None = Field(
        default=None,
        description="Reasoning mode for the model (e.g., 'low', 'medium', 'high'). Mutually exclusive with 'thinking'.",
    )
    thinking: bool | None = Field(
        default=None,
        description="Whether to enable thinking for the model (if supported). Mutually exclusive with 'reasoning'.",
    )
    temperature: float | None = Field(
        default=None,
        description="Temperature for generation",
    )
    top_p: float | None = Field(
        default=None,
        description="Top-p for generation",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens for generation",
    )
    temp_dir: str | None = Field(
        default=None,
        description="Temporary directory for intermediate results",
    )
    debug: bool = Field(
        default=False,
        description="Include system prompt, prompt, and completion in the output solutions file for debugging",
    )

    @field_validator("include", "exclude", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        return _process_list_field(cls, v)


def _process_list_field(cls, v):
    if v is None:
        return None
    elif isinstance(v, list):
        return v
    elif isinstance(v, str):
        return [item.strip() for item in v.split(",") if item.strip()]
    else:
        raise ValueError(f"Invalid value for list field: {v}")


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


def generate_samples_with_config(config_file: str | None = None, **cli_kwargs):
    config = _build_config(config_file, GenerateConfig, cli_kwargs)
    generate_samples(**config.model_dump())


def evaluate_functional_correctness_with_config(config_file: str | None = None, **cli_kwargs):
    config = _build_config(config_file, EvaluateConfig, cli_kwargs)
    evaluate_functional_correctness(**config.model_dump())


def main():
    fire.Fire(
        {
            "evaluate_functional_correctness": evaluate_functional_correctness_with_config,
            "generate_samples": generate_samples_with_config,
        }
    )


if __name__ == "__main__":
    main()
