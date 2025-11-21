import fire
import yaml
from pydantic import BaseModel, ConfigDict, Field

from compute_eval.data.data_model import ReleaseVersion
from compute_eval.evaluation import evaluate_functional_correctness
from compute_eval.generate_completions import generate_samples
from compute_eval.prompts import SYSTEM_PROMPT


class EvaluateConfig(BaseModel):
    """Configuration for functional correctness evaluation."""

    model_config = ConfigDict(extra="forbid")

    solutions_datapack: str = Field(
        default=...,
        description="Path to the solutions datapack",
    )
    problems_datapack_dir: str = Field(
        default="data/releases/",
        description="Directory where released problem datapacks are stored",
    )
    allow_execution: bool = Field(
        default=False,
        description="Whether to allow execution of untrusted code.  This must be set to True.",
    )
    k: int | tuple[int, ...] = Field(
        default=1,
        description="K value(s) for pass@k evaluation",
    )
    n_workers: int = Field(
        default=4,
        description="Number of worker threads",
    )
    results_file: str | None = Field(
        default=None,
        description="Path to output results file (defaults to {solutions_datapack's release}-graded-solutions.jsonl if not provided)",
    )


class GenerateConfig(BaseModel):
    """Configuration for solution generation."""

    model_config = ConfigDict(extra="forbid")

    release: ReleaseVersion = Field(
        default=ReleaseVersion.V2025_2,
        description="Release version to generate solutions for",
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
        description="Reasoning mode for the model (e.g., 'low', 'medium', 'high' for GPT models, or any value for Claude models to enable extended thinking)",
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
