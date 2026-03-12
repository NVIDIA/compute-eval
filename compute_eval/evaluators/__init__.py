from compute_eval.data.data_model import Problem

from .base import SolutionEvaluator
from .docker_evaluator import DockerEvaluator
from .local_evaluator import LocalEvaluator

__all__ = ["SolutionEvaluator", "create_evaluator"]

from ..profilers import create_profiler

_EVALUATOR_REGISTRY: dict[str, type[SolutionEvaluator]] = {
    "local": LocalEvaluator,
    "docker": DockerEvaluator,
}

# Include internal evaluators if available
try:
    # noinspection PyUnusedImports
    from compute_eval.evaluators.internal import INTERNAL_EVALUATORS

    _EVALUATOR_REGISTRY.update(INTERNAL_EVALUATORS)
except ImportError:
    pass


def create_evaluator(
    problem: Problem,
    eval_mode: str,
    profile_mode: str | None,
) -> SolutionEvaluator:
    """
    Factory function to create the appropriate evaluator based on mode.

    Handles all setup logic including version parsing, GPU detection, and
    Docker image preparation.

    Args:
        problem: The problem specification
        eval_mode: Evaluation mode (e.g., 'local', 'docker')
        profile_mode: Optional profiling mode for performance analysis of problems that support it

    Returns:
        Configured SolutionEvaluator instance

    Raises:
        ValueError: If cuda_toolkit version is invalid or mode is unknown
        RuntimeError: If no GPUs are available
    """
    if eval_mode not in _EVALUATOR_REGISTRY:
        raise ValueError(f"Unsupported evaluation mode: '{eval_mode}'.")

    evaluator_class = _EVALUATOR_REGISTRY[eval_mode]
    profiler = create_profiler(profile_mode)

    return evaluator_class.from_config(problem=problem, profiler=profiler)
