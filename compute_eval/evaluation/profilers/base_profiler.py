from abc import ABC, abstractmethod
from pathlib import Path

from compute_eval import EvaluatorRuntime
from compute_eval.data.metrics_data_model import PerformanceMetrics


class PerformanceProfiler(ABC):
    """Abstract base class for performance profiling."""

    @abstractmethod
    def profile(
        self,
        test_command: str,
        workdir_path: Path,
        timeout_seconds: float,
        execution_fn: EvaluatorRuntime,
    ) -> tuple[PerformanceMetrics | None, str | None]:
        """
        Run the test command with profiling instrumentation.

        Args:
            test_command: The command to execute and profile
            workdir_path: Working directory for execution
            timeout_seconds: Maximum execution time
            execution_fn: Function to execute commands. Should match EvaluatorRuntime signature.

        Returns:
            Tuple of (PerformanceMetrics if successful, error message if failed)
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, **kwargs) -> "PerformanceProfiler":
        """
        Create a profiler instance from configuration.

        Args:
            **kwargs: Configuration parameters specific to the profiler

        Returns:
            Configured PerformanceProfiler instance
        """
        pass
