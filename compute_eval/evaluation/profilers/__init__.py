from .base_profiler import PerformanceProfiler
from .cupti_profiler import LocalCUPTIProfiler
from .ncu_profiler import LocalNCUProfiler

__all__ = ["PerformanceProfiler", "create_profiler"]

_PROFILER_REGISTRY: dict[str, type[PerformanceProfiler]] = {
    "ncu": LocalNCUProfiler,
    "cupti": LocalCUPTIProfiler,
}


def create_profiler(mode: str | None, **kwargs) -> PerformanceProfiler | None:
    """
    Factory function to create the appropriate profiler based on mode.

    Args:
        mode: Profiling mode (e.g., 'ncu', 'cupti')
        **kwargs: Additional configuration parameters for the profiler

    Returns:
        Configured PerformanceProfiler instance

    Raises:
        ValueError: If mode is unknown
    """
    if mode is None:
        return None

    if mode not in _PROFILER_REGISTRY:
        raise ValueError(f"Unsupported profiling mode: '{mode}'.")

    profiler_class = _PROFILER_REGISTRY[mode]
    return profiler_class.from_config(**kwargs)
