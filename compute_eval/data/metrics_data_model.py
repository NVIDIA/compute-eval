import re
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class KernelMetrics(BaseModel):
    """Metrics for a single kernel."""

    name: str
    invocations: int
    total_duration_ms: float
    avg_duration_ms: float
    # NCU-specific aggregate metrics
    sm_throughput_avg: float | None = None
    dram_throughput_avg: float | None = None


class NvtxMemoryBreakdown(BaseModel):
    """Per-NVTX range breakdown for memory transfers."""

    bytes: int
    time_ms: float
    bandwidth_gbps: float


class MemoryTransferMetrics(BaseModel):
    """Metrics for a memory transfer direction (HtoD, DtoH, DtoD)."""

    count: int
    total_bytes: int
    total_time_ms: float
    bandwidth_gbps: float
    nvtx: dict[str, NvtxMemoryBreakdown] | None = None


class NvtxRangeMetrics(BaseModel):
    """Summary metrics for an NVTX range."""

    wall_clock_time_ms: float | None = None
    kernel_time_ms: float
    kernel_invocations: int
    memcpy_time_ms: float
    memcpy_count: int
    memcpy_bytes: int


class PerformanceSummary(BaseModel):
    """High-level summary statistics."""

    total_kernel_time_ms: float
    total_kernels: int
    total_kernel_invocations: int
    total_memory_transfers: int
    application_duration_ms: float


class PerformanceMetrics(BaseModel):
    """Complete performance profiling results - direct mapping from CUPTI JSON."""

    kernels: list[KernelMetrics]
    memory_transfers: dict[str, MemoryTransferMetrics]  # Keyed by kind: "HtoD", "DtoH", etc.
    nvtx_ranges: dict[str, NvtxRangeMetrics]  # Keyed by range name
    summary: PerformanceSummary

    def get_total_kernel_time(self) -> float | None:
        """Get total kernel execution time in milliseconds."""
        if self.summary is None:
            return None
        return self.summary.total_kernel_time_ms

    def get_average_sm_throughput(self) -> float | None:
        """Get average SM throughput across all kernels."""
        valid_throughputs = [k.sm_throughput_avg for k in self.kernels if k.sm_throughput_avg is not None]
        if not valid_throughputs:
            return None
        return sum(valid_throughputs) / len(valid_throughputs)

    def get_average_dram_throughput(self) -> float | None:
        """Get average DRAM throughput across all kernels."""
        valid_throughputs = [k.dram_throughput_avg for k in self.kernels if k.dram_throughput_avg is not None]
        if not valid_throughputs:
            return None
        return sum(valid_throughputs) / len(valid_throughputs)


class TimingMode(BaseModel, ABC):
    """Base class for performance score extraction strategies."""

    type: Literal["process", "kernels", "region", "gpu", "custom"]

    @abstractmethod
    def extract(self, metrics: PerformanceMetrics) -> float:
        """Extract performance score from metrics.

        Args:
            metrics: Performance metrics to extract score from

        Returns:
            Extracted performance score (e.g., total time in ms)
        """
        pass


class ProcessTimingMode(TimingMode):
    type: Literal["process"] = "process"

    def extract(self, metrics: PerformanceMetrics) -> float:
        return metrics.summary.application_duration_ms


class KernelsTimingMode(TimingMode):
    type: Literal["kernels"] = "kernels"
    include: list[str] | None = Field(
        default=None, description="Glob patterns for kernels to include (e.g., 'matmul_*', '*_optimized')"
    )
    exclude: list[str] | None = Field(
        default=None, description="Glob patterns for kernels to exclude (e.g., 'debug_*', 'test_*')"
    )

    def extract(self, metrics: PerformanceMetrics) -> float:
        return self._compute_kernel_time(metrics)

    def _compute_kernel_time(self, metrics: PerformanceMetrics) -> float:
        """Compute total kernel time based on include/exclude glob patterns."""
        # If no filters specified, use summary total
        if self.include is None and self.exclude is None:
            return metrics.summary.total_kernel_time_ms

        total = 0.0
        for kernel in metrics.kernels:
            # Check if kernel matches criteria
            if self._should_include(kernel.name):
                total += kernel.total_duration_ms

        return total

    def _should_include(self, name: str) -> bool:
        """Check if a kernel name should be included based on include/exclude patterns."""
        # If include patterns specified, must match at least one
        if self.include is not None:  # noqa: SIM102
            if not any(fnmatch(name, pattern) for pattern in self.include):
                return False

        # If exclude patterns specified, must not match any
        if self.exclude is not None:  # noqa: SIM102
            if any(fnmatch(name, pattern) for pattern in self.exclude):
                return False

        return True


class RegionTimingMode(TimingMode):
    type: Literal["region"] = "region"
    include: list[str] | None = Field(
        default=None, description="Glob patterns for NVTX ranges to include (e.g., 'PerfTest*', '*_benchmark')"
    )
    exclude: list[str] | None = Field(default=None, description="Glob patterns for NVTX ranges to exclude")
    time_type: Literal["kernel", "wall_clock"] = Field(
        default="kernel",
        description="Whether to extract kernel time or wall-clock time from NVTX ranges",
    )

    def extract(self, metrics: PerformanceMetrics) -> float:
        return self._compute_nvtx_time(metrics)

    def _compute_nvtx_time(self, metrics: PerformanceMetrics) -> float:
        """Compute total NVTX time based on include/exclude glob patterns and time_type."""
        # If no filters specified, sum all ranges
        if self.include is None and self.exclude is None:
            return sum(self._get_time_value(range_metrics) for range_metrics in metrics.nvtx_ranges.values())

        total = 0.0
        for range_name, range_metrics in metrics.nvtx_ranges.items():
            if self._should_include(range_name):
                total += self._get_time_value(range_metrics)

        return total

    def _should_include(self, name: str) -> bool:
        """Check if an NVTX range name should be included based on include/exclude patterns."""
        # If include patterns specified, must match at least one
        if self.include is not None:  # noqa: SIM102
            if not any(fnmatch(name, pattern) for pattern in self.include):
                return False

        # If exclude patterns specified, must not match any
        if self.exclude is not None:  # noqa: SIM102
            if any(fnmatch(name, pattern) for pattern in self.exclude):
                return False

        return True

    def _get_time_value(self, range_metrics: NvtxRangeMetrics) -> float:
        """Extract the appropriate time value based on time_type."""
        if self.time_type == "kernel":
            return range_metrics.kernel_time_ms
        else:  # wall_clock
            return range_metrics.wall_clock_time_ms or 0.0


class GpuTimingMode(TimingMode):
    type: Literal["gpu"] = "gpu"

    def extract(self, metrics: PerformanceMetrics) -> float:
        return self._compute_gpu_time(metrics)

    @staticmethod
    def _compute_gpu_time(metrics: PerformanceMetrics) -> float:
        kernel_time = metrics.summary.total_kernel_time_ms

        # Sum all memory transfer times
        memory_time = sum(transfer.total_time_ms for transfer in metrics.memory_transfers.values())

        return kernel_time + memory_time


class CustomTimingMode(TimingMode):
    """Self-reported timing mode where the benchmark program prints its own wall-clock time to STDOUT.

    The benchmark program must print a line matching:
        COMPUTE_EVAL_TIME_MS: <value>

    The value must be in milliseconds. If multiple matching lines are printed,
    the last one is used (to support warmup iterations).
    """

    type: Literal["custom"] = "custom"

    _PATTERN: re.Pattern = re.compile(r"^COMPUTE_EVAL_TIME_MS:\s*([\d.]+)\s*$", re.MULTILINE)

    def extract(self, metrics: PerformanceMetrics) -> float:
        raise ValueError(
            "CustomTimingMode does not extract from PerformanceMetrics. "
            "Use extract_from_output() to parse timing from benchmark STDOUT."
        )

    def extract_from_output(self, output: str) -> float | None:
        """Parse the self-reported timing value from benchmark STDOUT.

        Returns:
            The timing value in milliseconds, or None if no valid line was found.
        """
        matches = self._PATTERN.findall(output)
        if not matches:
            return None
        return float(matches[-1])


TimingModes = Annotated[
    ProcessTimingMode | KernelsTimingMode | RegionTimingMode | GpuTimingMode | CustomTimingMode,
    Field(discriminator="type"),
]
