import csv
import re
import subprocess
import threading
import time
from pathlib import Path
from string import Template

from compute_eval import EvaluatorRuntime
from compute_eval.data.metrics_data_model import (
    KernelMetrics,
    NvtxRangeMetrics,
    PerformanceMetrics,
    PerformanceSummary,
)
from compute_eval.profilers import PerformanceProfiler


class LocalNCUProfiler(PerformanceProfiler):
    @staticmethod
    def _check_ncu_available() -> bool:
        """Check if Nsight Compute (ncu) is available and usable."""
        if _ncu_perm_error_seen:
            return False
        try:
            result = subprocess.run(
                ["ncu", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def profile(
        self,
        test_command: str,
        workdir_path: Path,
        timeout_seconds: float,
        execution_fn: EvaluatorRuntime,
    ) -> tuple[PerformanceMetrics | None, str | None]:
        if not self._check_ncu_available():
            return None, "Nsight Compute (ncu) is not available or cannot run due to permissions."

        log_file = workdir_path / "ncu-results.csv"
        test_command = _NCU_TEMPLATE.substitute(test_command=test_command)
        # Coarse timing around ncu execution
        start_time = time.perf_counter_ns()
        code, timed_out, test_output = execution_fn(test_command, timeout_seconds, "profile")
        end_time = time.perf_counter_ns()

        # ERR_NVGPUCTRPERM is written to --log-file, not stdout/stderr
        try:
            if _check_and_log_ncu_permission_error(log_file.read_text(encoding="utf-8")):
                return None, "Permission error when running ncu. See logs for details."
        except OSError:
            pass

        if code != 0 or timed_out:
            return None, f"ncu execution failed with code {code} and timed_out={timed_out}. Output: {test_output}"

        try:
            return _parse_ncu_csv_output(log_file, (end_time - start_time) / 1e6), None
        except OSError as e:
            return None, f"Failed to read ncu output CSV: {e}"

    @classmethod
    def from_config(cls, **kwargs) -> "PerformanceProfiler":
        return cls()


_ncu_perm_error_seen = False
_ncu_perm_error_lock = threading.Lock()


def _check_and_log_ncu_permission_error(output: str) -> bool:
    """Check if output contains NCU permission error and log warning once."""
    global _ncu_perm_error_seen
    if "ERR_NVGPUCTRPERM" in output:
        with _ncu_perm_error_lock:
            if not _ncu_perm_error_seen:
                _ncu_perm_error_seen = True
                print("\n" + "=" * 80)
                print("[ERROR] Nsight Compute Permission Error")
                print("\nSee: https://developer.nvidia.com/ERR_NVGPUCTRPERM")
                print("=" * 80 + "\n")
        return True
    return False


_NCU_TEMPLATE = Template(
    """
ncu \
--nvtx \
--set none \
--clock-control boost \
--metrics \
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
--csv \
--log-file ncu-results.csv \
$test_command
"""
)


def _extract_nvtx_range(nvtx_column: str) -> str:
    """
    Extract NVTX range name from NCU's complex NVTX column.

    Format: "Domain:RangeName:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
    We want field 1 (RangeName) from the innermost (last) quoted section.

    Returns: RangeName or empty string if not found or generic.
    """
    # Find all quoted strings
    quoted_sections = re.findall(r'"([^"]+)"', nvtx_column)

    if not quoted_sections:
        return ""

    # Take the last (innermost) quoted section
    innermost = quoted_sections[-1]

    # Split by colons and take field 1 (RangeName)
    fields = innermost.split(":")
    if len(fields) >= 2:
        range_name = fields[1]  # Second field is the RangeName
        # Filter out generic/default names
        if range_name not in ("Main", "none", ""):
            return range_name

    return ""


_CSV_FIELD_NAMES = [
    "ID",
    "Process ID",
    "Process Name",
    "Host Name",
    "NVTX_Push",
    "NVTX_Range",
    "Kernel Name",
    "Context",
    "Stream",
    "Block Size",
    "Grid Size",
    "Device",
    "CC",
    "Section Name",
    "Metric Name",
    "Metric Unit",
    "Metric Value",
]
_CSV_METRICS = {
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
}


def _parse_ncu_csv_output(
    csv_file: Path,
    application_duration_ms: float = 0.0,
) -> PerformanceMetrics | None:
    """
    Parse NCU CSV output to extract per-kernel performance metrics with NVTX breakdown.

    NCU outputs one row per kernel invocation per metric. We aggregate these to compute
    totals and averages, grouped by kernel name and NVTX range.

    Args:
        csv_file: Path to NCU CSV output file
        application_duration_ms: Total application duration in milliseconds

    Returns:
        PerformanceMetrics with kernel data (no memory transfers or full NVTX ranges)
    """
    try:
        with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, fieldnames=_CSV_FIELD_NAMES)

            def _filter_row(row: dict[str, str]) -> bool:
                # Keep only rows that have one of the metrics we want AND have a Kernel Name
                m_name = row.get("Metric Name", "")
                k_name = row.get("Kernel Name", "")
                return k_name and m_name in _CSV_METRICS

            # Structure: {kernel_name: {nvtx_range: {metric_name: [values]}}}
            kernel_data: dict[str, dict[str, dict[str, list[float]]]] = {}

            for row in filter(_filter_row, reader):
                kernel_name = row.get("Kernel Name", "").strip()
                metric_name = row.get("Metric Name", "").strip()

                # Extract NVTX range (empty string if none)
                nvtx_range = _extract_nvtx_range(row.get("NVTX_Push", ""))

                # Parse metric value
                value_str = row.get("Metric Value", "").replace(",", "")
                try:
                    value = float(value_str)
                    # Initialize nested dicts as needed
                    if kernel_name not in kernel_data:
                        kernel_data[kernel_name] = {}
                    if nvtx_range not in kernel_data[kernel_name]:
                        kernel_data[kernel_name][nvtx_range] = {}
                    if metric_name not in kernel_data[kernel_name][nvtx_range]:
                        kernel_data[kernel_name][nvtx_range][metric_name] = []

                    kernel_data[kernel_name][nvtx_range][metric_name].append(value)
                except ValueError:
                    print(f"Warning: Could not parse value '{value_str}' for {metric_name}")

            if not kernel_data:
                return None

            # Convert to KernelMetrics objects, building nvtx_ranges in the same pass
            kernels = []
            nvtx_ranges = {}
            for kernel_name, nvtx_data in kernel_data.items():
                # Aggregate across all NVTX ranges for totals
                all_times_ns = []
                all_sm = []
                all_dram = []

                for nvtx_range, metrics in nvtx_data.items():
                    times_ns = metrics.get("gpu__time_duration.sum", [])
                    sm_throughputs = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", [])
                    dram_throughputs = metrics.get(
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", []
                    )

                    all_times_ns.extend(times_ns)
                    all_sm.extend(sm_throughputs)
                    all_dram.extend(dram_throughputs)

                    # Accumulate nvtx_ranges directly from the raw data
                    if nvtx_range:
                        times_ms_range = [t / 1_000_000 for t in times_ns]
                        if nvtx_range not in nvtx_ranges:
                            nvtx_ranges[nvtx_range] = NvtxRangeMetrics(
                                wall_clock_time_ms=None,
                                kernel_time_ms=0.0,
                                kernel_invocations=0,
                                memcpy_time_ms=0.0,
                                memcpy_count=0,
                                memcpy_bytes=0,
                            )
                        nvtx_ranges[nvtx_range].kernel_time_ms += sum(times_ms_range)
                        nvtx_ranges[nvtx_range].kernel_invocations += len(times_ns)

                # Convert nanoseconds to milliseconds
                times_ms = [t / 1_000_000 for t in all_times_ns]

                kernels.append(
                    KernelMetrics(
                        name=kernel_name,
                        invocations=len(all_times_ns),
                        total_duration_ms=sum(times_ms),
                        avg_duration_ms=sum(times_ms) / len(times_ms) if times_ms else 0.0,
                        sm_throughput_avg=sum(all_sm) / len(all_sm) if all_sm else None,
                        dram_throughput_avg=sum(all_dram) / len(all_dram) if all_dram else None,
                    )
                )

            # Create minimal summary
            total_kernel_time = sum(k.total_duration_ms for k in kernels)
            total_invocations = sum(k.invocations for k in kernels)

            summary = PerformanceSummary(
                total_kernel_time_ms=total_kernel_time,
                total_kernels=len(kernels),
                total_kernel_invocations=total_invocations,
                total_memory_transfers=0,
                application_duration_ms=application_duration_ms,
            )

            return PerformanceMetrics(
                kernels=kernels,
                memory_transfers={},
                nvtx_ranges=nvtx_ranges,
                summary=summary,
            )

    except Exception as e:
        print(f"Error parsing NCU output: {e}")
        return None
