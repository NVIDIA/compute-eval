import json
import os
from importlib import resources
from pathlib import Path

from compute_eval import EvaluatorRuntime
from compute_eval.data.metrics_data_model import PerformanceMetrics
from compute_eval.profilers import PerformanceProfiler


class LocalCUPTIProfiler(PerformanceProfiler):
    """CUPTI-based lightweight profiler using LD_PRELOAD."""

    def __init__(self, output_file: str = "profile.json"):
        self.output_file = output_file

    def profile(
        self,
        test_command: str,
        workdir_path: Path,
        timeout_seconds: float,
        execution_fn: EvaluatorRuntime,
    ) -> tuple[PerformanceMetrics | None, str | None]:
        """Run profiling using CUPTI LD_PRELOAD interceptor."""
        test_command = f"NVTX_INJECTION64_PATH=$CUPTI_PROFILER_LIB LD_PRELOAD=$CUPTI_PROFILER_LIB {test_command}"
        code, timed_out, output = execution_fn(test_command, timeout_seconds, "profile")

        if code != 0 or timed_out:
            return None, output

        return _parse_cupti_output(workdir_path / self.output_file), None

    @classmethod
    def from_config(
        cls,
        cupti_lib_path: Path | None = None,
        output_file: str = "profile.json",
        **kwargs,
    ) -> "LocalCUPTIProfiler":
        if cupti_lib_path is not None:
            os.environ["CUPTI_PROFILER_LIB"] = str(Path(cupti_lib_path).resolve())
        elif "CUPTI_PROFILER_LIB" not in os.environ:
            try:
                candidate = _get_packaged_cupti_sources_dir() / "libcuda_profile.so"
                if candidate.exists():
                    os.environ["CUPTI_PROFILER_LIB"] = str(candidate.resolve())
            except FileNotFoundError:
                pass
        return cls(output_file=output_file)


def _get_packaged_cupti_sources_dir() -> Path:
    """Return filesystem path to CUPTI profiler sources shipped with compute-eval."""
    cuda_tools = Path(str(resources.files("compute_eval"))).resolve() / "cuda_tools"
    if cuda_tools.exists():
        return cuda_tools
    raise FileNotFoundError(f"CUPTI profiler sources not found at {cuda_tools}")


def _parse_cupti_output(output_path: Path) -> PerformanceMetrics | None:
    if not output_path.exists():
        return None

    try:
        with open(output_path, "r") as f:
            data = json.load(f)

        return PerformanceMetrics.model_validate(data)

    except (OSError, json.JSONDecodeError, Exception) as e:
        print(f"Error parsing CUPTI output: {e}")
        return None
