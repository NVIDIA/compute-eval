import subprocess
import threading
from pathlib import Path

from compute_eval import EvaluatorRuntime, EvaluatorRuntimeResult
from compute_eval.data.data_model import Problem
from compute_eval.evaluators import SolutionEvaluator
from compute_eval.profilers import PerformanceProfiler
from compute_eval.utils.eval_utils import GpuInfo, get_nvcc_version, parse_gpu_info, parse_semver

_gpu_info: GpuInfo | None = None
_gpu_info_lock = threading.Lock()


class LocalEvaluator(SolutionEvaluator):
    """
    Evaluator that executes solutions locally using subprocess.

    Assumes the local environment has all necessary dependencies (nvcc, cmake, etc.).
    """

    def __init__(
        self,
        ctk_major: int,
        ctk_minor: int,
        gpu_info: GpuInfo | None = None,
        profiler: PerformanceProfiler | None = None,
    ):
        super().__init__(
            ctk_major=ctk_major,
            ctk_minor=ctk_minor,
            gpu_info=gpu_info or _get_gpu_info_local(),
            base_dir=None,
            profiler=profiler,
        )

    @classmethod
    def from_config(
        cls,
        problem: Problem,
        profiler: PerformanceProfiler | None,
    ) -> "LocalEvaluator":
        ctk_version = get_nvcc_version()
        ctk_major, ctk_minor, _ = parse_semver(ctk_version) if ctk_version else (0, 0, 0)
        return cls(
            ctk_major=ctk_major,
            ctk_minor=ctk_minor,
            gpu_info=_get_gpu_info_local(),
            profiler=profiler,
        )

    def evaluator_runtime(self, workdir: Path) -> EvaluatorRuntime:
        # noinspection PyUnusedLocal
        def _exec_fn(command: str, timeout: float, tag: str | None) -> EvaluatorRuntimeResult:
            try:
                r = subprocess.run(
                    command,
                    shell=True,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                )
                return r.returncode, False, r.stdout + "\n" + r.stderr
            except subprocess.TimeoutExpired as e:
                return (-1, True, f"[TIMEOUT EXPIRED after {e.timeout} seconds]\n{e.stdout}\n{e.stderr}")
            except subprocess.CalledProcessError as e:
                return (e.returncode, False, f"[ERROR]\n{e.stdout}\n{e.stderr}")

        return _exec_fn

    def teardown(self) -> None:
        pass


def _get_gpu_info_local() -> GpuInfo | None:
    global _gpu_info

    with _gpu_info_lock:
        if _gpu_info is None:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                gpu_info = parse_gpu_info(result.stdout)
                if gpu_info is not None:
                    _gpu_info = gpu_info

            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                return None

    return _gpu_info
