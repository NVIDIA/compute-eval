import contextlib
import os
import queue
import signal
import subprocess
import threading
from pathlib import Path

from compute_eval import EvaluatorRuntime, EvaluatorRuntimeResult
from compute_eval.data.data_model import Problem
from compute_eval.evaluators import SolutionEvaluator
from compute_eval.profilers import PerformanceProfiler
from compute_eval.utils.eval_utils import GpuInfo, get_nvcc_version, parse_gpu_info, parse_semver


class GpuPool:
    """
    Dynamic GPU pool. Workers block on acquire() until a GPU is free, then hold
    it exclusively for the duration of GPU-bound work. This ensures all GPUs stay
    saturated regardless of how many workers are concurrently blocked on builds.
    """

    def __init__(self, num_gpus: int):
        self._available: queue.Queue[int] = queue.Queue()
        for i in range(num_gpus):
            self._available.put(i)

    @contextlib.contextmanager
    def acquire(self):
        gpu_id = self._available.get()
        try:
            yield gpu_id
        finally:
            self._available.put(gpu_id)


_gpu_info: GpuInfo | None = None
_gpu_info_lock = threading.Lock()

_gpu_pool: GpuPool | None = None
_gpu_pool_lock = threading.Lock()


def _get_gpu_pool(gpu_info: GpuInfo) -> GpuPool:
    global _gpu_pool
    if _gpu_pool is not None:
        return _gpu_pool
    with _gpu_pool_lock:
        if _gpu_pool is None:
            _gpu_pool = GpuPool(gpu_info.num_gpus)
    return _gpu_pool


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
        self._profiler = profiler

        self.gpu_pool = (
            _get_gpu_pool(self._gpu_info) if self._gpu_info is not None and self._gpu_info.num_gpus > 0 else None
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
        def _exec_fn(command: str, timeout: float, tag: str | None) -> EvaluatorRuntimeResult:
            env = os.environ.copy()

            # Build steps are CPU/IO-bound: run without holding a GPU slot so that
            # GPU-ready workers are never blocked waiting on compilation.
            # All other steps (test, benchmark) dynamically acquire an exclusive GPU.
            gpu_ctx = (
                self.gpu_pool.acquire()
                if tag != "build" and self.gpu_pool is not None
                else contextlib.nullcontext(None)
            )

            try:
                with gpu_ctx as gpu_id:
                    if gpu_id is not None:
                        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                    proc = subprocess.Popen(
                        command,
                        shell=True,
                        start_new_session=True,
                        cwd=workdir,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    try:
                        stdout, stderr = proc.communicate(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)  # Kill process group
                        proc.wait()  # Reap zombie
                        return -1, True, f"[TIMEOUT EXPIRED after {timeout} seconds]"

                    if proc.returncode != 0:
                        return proc.returncode, False, f"[ERROR]\n{stdout}\n{stderr}"
                    return proc.returncode, False, stdout + "\n" + stderr

            except Exception as e:
                return -1, False, f"[UNEXPECTED ERROR] {e}"

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
