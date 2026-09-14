import contextlib
import itertools
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

from compute_eval import EvaluatorRuntime, EvaluatorRuntimeResult
from compute_eval.data.data_model import Problem
from compute_eval.evaluation.evaluators import SolutionEvaluator
from compute_eval.evaluation.profilers import PerformanceProfiler
from compute_eval.utils.eval_utils import GpuInfo, parse_gpu_info, parse_semver

# Registry and image tag are controlled by environment variables so the same
# harness code can target different registries (local build, CI, NGC) without
# code changes.
#
# IMAGE_REGISTRY  – full registry prefix, e.g. "registry.example.com/my-org".
#                   Defaults to "localhost", which resolves to the local Podman
#                   image store — matching an image built with:
#                       podman build -f docker/Dockerfile.eval -t compute-eval:13.1.0 .
# CTK_VERSION     – image tag, matching the CUDA toolkit version baked into
#                   docker/Dockerfile.eval (default: "13.1.0").
_CTK_VERSION = os.getenv("CTK_VERSION", "13.1.0")
_IMAGE_REGISTRY = os.getenv("IMAGE_REGISTRY", "localhost")


class GpuPool:
    """Round-robin GPU assignment with per-GPU locking for concurrent evaluators."""

    def __init__(self, num_gpus: int):
        self._num_gpus = num_gpus
        self._locks = [threading.Lock() for _ in range(num_gpus)]
        self._counter = itertools.count()

    def assign(self) -> int:
        return next(self._counter) % self._num_gpus

    def acquire(self, gpu_id: int):
        self._locks[gpu_id].acquire()

    def release(self, gpu_id: int):
        self._locks[gpu_id].release()

    @contextlib.contextmanager
    def hold(self, gpu_id: int):
        self.acquire(gpu_id)
        try:
            yield
        finally:
            self.release(gpu_id)


_gpu_pool: GpuPool | None = None
_gpu_pool_lock = threading.Lock()
_pulled_images: set[str] = set()
_image_pull_lock = threading.Lock()

_gpu_info: GpuInfo | None = None
_gpu_info_lock = threading.Lock()


def _get_gpu_pool(gpu_info: GpuInfo) -> GpuPool:
    global _gpu_pool
    if _gpu_pool is not None:
        return _gpu_pool
    with _gpu_pool_lock:
        if _gpu_pool is None:
            _gpu_pool = GpuPool(gpu_info.num_gpus)
    return _gpu_pool


class PodmanEvaluator(SolutionEvaluator):
    """
    Evaluator that executes solutions in ephemeral Podman containers.

    Each command (build, test, benchmark) runs in its own short-lived container
    with --network=none, providing full network isolation across all phases.

    Works with both rootless and rootful Podman.  In rootless single-uid mode
    (no /etc/subuid entries) UID 0 inside the container maps to the host user,
    so files written to the bind-mounted workspace are owned by the launching
    user and can be cleaned up without elevation.

    Note: --pids-limit is intentionally omitted because cgroup-based limits are
    not available to unprivileged users in typical rootless configurations.
    """

    def __init__(
        self,
        base_image: str,
        cuda_toolkit_major: int,
        cuda_toolkit_minor: int,
        gpu_info: GpuInfo | None = None,
        profiler: PerformanceProfiler | None = None,
    ):
        super().__init__(
            ctk_major=cuda_toolkit_major,
            ctk_minor=cuda_toolkit_minor,
            gpu_info=gpu_info,
            base_dir=None,
            profiler=profiler,
        )
        self._profiler = profiler
        self.base_image = base_image
        self.gpu_pool = _get_gpu_pool(gpu_info)
        self.gpu_id = self.gpu_pool.assign()

    @classmethod
    def from_config(
        cls,
        problem: Problem,
        profiler: PerformanceProfiler | None,
    ) -> "PodmanEvaluator":
        """Create PodmanEvaluator from configuration.

        The evaluation image is ``<IMAGE_REGISTRY>/compute-eval:<CTK_VERSION>``
        by default (both controlled by environment variables). If
        ``problem.base_image`` is set it overrides the default image entirely.

        The CTK version parsed from ``CTK_VERSION`` is used for
        ``min_cuda_toolkit`` validation regardless of which image runs, on the
        assumption that an operator-supplied ``base_image`` satisfies the same
        toolkit requirement.
        """
        ctk_major, ctk_minor, ctk_patch = parse_semver(_CTK_VERSION) or (0, 0, 0)
        default_image = f"{_IMAGE_REGISTRY}/compute-eval:{_CTK_VERSION}"
        image = problem.base_image or default_image

        with _image_pull_lock:
            if image not in _pulled_images:
                _ensure_image(image)
                _pulled_images.add(image)

        return cls(
            base_image=image,
            profiler=profiler,
            cuda_toolkit_major=ctk_major,
            cuda_toolkit_minor=ctk_minor,
            gpu_info=_get_gpu_info(image),
        )

    def evaluator_runtime(self, workdir: Path) -> EvaluatorRuntime:
        def _exec_fn(cmd: str, to_seconds: float, tag: str | None) -> EvaluatorRuntimeResult:
            gpu_ctx = (
                self.gpu_pool.hold(self.gpu_id)
                if tag != "build" and self._profiler is not None
                else contextlib.nullcontext()
            )

            with gpu_ctx:
                ec, out, timed_out = _run_in_container(
                    image=self.base_image,
                    command=cmd,
                    workdir_host=workdir,
                    gpu_id=self.gpu_id,
                    network="none",
                    timeout_seconds=to_seconds,
                )
                return ec, timed_out, out

        return _exec_fn

    def teardown(self) -> None:
        # Each container is ephemeral — nothing to clean up here.
        pass


def _run_in_container(
    image: str,
    command: str,
    workdir_host: Path,
    gpu_id: int,
    network: str | None,
    timeout_seconds: float | None,
) -> tuple[int, str, bool]:
    """
    Run a shell command in an ephemeral Podman container.

    The container is started detached, the command is exec'd inside it, then the
    container is stopped and removed regardless of outcome.

    Args:
        image: Container image to use
        command: Shell command to execute
        workdir_host: Host path to bind-mount as /workspace
        gpu_id: GPU index to expose via CDI (nvidia.com/gpu=<id>)
        network: Network mode string, or None to use the default bridge
        timeout_seconds: Per-command timeout; None means no limit

    Returns:
        (exit_code, combined_output, timed_out)
    """
    container_id: str | None = None
    try:
        start_cmd = [
            "podman",
            "run",
            "--detach",
            "--tty",
            "--volume",
            f"{workdir_host.absolute()}:/workspace",
            "--workdir",
            "/workspace",
            # In rootless single-uid podman, UID 0 maps to the host user so files
            # created in /workspace are owned by the launching user for safe cleanup.
            "--user",
            "root",
            "--device",
            f"nvidia.com/gpu={gpu_id}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            # cgroup controllers (pids etc.) are not available when running inside
            # a Docker container or in rootless environments without delegation.
            "--cgroups=disabled",
        ]
        if network is not None:
            start_cmd += ["--network", network]
        start_cmd += [image, "bash"]

        result = subprocess.run(start_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return -1, f"[CONTAINER START ERROR: {result.stderr.strip()}]", False

        container_id = result.stdout.strip()
        ec, output, timed_out = _exec_with_timeout(container_id, command, timeout_seconds)
        return ec, output, timed_out

    except subprocess.TimeoutExpired:
        return -1, "[CONTAINER START TIMEOUT]", False
    except Exception as exc:
        return -1, f"[CONTAINER ERROR: {exc}]", False
    finally:
        if container_id:
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["podman", "stop", container_id, "--time", "1"],
                    capture_output=True,
                    timeout=10,
                )
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["podman", "rm", "-f", container_id],
                    capture_output=True,
                    timeout=10,
                )


def _exec_with_timeout(
    container_id: str,
    command: str,
    timeout_seconds: float | None,
) -> tuple[int, str, bool]:
    """
    Execute a command inside a running container with an optional timeout.

    Uses a background thread so that a blocking
    podman exec call can be interrupted by killing the container on timeout.

    Returns:
        (exit_code, combined_stdout_stderr, timed_out)
    """
    result_holder: dict[str, Any] = {}

    def run_command():
        try:
            proc = subprocess.Popen(
                ["podman", "exec", container_id, "bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate()
            result_holder["exit_code"] = proc.returncode
            result_holder["output"] = stdout.decode("utf-8") + "\n" + stderr.decode("utf-8")
        except Exception as exc:
            result_holder["exit_code"] = -1
            result_holder["output"] = f"[EXEC ERROR: {exc}]"

    thread = threading.Thread(target=run_command, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        with contextlib.suppress(Exception):
            subprocess.run(["podman", "kill", container_id], capture_output=True, timeout=5)
        thread.join(timeout=5)
        return -1, f"[TIMEOUT EXPIRED after {timeout_seconds} seconds]", True

    return (
        result_holder.get("exit_code", -1),
        result_holder.get("output", "[NO OUTPUT]"),
        False,
    )


def _ensure_image(image_name: str) -> None:
    """Pull the image if it is not already present locally."""
    check = subprocess.run(["podman", "image", "exists", image_name], capture_output=True)
    if check.returncode != 0:
        result = subprocess.run(
            ["podman", "pull", image_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull image '{image_name}': {result.stderr.strip()}")


def _get_gpu_info(image_name: str) -> GpuInfo | None:
    """
    Detect GPU count and compute capability.

    Tries nvidia-smi directly first (works when running inside a Docker container
    with --gpus all, where the toolkit injects nvidia-smi onto the host PATH).
    Falls back to running nvidia-smi inside a Podman container (the normal case
    when running natively on a GPU host where nvidia-smi isn't on the host PATH).

    Returns None if no GPUs are accessible.
    """
    global _gpu_info

    if _gpu_info is not None:
        return _gpu_info

    with _gpu_info_lock:
        if _gpu_info is not None:
            return _gpu_info

        # First try: direct nvidia-smi (works inside docker --gpus all)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                gpu_info = parse_gpu_info(result.stdout)
                if gpu_info is not None:
                    _gpu_info = gpu_info
                    return _gpu_info
        except Exception:
            pass

        # Second try: run nvidia-smi inside a Podman container via CDI
        try:
            result = subprocess.run(
                [
                    "podman",
                    "run",
                    "--rm",
                    "--device",
                    "nvidia.com/gpu=all",
                    "--cgroups=disabled",
                    image_name,
                    "nvidia-smi",
                    "--query-gpu=compute_cap,name",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                gpu_info = parse_gpu_info(result.stdout)
                if gpu_info is not None:
                    _gpu_info = gpu_info
        except Exception as e:
            print(f"[WARN] Failed to get GPU info from Podman container: {e}")

    return _gpu_info
