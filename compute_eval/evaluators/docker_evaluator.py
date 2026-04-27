import contextlib
import itertools
import os
import threading
from pathlib import Path
from typing import Any

import docker
from compute_eval import EvaluatorRuntime, EvaluatorRuntimeResult
from compute_eval.data.data_model import Problem
from compute_eval.evaluators import SolutionEvaluator
from compute_eval.profilers import PerformanceProfiler
from compute_eval.utils.eval_utils import GpuInfo, get_cuda_image, parse_gpu_info
from docker.errors import APIError, ContainerError, ImageNotFound
from docker.types import DeviceRequest


class GpuPool:
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


class DockerEvaluator(SolutionEvaluator):
    """
    Evaluator that executes solutions in Docker containers.

    Provides isolation and consistent environments via containerization.
    """

    container = None

    def __init__(
        self,
        docker_client,
        base_image: str,
        cuda_toolkit_major: int,
        cuda_toolkit_minor: int,
        gpu_info: GpuInfo | None = None,
        profiler: PerformanceProfiler | None = None,
    ):
        """
        Initialize container evaluator.

        Args:
            docker_client: The docker client
            base_image: Container image to use for execution
            cuda_toolkit_major: Major version of installed CUDA toolkit
            cuda_toolkit_minor: Minor version of installed CUDA toolkit
            gpu_info: Optional GPU info for compute capability checks
            profiler: Performance profiler instance
        """
        super().__init__(
            ctk_major=cuda_toolkit_major,
            ctk_minor=cuda_toolkit_minor,
            gpu_info=gpu_info,
            base_dir=None,
            profiler=profiler,
        )
        self.docker_client = docker_client
        self._profiler = profiler
        self.cuda_toolkit_major = cuda_toolkit_major
        self.cuda_toolkit_minor = cuda_toolkit_minor
        self.base_image = base_image
        self.gpu_pool = _get_gpu_pool(gpu_info)
        self.gpu_id = self.gpu_pool.assign()

    @classmethod
    def from_config(
        cls,
        problem: Problem,
        profiler: PerformanceProfiler | None,
    ) -> "DockerEvaluator":
        """Create DockerEvaluator from configuration."""
        language = "cpp" if problem.type == "cuda_cpp" else "python"
        cuda_image = get_cuda_image(language)
        docker_client = docker.from_env()

        # Ensure that we only pull one image at a time in multi-threaded scenarios
        with _image_pull_lock:
            if cuda_image.image not in _pulled_images:
                # First check if image exists locally
                try:
                    docker_client.images.get(cuda_image.image)
                except docker.errors.ImageNotFound:
                    # Image not found locally, pull it
                    try:
                        docker_client.images.pull(cuda_image.image)
                    except Exception as e:
                        raise RuntimeError(f"Failed to pull image '{cuda_image.image}': {e}") from e
                except Exception as e:
                    raise RuntimeError(f"Failed to check for image '{cuda_image.image}': {e}") from e

                # Mark as pulled
                _pulled_images.add(cuda_image.image)

        return cls(
            docker_client=docker_client,
            base_image=cuda_image.image,
            profiler=profiler,
            cuda_toolkit_major=cuda_image.ctk_major,
            cuda_toolkit_minor=cuda_image.ctk_minor,
            gpu_info=_get_gpu_info(docker_client, cuda_image.image),
        )

    def evaluator_runtime(self, workdir: Path) -> EvaluatorRuntime:
        try:
            if self.container is None:
                # Start long-lived container with GPU access
                self.container = self.docker_client.containers.run(
                    image=self.base_image,
                    detach=True,
                    tty=True,
                    working_dir="/workspace",
                    # Volume mount (only workspace is accessible)
                    volumes={str(workdir.absolute()): {"bind": "/workspace", "mode": "rw"}},
                    # Run as host user so files created in /workspace are owned by the host UID/GID,
                    # allowing TemporaryDirectory cleanup to delete them without root.
                    user=f"{os.getuid()}:{os.getgid()}",
                    # GPU access
                    device_requests=[DeviceRequest(device_ids=[str(self.gpu_id)], capabilities=[["gpu"]])],
                    # Network isolation - enabled during only build
                    network_mode="none",
                    # Resource limits
                    # mem_limit="2g",  # 2GB RAM limit
                    # memswap_limit="2g",  # Disable swap (set equal to mem_limit)
                    # cpu_period=100000,  # CPU quota period (100ms)
                    # cpu_quota=200000,  # 2 CPU cores max (200% of period)
                    pids_limit=100,  # Limit number of processes (prevent fork bombs)
                    # Security options
                    security_opt=["no-new-privileges:true"],  # Prevent privilege escalation
                    cap_drop=["ALL"],  # Drop all capabilities
                    # Prevent container from gaining additional privileges
                    privileged=False,
                    auto_remove=False,
                )

            def _exec_fn(cmd: str, to_seconds: float, tag: str | None) -> EvaluatorRuntimeResult:
                try:
                    if tag == "build":
                        # Connect to the default bridge network during build
                        self.docker_client.api.disconnect_container_from_network(
                            container=self.container.id, net_id="none"
                        )
                        self.docker_client.api.connect_container_to_network(
                            container=self.container.id, net_id="bridge"
                        )

                    gpu_ctx = (
                        self.gpu_pool.hold(self.gpu_id)
                        if tag != "build" and self._profiler is not None
                        else contextlib.nullcontext()
                    )

                    with gpu_ctx:
                        ec, out, to_occurred = self._exec_with_timeout(
                            container=self.container,
                            command=cmd,
                            workdir="/workspace",
                            timeout_seconds=to_seconds,
                        )
                        return ec, to_occurred, out
                finally:
                    if tag == "build":
                        # After build, disconnect from the network to isolate further commands
                        self.docker_client.api.disconnect_container_from_network(
                            container=self.container.id, net_id="bridge"
                        )
                        self.docker_client.api.connect_container_to_network(container=self.container.id, net_id="none")

            return _exec_fn

        except ContainerError as e:
            raise RuntimeError("Docker container failed to start") from e
        except ImageNotFound as e:
            raise RuntimeError("Docker image not found") from e
        except APIError as e:
            raise RuntimeError("Docker API error") from e

    def teardown(self) -> None:
        if self.container is not None:
            # noinspection PyBroadException
            try:
                self.container.exec_run(["bash", "-c", "rm -rf /workspace/*"])
                self.container.stop(timeout=1)
                self.container.remove()
            except Exception:
                # We get an exception if the container was already killed due to timeout
                contextlib.suppress(Exception)
            self.container = None

    @staticmethod
    def _exec_with_timeout(
        container,
        command: str,
        workdir: str,
        timeout_seconds: float | None,
    ) -> tuple[int, str, bool]:
        """
        Execute a command in a container with optional timeout.

        Args:
            container: Docker container object
            command: Shell command to execute
            workdir: Working directory in container
            timeout_seconds: Timeout in seconds, or None for no timeout

        Returns:
            Tuple of (exit_code, output, timeout_occurred)
        """
        result_holder: dict[str, Any] = {}

        def run_command():
            try:
                code, result = container.exec_run(
                    ["bash", "-c", command],
                    workdir=workdir,
                    demux=True,
                )
                stdout = result[0].decode("utf-8") if result[0] else ""
                stderr = result[1].decode("utf-8") if result[1] else ""
                result_holder["exit_code"] = code
                result_holder["output"] = stdout + "\n" + stderr
            except Exception as exc:
                result_holder["exit_code"] = -1
                result_holder["output"] = f"[EXEC ERROR: {exc}]"

        thread = threading.Thread(target=run_command, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            # Timeout — kill the container to force the blocking exec_run to return
            with contextlib.suppress(Exception):
                container.kill()

            # Give the thread a chance to finish after the kill
            thread.join(timeout=5)

            return -1, f"[TIMEOUT EXPIRED after {timeout_seconds} seconds]", True

        return (
            result_holder.get("exit_code", -1),
            result_holder.get("output", "[NO OUTPUT]"),
            False,
        )


def _get_gpu_info(client, image_name) -> GpuInfo | None:
    """
    Get GPU information using nvidia-smi in a temporary container.

    Args:
        client: Docker client implementation
    Returns:
        GpuInfo object with GPU count, compute capability, and datacenter status.
    """

    global _gpu_info

    if _gpu_info is not None:
        return _gpu_info

    with _gpu_info_lock:
        if _gpu_info is not None:
            return _gpu_info

        """Try to get GPU info using nvidia-smi in a docker container."""
        # noinspection PyBroadException
        try:
            # Run nvidia-smi in a temporary container
            container = client.containers.run(
                image=image_name,
                command=["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
                device_requests=[DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
            )

            output = container.decode("utf-8") if isinstance(container, bytes) else container
            gpu_info = parse_gpu_info(output)

            if gpu_info is not None:
                _gpu_info = gpu_info

        except Exception as e:
            print(f"[WARN] Failed to get GPU info from Docker container: {e}")
            import traceback

            traceback.print_exc()
            return None

    return _gpu_info
