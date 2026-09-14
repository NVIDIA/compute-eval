"""Host-side GPU process scheduler for Harbor agent containers.

The scheduler keeps agent containers untouched: it does not wrap ``ncu`` or
install anything inside the task image. Instead, the host watches CUDA process
lifetimes and lets only one labeled Harbor container hold the GPU token at a
time. All other CUDA processes are held with ``cuda-checkpoint --action lock``.

This is intentionally strict when enabled. If the host cannot enumerate
containers, initialize NVML, or lock a competing CUDA process, the scheduler
raises instead of silently producing contaminated timing data.
"""

import asyncio
import contextlib
import fcntl
import logging
import os
import platform
import shutil
import subprocess
from collections.abc import Awaitable
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import pynvml

LOGGER = logging.getLogger(__name__)

GPU_TASK_LABEL = "com.nvidia.compute-eval.gpu-task"
# Map platform.machine() to the vendored binary directory. cuda-checkpoint ships
# a per-architecture ELF; both come from the same pinned upstream commit (see
# vendor/cuda_checkpoint/UPSTREAM.md).
_CUDA_CHECKPOINT_ARCH_DIRS = {"x86_64": "x86_64_linux", "aarch64": "aarch64_linux"}
_VENDOR_ROOT = Path(__file__).parent / "vendor" / "cuda_checkpoint"
CUDA_CHECKPOINT_PATH = (
    _VENDOR_ROOT / _CUDA_CHECKPOINT_ARCH_DIRS.get(platform.machine(), platform.machine()) / "cuda-checkpoint"
)
CONTAINER_CUDA_CHECKPOINT_PATH = "/tmp/compute-eval-cuda-checkpoint"
POLL_INTERVAL_SECONDS = 0.05
CHECKPOINT_LOCK_TIMEOUT_MS = 30_000
HOST_LOCK_PATH = Path("/tmp/compute-eval-gpu-scheduler.lock")


class GpuSchedulerError(RuntimeError):
    pass


def _run_command(command: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=f"{command[0]} not found")


class GpuProcessScheduler:
    def __init__(
        self,
        task_label_value: str,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    ):
        self._task_label_value = task_label_value
        self._poll_interval_seconds = poll_interval_seconds
        self._holder: str | None = None
        self._locked: dict[str, set[int]] = {}
        self._task: asyncio.Task | None = None
        self._stop_requested = asyncio.Event()
        self._host_lock_file: TextIOWrapper | None = None
        self._gpu_handles: list = []
        self._nvml_initialized = False

    async def start(self) -> None:
        await asyncio.to_thread(self._preflight)
        await asyncio.to_thread(self._acquire_host_lock)
        try:
            await asyncio.to_thread(self._init_nvml)
        except BaseException:
            self._shutdown_nvml()
            self._release_host_lock()
            raise
        self._stop_requested.clear()
        self._task = asyncio.create_task(self._run())
        LOGGER.info("GPU process scheduler started (poll %.2fs).", self._poll_interval_seconds)

    async def stop(self) -> None:
        task_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        if self._task is not None:
            self._stop_requested.set()
            try:
                await self._task
            except BaseException as error:
                task_error = error
            self._task = None
        try:
            await asyncio.to_thread(self._release)
        except BaseException as error:
            cleanup_error = error
        finally:
            self._shutdown_nvml()
            self._release_host_lock()
        if task_error is not None:
            if cleanup_error is not None:
                LOGGER.exception("[gpu-scheduler] cleanup also failed", exc_info=cleanup_error)
            raise task_error
        if cleanup_error is not None:
            raise cleanup_error

    async def run_guarded(self, awaitable: Awaitable[Any]) -> Any:
        """Run ``awaitable``, failing fast if the scheduler loop dies.

        Without this, a crashed loop would leave competing containers
        checkpoint-locked and the survivors unserialized until the job ended.
        """
        assert self._task is not None, "run_guarded requires start()"
        guarded = asyncio.ensure_future(awaitable)
        try:
            done, _ = await asyncio.wait({guarded, self._task}, return_when=asyncio.FIRST_COMPLETED)
            if self._task not in done:
                return await guarded
            if not guarded.done():
                guarded.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await guarded
            await self._task
            raise GpuSchedulerError("GPU scheduler stopped before the Harbor job completed.")
        except BaseException:
            if not guarded.done():
                guarded.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await guarded
            raise

    async def _run(self) -> None:
        while not self._stop_requested.is_set():
            await asyncio.to_thread(self._tick)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_requested.wait(), timeout=self._poll_interval_seconds)

    def _tick(self) -> None:
        task_containers = self._task_containers()
        cuda_pids = self._cuda_pids()
        pid_container = {pid: self._container_of(pid, task_containers) for pid in cuda_pids}
        self._prune_stale_locks(set(cuda_pids))
        cuda_containers = {container for container in pid_container.values() if container is not None}

        if self._holder is None:
            if cuda_containers:
                self._grant(sorted(cuda_containers)[0], cuda_pids, pid_container)
            return

        self._lock_others(cuda_pids, pid_container)
        if self._holder not in cuda_containers:
            LOGGER.info("GPU holder %s has no CUDA processes; handing off token.", self._holder)
            self._handoff(task_containers)

    def _grant(self, container: str, cuda_pids: list[int], pid_container: dict[int, str | None]) -> None:
        self._holder = container
        LOGGER.info("GPU token granted to container %s.", container)
        self._lock_others(cuda_pids, pid_container)

    def _lock_others(self, cuda_pids: list[int], pid_container: dict[int, str | None]) -> None:
        locked_pids = {pid for pids in self._locked.values() for pid in pids}
        for pid in cuda_pids:
            container = pid_container.get(pid)
            if container is None or container == self._holder or pid in locked_pids:
                continue
            if self._checkpoint(container, pid, "lock"):
                self._locked.setdefault(container, set()).add(pid)

    def _handoff(self, task_containers: set[str]) -> None:
        previous_holder = self._holder
        self._holder = None

        if not self._locked:
            LOGGER.info("GPU token released; no task containers are waiting.")
            return

        while self._locked:
            container = next(iter(self._locked))
            if container not in task_containers:
                self._unlock_container(container)
                continue
            self._unlock_container(container)
            self._holder = container
            LOGGER.info("GPU token handed off from %s to %s.", previous_holder, container)
            return

        LOGGER.info("GPU token released; no live task containers are waiting.")

    def _unlock_container(self, container: str, live_cuda_pids: set[int] | None = None) -> None:
        pids = self._locked[container]
        for pid in list(pids):
            if live_cuda_pids is not None and pid not in live_cuda_pids:
                LOGGER.info("Skipping cuda-checkpoint unlock for pid %s; CUDA process already exited.", pid)
            else:
                self._checkpoint(container, pid, "unlock")
            pids.remove(pid)
        self._locked.pop(container)

    def _prune_stale_locks(self, live_cuda_pids: set[int]) -> None:
        for container, pids in list(self._locked.items()):
            pids.intersection_update(live_cuda_pids)
            if not pids:
                self._locked.pop(container)

    def _release(self) -> None:
        live_cuda_pids = self._live_cuda_pids()
        had_locks = bool(self._locked)
        for container in list(self._locked):
            self._unlock_container(container, live_cuda_pids)
        if self._holder is not None or had_locks:
            LOGGER.info("GPU token released.")
        self._locked.clear()
        self._holder = None

    def _live_cuda_pids(self) -> set[int] | None:
        try:
            return set(self._cuda_pids())
        except GpuSchedulerError as error:
            LOGGER.warning("[gpu-scheduler] could not refresh CUDA pid list: %s", error)
            return None

    def _task_containers(self) -> set[str]:
        result = _run_command(
            ["docker", "ps", "-q", "--no-trunc", "--filter", f"label={GPU_TASK_LABEL}={self._task_label_value}"]
        )
        if result.returncode != 0:
            raise GpuSchedulerError(f"docker ps failed; cannot identify task containers: {result.stderr.strip()}")
        return {cid for cid in result.stdout.split() if cid}

    def _cuda_pids(self) -> list[int]:
        try:
            return sorted(
                {
                    process.pid
                    for handle in self._gpu_handles
                    for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                }
            )
        except pynvml.NVMLError as error:
            raise GpuSchedulerError(f"NVML process query failed: {error}") from error

    def _container_of(self, pid: int, task_containers: set[str]) -> str | None:
        try:
            cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
        except OSError:
            return None
        return next((cid for cid in task_containers if cid in cgroup), None)

    def _checkpoint(self, container: str, pid: int, action: str) -> bool:
        try:
            container_pid = self._container_pid(pid)
        except GpuSchedulerError:
            if not self._checkpoint_target_is_live(container, pid):
                LOGGER.info("Skipping cuda-checkpoint %s for pid %s; target already exited.", action, pid)
                return False
            raise
        command = [
            "docker",
            "exec",
            container,
            CONTAINER_CUDA_CHECKPOINT_PATH,
            "--action",
            action,
            "--pid",
            str(container_pid),
        ]
        if action == "lock":
            command += ["--timeout", str(CHECKPOINT_LOCK_TIMEOUT_MS)]
        result = _run_command(command)
        if result.returncode != 0:
            if not self._checkpoint_target_is_live(container, pid):
                LOGGER.info("Skipping cuda-checkpoint %s failure for pid %s; target already exited.", action, pid)
                return False
            details = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            message = f"cuda-checkpoint {action} failed for pid {pid}: {details}"
            LOGGER.error("[gpu-scheduler] %s", message)
            raise GpuSchedulerError(message)
        LOGGER.info(
            "cuda-checkpoint %s host pid %s (container pid %s).",
            action,
            pid,
            container_pid,
        )
        return True

    def _container_pid(self, pid: int) -> int:
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except OSError as error:
            raise GpuSchedulerError(f"Could not read namespace PID for CUDA process {pid}: {error}") from error
        for line in status.splitlines():
            if line.startswith("NSpid:"):
                namespace_pids = line.removeprefix("NSpid:").split()
                if namespace_pids:
                    return int(namespace_pids[-1])
        raise GpuSchedulerError(f"CUDA process {pid} has no NSpid entry in /proc/{pid}/status.")

    def _cuda_pid_is_live(self, pid: int) -> bool:
        # Re-queries on purpose: the pid list from the start of the tick is
        # stale by the time a checkpoint fails, and a fresh miss means the
        # process exited rather than the checkpoint being broken.
        live_cuda_pids = self._live_cuda_pids()
        if live_cuda_pids is None:
            return Path(f"/proc/{pid}").exists()
        return pid in live_cuda_pids

    def _checkpoint_target_is_live(self, container: str, pid: int) -> bool:
        return self._cuda_pid_is_live(pid) and container in self._task_containers()

    def _preflight(self) -> None:
        if platform.system() != "Linux" or platform.machine() not in _CUDA_CHECKPOINT_ARCH_DIRS:
            raise GpuSchedulerError(
                f"The bundled cuda-checkpoint binary requires Linux on one of "
                f"{sorted(_CUDA_CHECKPOINT_ARCH_DIRS)}; got {platform.system()}/{platform.machine()}."
            )
        if not CUDA_CHECKPOINT_PATH.is_file() or not os.access(CUDA_CHECKPOINT_PATH, os.X_OK):
            raise GpuSchedulerError(f"Bundled cuda-checkpoint is missing or not executable: {CUDA_CHECKPOINT_PATH}")
        if shutil.which("docker") is None:
            raise GpuSchedulerError("docker is not available.")
        self._task_containers()

    def _acquire_host_lock(self) -> None:
        lock_file = HOST_LOCK_PATH.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            raise GpuSchedulerError("Another GPU scheduler is already running on this host.") from error
        self._host_lock_file = lock_file

    def _release_host_lock(self) -> None:
        if self._host_lock_file is not None:
            fcntl.flock(self._host_lock_file.fileno(), fcntl.LOCK_UN)
            self._host_lock_file.close()
            self._host_lock_file = None

    def _init_nvml(self) -> None:
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            gpu_count = pynvml.nvmlDeviceGetCount()
            if gpu_count == 0:
                raise GpuSchedulerError("NVML found no GPUs.")
            self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]
        except pynvml.NVMLError as error:
            raise GpuSchedulerError(f"NVML unavailable; GPU scheduler cannot detect activity: {error}") from error

    def _shutdown_nvml(self) -> None:
        if self._nvml_initialized:
            with contextlib.suppress(pynvml.NVMLError):
                pynvml.nvmlShutdown()
            self._gpu_handles = []
            self._nvml_initialized = False
