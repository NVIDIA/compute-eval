"""Build the CUPTI profiler shared library (``libcuda_profile.so``) in place.

This is the pip-install counterpart to ``compute_eval/cuda_tools/Makefile``
(which is what the podman images use). It locates the profiler sources that
ship inside the installed ``compute_eval`` package, compiles them next to those
sources, and drops ``libcuda_profile.so`` exactly where the runtime profiler
looks for it by default (see ``cupti_profiler._get_packaged_cupti_sources_dir``).

Usage after ``pip install compute-eval[eval]``::

    compute_eval_build_profiler

Environment variables (all optional, matching the Makefile):
    CXX           - C++ compiler          (default: g++)
    CUDA_HOME     - CUDA toolkit root     (default: /usr/local/cuda)
    CUPTI_INCLUDE - CUPTI header dir      (auto-detected from CUDA_HOME)
    CUPTI_LIB     - CUPTI library dir     (auto-detected from CUDA_HOME)
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Keep in sync with the SOURCES list in cuda_tools/Makefile.
_SOURCES = (
    "libcuda_profile.cpp",
    "nvtx_interpose.cpp",
    "profiler_state.cpp",
    "nvtx_correlation.cpp",
    "activity_processing.cpp",
    "json_output.cpp",
    "profiler_lifecycle.cpp",
)
_OUTPUT = "libcuda_profile.so"


class ProfilerBuildError(RuntimeError):
    """Raised when the CUPTI profiler library cannot be built."""


def _sources_dir() -> Path:
    """Directory holding the packaged CUPTI profiler sources."""
    return Path(__file__).resolve().parent


def _resolve_cupti_paths() -> tuple[Path, Path]:
    """Resolve (include_dir, lib_dir) for CUPTI, mirroring the Makefile fallbacks."""
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    machine = platform.machine().strip().lower()
    target = "sbsa-linux" if machine in {"arm64", "aarch64"} else "x86_64-linux"

    include_dir = Path(os.environ["CUPTI_INCLUDE"]) if "CUPTI_INCLUDE" in os.environ else None
    if include_dir is None:
        for candidate in (cuda_home / "targets" / target / "include", cuda_home / "include"):
            if (candidate / "cupti.h").exists():
                include_dir = candidate
                break

    lib_dir = Path(os.environ["CUPTI_LIB"]) if "CUPTI_LIB" in os.environ else None
    if lib_dir is None:
        for candidate in (cuda_home / "lib64", cuda_home / "targets" / target / "lib"):
            if candidate.exists():
                lib_dir = candidate
                break

    if include_dir is None or not include_dir.exists():
        raise ProfilerBuildError(
            "Could not find the CUPTI headers (cupti.h). Install the CUDA toolkit "
            "with CUPTI, set CUDA_HOME (or CUPTI_INCLUDE) to point at it, or run the "
            "evaluation with `--mode podman` (the container builds the profiler for you)."
        )
    if lib_dir is None or not lib_dir.exists():
        raise ProfilerBuildError(
            "Could not find the CUDA/CUPTI library directory. Set CUDA_HOME (or "
            "CUPTI_LIB) to point at your CUDA toolkit, or run with `--mode podman`."
        )
    return include_dir, lib_dir


def build_profiler() -> Path:
    """Compile ``libcuda_profile.so`` in place next to the packaged sources.

    Returns the path to the built library. Raises ProfilerBuildError on failure.
    """
    source_dir = _sources_dir()

    compiler = os.environ.get("CXX", "g++")
    if shutil.which(compiler) is None:
        raise ProfilerBuildError(
            f"C++ compiler '{compiler}' not found on PATH. Install g++ (or set CXX), "
            "or run the evaluation with `--mode podman`."
        )

    include_dir, lib_dir = _resolve_cupti_paths()
    output = source_dir / _OUTPUT

    command = [
        compiler,
        "-std=c++14",
        "-fPIC",
        "-shared",
        f"-I{include_dir}",
        f"-L{lib_dir}",
        "-Wl,--no-as-needed",
        "-lcupti",
        "-ldl",
        f"-Wl,-rpath,{lib_dir}",
        "-Wl,--disable-new-dtags",
        "-o",
        str(output),
        *_SOURCES,
    ]

    try:
        subprocess.run(command, cwd=source_dir, check=True)
    except subprocess.CalledProcessError as exc:
        raise ProfilerBuildError(f"Compilation failed (exit {exc.returncode}): {' '.join(command)}") from exc

    return output


def main() -> int:
    try:
        output = build_profiler()
    except ProfilerBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Built CUPTI profiler library: {output}")
    print("Local CUPTI profiling is now available (no further configuration needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
