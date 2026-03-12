from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_CUPTI_SOURCES = (
    "libcuda_profile.cpp",
    "profiler_state.cpp",
    "nvtx_correlation.cpp",
    "activity_processing.cpp",
    "json_output.cpp",
    "profiler_lifecycle.cpp",
)
_OUTPUT_PACKAGE_PATH = "compute_eval/cuda_tools/libcuda_profile.so"


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        project_root = Path(self.root)
        source_dir = project_root / "compute_eval" / "cuda_tools"
        if not source_dir.exists():
            print(f"[compute-eval build] WARNING: cuda_tools directory not found at {source_dir}")
            return

        compiler = os.environ.get("CXX", "g++")
        if shutil.which(compiler) is None:
            print(f"[compute-eval build] WARNING: C++ compiler '{compiler}' not found in PATH")
            return

        cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        target_arch = os.environ.get("TARGETARCH", platform.machine()).strip().lower()
        target = "sbsa-linux" if target_arch in {"arm64", "aarch64"} else "x86_64-linux"

        include_dir = Path(os.environ.get("CUPTI_INCLUDE", str(cuda_home / "targets" / target / "include")))
        lib_dir = Path(os.environ.get("CUPTI_LIB", str(cuda_home / "lib64")))
        if not include_dir.exists():
            include_dir = cuda_home / "include"
        if not lib_dir.exists():
            lib_dir = cuda_home / "targets" / target / "lib"

        if not include_dir.exists() or not lib_dir.exists():
            print(f"[compute-eval build] WARNING: CUPTI paths not found: include={include_dir}, lib={lib_dir}")
            return

        output_dir = project_root / "build" / "cupti"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_lib = output_dir / "libcuda_profile.so"

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
            str(output_lib),
            *_CUPTI_SOURCES,
        ]

        try:
            subprocess.run(command, cwd=source_dir, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[compute-eval build] WARNING: Failed to build CUPTI profiler library: {' '.join(command)}")
            print(str(exc))
            return

        print(f"[compute-eval build] Built CUPTI profiler library at {output_lib}")

        force_include = build_data.setdefault("force_include", {})
        force_include[str(output_lib)] = _OUTPUT_PACKAGE_PATH
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
