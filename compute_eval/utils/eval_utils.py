import re
import shutil
import subprocess
import threading
from dataclasses import dataclass

_ncu_perm_error_shown = False
_ncu_perm_error_lock = threading.Lock()


@dataclass
class GpuInfo:
    num_gpus: int
    compute_capability: tuple[int, int] | None
    is_datacenter_gpu: bool


# noinspection PyBroadException
def _run(cmd) -> str | None:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        return p.stdout.strip()
    except Exception:
        return None


def _parse_nvcc_version(text):
    m = re.search(r"(?i)\bV(\d+\.\d+\.\d+)\b", text)
    return m.group(1) if m else None


def get_nvcc_version() -> str | None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    out = _run([nvcc, "--version"])
    return _parse_nvcc_version(out)


def parse_semver(version: str | None) -> tuple[int, int, int] | None:
    if version is None:
        return None

    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
    if not m:
        return None

    major, minor, patch = m.groups()
    return int(major), int(minor or 0), int(patch or 0)


def _is_datacenter_gpu(gpu_name: str) -> bool:
    """
    Heuristic to determine if GPU is datacenter-class.
    """
    name_upper = gpu_name.upper()
    datacenter_patterns = [
        "A100",
        "A40",
        "A30",
        "A16",
        "A10",
        "A2",
        "H100",
        "H200",
        "GH200",
        "B100",
        "B200",
        "GB200",
        "B300",
        "GB300",
        "GB10",
        "V100",
        "L40S",
        "L40",
        "L4",
    ]

    return any(pattern in name_upper for pattern in datacenter_patterns)


def parse_gpu_info(output: str) -> GpuInfo | None:
    # Filter lines to only include valid CSV format (starts with digit.digit)
    csv_pattern = re.compile(r"^\d+\.\d+,")

    lines = [line.strip() for line in output.strip().split("\n") if line.strip() and csv_pattern.match(line.strip())]

    if not lines:
        return None

    num_gpus = len(lines)

    # Parse first GPU (primary)
    first_line = lines[0]
    parts = [p.strip() for p in first_line.split(",", 1)]

    if len(parts) != 2:
        return None

    cc_str, gpu_name = parts

    # Parse compute capability (e.g., "8.0" -> (8, 0))
    try:
        major, minor = cc_str.split(".")
        compute_capability = (int(major), int(minor))
    except (ValueError, AttributeError):
        compute_capability = None

    is_datacenter = _is_datacenter_gpu(gpu_name)

    return GpuInfo(
        num_gpus=num_gpus,
        compute_capability=compute_capability,
        is_datacenter_gpu=is_datacenter,
    )
