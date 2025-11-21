import re
import shutil
import subprocess


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
