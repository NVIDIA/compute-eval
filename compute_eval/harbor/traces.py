import copy
import io
import os
import tarfile
import tempfile
from pathlib import Path

from compute_eval.harbor.telemetry import (
    AttemptMetadata,
    GenerationArtifacts,
    TelemetryRun,
    TraceManifestRecord,
    TrialAttempt,
)

MANIFEST_PATH = "manifest.jsonl"


def _manifest_record(
    run: TelemetryRun,
    attempt: TrialAttempt,
) -> TraceManifestRecord:
    metadata = AttemptMetadata.from_attempt(run, attempt)
    return TraceManifestRecord(
        **metadata.model_dump(),
        trajectory_path=attempt.trajectory_path(run.run_id),
    )


def _add_bytes(archive: tarfile.TarFile, path: str, content: bytes) -> None:
    info = tarfile.TarInfo(path)
    info.size = len(content)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(content))


def _copy_previous_archive(
    archive: tarfile.TarFile,
    previous_archive: Path | None,
) -> bytes:
    if previous_archive is None or not previous_archive.exists():
        return b""

    manifest = b""
    with tarfile.open(previous_archive, "r:gz") as previous:
        for member in previous.getmembers():
            if member.name == MANIFEST_PATH:
                extracted = previous.extractfile(member)
                manifest = extracted.read() if extracted is not None else b""
                continue
            extracted = previous.extractfile(member) if member.isfile() else None
            archive.addfile(copy.copy(member), extracted)
    return manifest


def write_trace_archive(
    artifacts: GenerationArtifacts,
    run: TelemetryRun | None,
    previous_archive: Path | None = None,
) -> Path:
    output_path = artifacts.trajectories
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        with tarfile.open(temporary_path, "w:gz") as archive:
            previous_manifest = _copy_previous_archive(archive, previous_archive)
            manifest_records: list[TraceManifestRecord] = []
            if run is not None:
                for attempt in run.attempts:
                    record = _manifest_record(run, attempt)
                    manifest_records.append(record)
                    if attempt.trajectory_json is not None and record.trajectory_path is not None:
                        _add_bytes(
                            archive,
                            record.trajectory_path,
                            attempt.trajectory_json.encode("utf-8"),
                        )

            manifest = previous_manifest
            if manifest and not manifest.endswith(b"\n"):
                manifest += b"\n"
            manifest += b"".join(f"{record.model_dump_json()}\n".encode() for record in manifest_records)
            _add_bytes(archive, MANIFEST_PATH, manifest)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
