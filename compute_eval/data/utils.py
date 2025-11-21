import gzip
import json
import os
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, TypeAdapter

from compute_eval.data.data_model import (
    CudaCppProblem,
    CudaPythonProblem,
    FileSolution,
    GradedSolution,
    PatchSolution,
    Problem,
    Solution,
)


def _open_file(file_path: str | Path, mode: str):
    file_path = os.path.expanduser(file_path)
    if file_path.endswith(".gz"):
        return gzip.open(file_path, mode)
    else:
        return open(file_path, mode)


def stream_jsonl(path: str | Path) -> Iterable[dict]:
    """
    Yield each parsed JSON object from JSONL file(s).

    Args:
        path: Path to a single .jsonl/.jsonl.gz file or directory containing such files

    Yields:
        Parsed JSON objects from all matching files (in sorted filename order for directories)
    """

    def _is_jsonl_file(p: Path) -> bool:
        """Check if file has .jsonl or .jsonl.gz extension."""
        return p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz")

    def _stream_file(p: str | Path) -> Iterable[dict]:
        """Yield each parsed JSON object from a JSONL file (gzip supported)."""
        with _open_file(p, "rt") as fp:
            for line in fp:
                if any(not ch.isspace() for ch in line):
                    yield json.loads(line, strict=False)

    path = Path(os.path.expanduser(path))

    # Determine files to process
    if path.is_file():
        if not _is_jsonl_file(path):
            raise ValueError(f"File must end with .jsonl or .jsonl.gz, got: {path}")

        yield from _stream_file(path)
    elif path.is_dir():
        # Find all .jsonl and .jsonl.gz files in directory (sorted)
        files = sorted(f for f in path.iterdir() if f.is_file() and _is_jsonl_file(f))
        if not files:
            raise ValueError(f"No .jsonl or .jsonl.gz files found in directory: {path}")

        for file in files:
            yield from _stream_file(str(file))
    else:
        raise ValueError(f"Path does not exist: {path}")


def write_jsonl(file_path: str, data: list[dict | BaseModel], append: bool = False):
    """Write iterable of dicts or Pydantic model instances to a JSONL file."""
    mode = "at" if append else "wt"
    with _open_file(file_path, mode) as fp:
        for item in data:
            if isinstance(item, BaseModel):
                fp.write(item.model_dump_json(serialize_as_any=True) + "\n")
            elif isinstance(item, dict):
                fp.write(json.dumps(item) + "\n")
            else:
                raise ValueError(f"Cannot write object of type {type(item)}")


def read_problems(file_path: str) -> Generator[Problem, None, None]:
    adapter = TypeAdapter(Annotated[CudaCppProblem | CudaPythonProblem, Field(discriminator="type")])
    yield from (adapter.validate_python(data) for data in stream_jsonl(file_path))


def write_problems(file_path: str, problems: list[Problem], append: bool = False):
    write_jsonl(file_path, problems, append=append)


def read_solutions(file_path: str) -> Generator[Solution, None, None]:
    adapter = TypeAdapter(Annotated[FileSolution | PatchSolution, Field(discriminator="type")])
    yield from (adapter.validate_python(data) for data in stream_jsonl(file_path))


def write_solutions(file_path: str, solutions: list[Solution], append: bool = False):
    write_jsonl(file_path, solutions, append=append)


def read_graded_solutions(file_path: str) -> Generator[GradedSolution, None, None]:
    yield from (GradedSolution.model_validate(data) for data in stream_jsonl(file_path))


def write_graded_solutions(file_path: str, graded_solutions: list[GradedSolution], append: bool = False):
    write_jsonl(file_path, graded_solutions, append=append)
