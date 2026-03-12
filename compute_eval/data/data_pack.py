import hashlib
import io
import json
import os
import tarfile
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from datetime import datetime
from pathlib import Path
from typing import Annotated, get_args

from pydantic import BaseModel, Field, TypeAdapter

from compute_eval.data.data_model import (
    CudaCppProblem,
    CudaPythonProblem,
    FileSolution,
    PatchSolution,
    Problem,
    ReleaseVersion,
    Solution,
    ValidGroup,
)


class DatapackMetadata(BaseModel):
    release: ReleaseVersion
    total_count: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    description: str | None = None


class ProblemDatapackMetadata(DatapackMetadata):
    task_id_hashes: dict[str, str] = Field(default_factory=dict)


class SolutionDatapackMetadata(DatapackMetadata):
    groups: list[ValidGroup] | None = None


class Datapack(ABC):
    _metadata_class: type[DatapackMetadata] = DatapackMetadata
    _data_filename = "data.jsonl"

    def __init__(self, path: str | Path):
        self._path = Path(os.path.expanduser(path))
        self._tar = None
        self._metadata = None

    def __enter__(self):
        if self._tar is not None:
            raise RuntimeError("DatapackReader is already open")

        self._tar = tarfile.open(self._path, "r:gz")
        return self

    def __exit__(self, *args):
        if self._tar:
            self._tar.close()
            self._tar = None

    @property
    def metadata(self) -> DatapackMetadata:
        if self._metadata is None:
            with tarfile.open(self._path, "r:gz") as tar:
                try:
                    f = tar.extractfile("metadata.json")
                    if f is None:
                        raise ValueError("metadata.json not found in data pack")

                    self._metadata = self._metadata_class.model_validate_json(f.read().decode("utf-8"))
                except KeyError as e:
                    raise ValueError(f"Invalid data pack: missing metadata.json in {self._path}") from e
        return self._metadata

    @abstractmethod
    def read_items(self) -> Generator[BaseModel, None, None]:
        pass

    def _stream(self) -> Generator[dict, None, None]:
        if self._tar is None:
            raise RuntimeError("DatapackReader must be used as a context manager")

        try:
            f = self._tar.extractfile(self._data_filename)
            if f is None:
                raise ValueError(f"{self._data_filename} not found in data pack")
        except KeyError as e:
            raise ValueError(f"Invalid data pack: missing {self._data_filename} in {self._path}") from e

        text_stream = io.TextIOWrapper(f, encoding="utf-8")
        for line in text_stream:
            if any(not ch.isspace() for ch in line):
                yield json.loads(line, strict=False)

    @classmethod
    def _write_item(cls, item: BaseModel, file, metadata: DatapackMetadata) -> str:
        """
        Write a single item and update metadata as needed.

        Returns the written line for use by subclasses.
        """
        line = item.model_dump_json(serialize_as_any=True) + "\n"
        file.write(line)
        metadata.total_count += 1

        return line

    @classmethod
    def create(
        cls,
        file_path: str | Path,
        items: Iterable[BaseModel],
        release: ReleaseVersion,
        description: str | None = None,
        **metadata_kwargs,
    ):
        file_path = Path(os.path.expanduser(file_path))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

            try:
                # Create metadata instance that will be built up
                metadata = cls._metadata_class(
                    release=release,
                    total_count=0,
                    created_at=datetime.now().isoformat(),
                    description=description,
                    **metadata_kwargs,  # type: ignore
                )

                for item in items:
                    cls._write_item(item, tmp, metadata)

                tmp.flush()

                # Create the tar.gz data pack
                with tarfile.open(file_path, "w:gz") as tar:
                    metadata_bytes = metadata.model_dump_json(indent=2).encode("utf-8")
                    metadata_info = tarfile.TarInfo(name="metadata.json")
                    metadata_info.size = len(metadata_bytes)
                    tar.addfile(metadata_info, io.BytesIO(metadata_bytes))
                    tar.add(tmp_path, arcname=cls._data_filename)

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


class ProblemDatapack(Datapack):
    _metadata_class = ProblemDatapackMetadata
    _data_filename = "problems.jsonl"
    _adapter = TypeAdapter(Annotated[CudaCppProblem | CudaPythonProblem, Field(discriminator="type")])

    def __init__(
        self,
        path: str | Path,
        *,
        include: list[ValidGroup] | None = None,
        exclude: list[ValidGroup] | None = None,
    ):
        super().__init__(path)

        if include and exclude:
            raise ValueError("Cannot specify both include_groups and exclude_groups")

        all_groups: set[ValidGroup] = set(get_args(ValidGroup))

        if include:
            self._groups = set(include)
        elif exclude:
            self._groups = all_groups - set(exclude)
        else:
            self._groups = all_groups

    @property
    def metadata(self) -> ProblemDatapackMetadata:
        return super().metadata  # type: ignore

    @property
    def groups(self) -> set[ValidGroup]:
        return self._groups

    def read_items(self) -> Generator[Problem, None, None]:
        """
        Read problems from the datapack, optionally filtering by group(s) if specified at initialization.

        Yields:
            Problem objects, filtered by group if specified at initialization.
        """

        def _filter(p: Problem) -> bool:
            if self._groups is None:
                return True
            return p.group in self._groups

        yield from filter(_filter, (self._adapter.validate_python(item) for item in self._stream()))

    @classmethod
    def _write_item(cls, item: Problem, file, metadata: ProblemDatapackMetadata) -> str:
        if item.task_id in metadata.task_id_hashes:
            raise ValueError(f"Duplicate task_id found when writing problem datapack: {item.task_id}")

        line = super()._write_item(item, file, metadata)
        problem_hash = hashlib.md5(line.encode("utf-8")).hexdigest()
        metadata.task_id_hashes[item.task_id] = problem_hash

        return line


class SolutionDatapack(Datapack):
    _metadata_class = SolutionDatapackMetadata
    _data_filename = "solutions.jsonl"
    _adapter = TypeAdapter(Annotated[FileSolution | PatchSolution, Field(discriminator="type")])

    def __init__(self, path: str | Path):
        super().__init__(path)

    @property
    def metadata(self) -> SolutionDatapackMetadata:
        return super().metadata  # type: ignore

    def read_items(self) -> Generator[Solution, None, None]:
        for s in (self._adapter.validate_python(item) for item in self._stream()):
            s.datapack_name = self._path.name.replace("-solutions.tar.gz", "")  # type: ignore
            yield s
