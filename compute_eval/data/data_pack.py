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

# HuggingFace dataset repo and per-release file paths (Option A: explicit mapping).
# Update both this dict and ReleaseVersion when a new release ships.
_HF_REPO_ID = "nvidia/compute-eval"
_HF_RELEASE_PATHS: dict[ReleaseVersion, str] = {
    ReleaseVersion.V2026_2: "data/problems.jsonl",
    ReleaseVersion.V2026_1: "data/archive/2026-1/problems.jsonl",
    ReleaseVersion.V2025_3: "data/archive/2025-3/problems.jsonl",
    ReleaseVersion.V2025_2: "data/archive/2025-2/problems.jsonl",
    ReleaseVersion.V2025_1: "data/archive/2025-1/problems.jsonl",
}


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

    @classmethod
    def from_huggingface(
        cls,
        release: ReleaseVersion,
        *,
        include: list[ValidGroup] | None = None,
        exclude: list[ValidGroup] | None = None,
        token: str | None = None,
    ) -> "_HuggingFaceProblemDatapack":
        """Load problems from the HuggingFace dataset (nvidia/compute-eval).

        Results are cached locally by huggingface_hub. Requires compute-eval[eval].

        Args:
            release: Release version to load.
            include: If set, only yield problems in these groups.
            exclude: If set, skip problems in these groups.
            token: Optional HuggingFace API token (dataset is public).
        """
        return _HuggingFaceProblemDatapack(release, include=include, exclude=exclude, token=token)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        release: ReleaseVersion,
        *,
        include: list[ValidGroup] | None = None,
        exclude: list[ValidGroup] | None = None,
    ) -> "ProblemDatapack":
        """Load problems from a local .tar.gz datapack file.

        Args:
            path: Path to the .tar.gz datapack file.
            release: Expected release version. Raises ValueError if the file's
                embedded release does not match.
            include: If set, only yield problems in these groups.
            exclude: If set, skip problems in these groups.
        """
        dp = cls(path, include=include, exclude=exclude)
        if dp.metadata.release != release:
            raise ValueError(
                f"Problems datapack release '{dp.metadata.release.value}' does not match requested release '{release.value}'."
            )
        return dp

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        release: ReleaseVersion = ReleaseVersion.INTERNAL,
        include: list[ValidGroup] | None = None,
        exclude: list[ValidGroup] | None = None,
    ) -> "_DirectoryProblemDatapack":
        """Load problems directly from a source YAML directory tree.

        Reads problem-spec.yaml + context/ + test/ + solution/ per problem dir.
        Baseline solutions are always attached. Intended for CI and local authoring.

        Args:
            directory: Root directory containing problem subdirectories.
            release: Release version tag for metadata. Defaults to ReleaseVersion.INTERNAL.
            include: If set, only yield problems in these groups.
            exclude: If set, skip problems in these groups.
        """
        return _DirectoryProblemDatapack(directory, release=release, include=include, exclude=exclude)


def _build_groups(
    include: list[ValidGroup] | None,
    exclude: list[ValidGroup] | None,
) -> set[ValidGroup]:
    if include and exclude:
        raise ValueError("Cannot specify both include and exclude")
    all_groups: set[ValidGroup] = set(get_args(ValidGroup))
    if include:
        return set(include)
    if exclude:
        return all_groups - set(exclude)
    return all_groups


class _HuggingFaceProblemDatapack(ProblemDatapack):
    """Loads problems from the HuggingFace dataset via huggingface_hub (cached locally)."""

    def __init__(
        self,
        release: ReleaseVersion,
        *,
        include: list[ValidGroup] | None = None,
        exclude: list[ValidGroup] | None = None,
        token: str | None = None,
    ):
        self._path = None
        self._tar = None
        self._metadata = None
        self._file = None
        self._groups = _build_groups(include, exclude)
        self._release = release
        self._token = token

    def __enter__(self):
        hf_path = _HF_RELEASE_PATHS.get(self._release)
        if hf_path is None:
            available = [r.value for r in _HF_RELEASE_PATHS]
            raise ValueError(f"Release '{self._release.value}' is not available on HuggingFace. Available: {available}")

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("huggingface_hub is required: pip install compute-eval") from e

        kwargs = dict(repo_id=_HF_REPO_ID, filename=hf_path, repo_type="dataset", token=self._token)
        try:
            local_path = hf_hub_download(**kwargs)
        except Exception as network_err:
            try:
                local_path = hf_hub_download(**kwargs, local_files_only=True)
            except Exception:
                raise RuntimeError(
                    f"Failed to reach HuggingFace and no cached copy found for "
                    f"release '{self._release.value}'. Pre-warm the huggingface_hub cache "
                    f"on a machine with network access using huggingface_hub.hf_hub_download."
                ) from network_err
        self._file = open(local_path, encoding="utf-8")
        return self

    def __exit__(self, *args):
        if self._file:
            self._file.close()
            self._file = None

    def _stream(self) -> Generator[dict, None, None]:
        if self._file is None:
            raise RuntimeError("Must be used as a context manager")
        self._file.seek(0)
        for line in self._file:
            if any(not ch.isspace() for ch in line):
                yield json.loads(line, strict=False)

    @property
    def metadata(self) -> ProblemDatapackMetadata:
        if self._metadata is not None:
            return self._metadata
        if self._file is None:
            raise RuntimeError("Must be used as a context manager to access metadata")
        self._file.seek(0)
        count = 0
        hashes = {}
        for line in self._file:
            if any(not ch.isspace() for ch in line):
                item = json.loads(line, strict=False)
                hashes[item["task_id"]] = hashlib.md5(line.encode("utf-8")).hexdigest()
                count += 1
        self._file.seek(0)
        self._metadata = ProblemDatapackMetadata(
            release=self._release,
            total_count=count,
            description=None,
            task_id_hashes=hashes,
        )
        return self._metadata


class _DirectoryProblemDatapack(ProblemDatapack):
    """Loads problems from a local source directory tree of YAML problem specs."""

    def __init__(
        self,
        directory: str | Path,
        *,
        release: ReleaseVersion = ReleaseVersion.INTERNAL,
        include: list[ValidGroup] | None = None,
        exclude: list[ValidGroup] | None = None,
    ):
        self._path = None
        self._tar = None
        self._metadata = None
        self._groups = _build_groups(include, exclude)
        self._directory = Path(directory)
        self._release = release

    def __enter__(self):
        if not self._directory.is_dir():
            raise ValueError(f"Directory not found: {self._directory}")
        return self

    def __exit__(self, *args):
        pass

    def _matches_release(self, problem: Problem) -> bool:
        if self._release == ReleaseVersion.INTERNAL:
            return True
        return self._release in problem.metadata.releases

    def _iter_problems(self) -> Generator[Problem, None, None]:
        from compute_eval.data.loaders import _find_problem_dirs, load_problem_dir

        for problem_dir in _find_problem_dirs(self._directory):
            try:
                problem, solution = load_problem_dir(problem_dir)
            except Exception as e:
                raise RuntimeError(f"Error loading problem from '{problem_dir}': {e}") from e
            if problem.group in self._groups and self._matches_release(problem):
                problem.baseline_solution = solution
                yield problem

    def read_items(self) -> Generator[Problem, None, None]:
        yield from self._iter_problems()

    @property
    def metadata(self) -> ProblemDatapackMetadata:
        if self._metadata is not None:
            return self._metadata
        count = 0
        hashes = {}
        for problem in self._iter_problems():
            line = problem.model_dump_json(serialize_as_any=True) + "\n"
            hashes[problem.task_id] = hashlib.md5(line.encode("utf-8")).hexdigest()
            count += 1
        self._metadata = ProblemDatapackMetadata(
            release=self._release,
            total_count=count,
            description=None,
            task_id_hashes=hashes,
        )
        return self._metadata


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


def open_problem_datapack(
    release: ReleaseVersion,
    source: str | Path | None = None,
    *,
    include: list[ValidGroup] | None = None,
    exclude: list[ValidGroup] | None = None,
    token: str | None = None,
) -> ProblemDatapack:
    """Return a ProblemDatapack context manager for the given source.

    Args:
        release: Release version to load.
        source: Where to load problems from. ``None`` loads from HuggingFace
            (nvidia/compute-eval). A path ending in ``.tar.gz`` opens that
            datapack file directly. Any other path is treated as a directory
            of YAML problem specs.
        include: If set, only yield problems in these groups.
        exclude: If set, skip problems in these groups.
        token: Optional HuggingFace API token (only used when source is None).
    """
    if source is None:
        return ProblemDatapack.from_huggingface(release, include=include, exclude=exclude, token=token)

    source = Path(source)
    if source.name.endswith(".tar.gz"):
        if not source.exists():
            raise FileNotFoundError(f"Problems datapack not found: {source}")
        return ProblemDatapack.from_file(source, release, include=include, exclude=exclude)

    if not source.is_dir():
        raise ValueError(f"Problems source must be a .tar.gz file or an existing directory: {source}")

    return ProblemDatapack.from_directory(source, release=release, include=include, exclude=exclude)
