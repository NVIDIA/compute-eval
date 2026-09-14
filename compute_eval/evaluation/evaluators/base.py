import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

from compute_eval import EvaluatorRuntime
from compute_eval.data.data_model import (
    FileSolution,
    GradedSolution,
    PatchSolution,
    Problem,
    SanitizerStatus,
    SanitizerTool,
    Solution,
    SourceReferences,
)
from compute_eval.data.metrics_data_model import CustomTimingMode, PerformanceMetrics
from compute_eval.evaluation.profilers import PerformanceProfiler
from compute_eval.evaluation.sanitizer import DEFAULT_SANITIZER_TIMEOUT_SECONDS, run_compute_sanitizer
from compute_eval.utils.eval_utils import GpuInfo, parse_semver
from compute_eval.utils.parsing import get_most_likely_language

SOURCE_ENCODING = "utf-8"


def resolve_workspace_path(work_dir: Path, file_path: str) -> Path:
    if Path(file_path).is_absolute():
        raise ValueError(f"Absolute file paths are not allowed: {file_path}")
    if ".." in Path(file_path).parts:
        raise ValueError(f"Parent directory segments are not allowed: {file_path}")

    workspace_root = work_dir.resolve()
    target_path = (workspace_root / file_path).resolve(strict=False)
    target_path.relative_to(workspace_root)
    return target_path


def verify_source_references(
    solution: Solution,
    source_references: str | list[str] | SourceReferences | None,
) -> bool:
    """Check that solution files contain all required source references."""
    if source_references is None or not isinstance(solution, FileSolution):
        return True

    all_refs: list[str] = []
    any_refs: list[str] = []

    if isinstance(source_references, str):
        all_refs = [source_references]
    elif isinstance(source_references, list):
        all_refs = source_references
    elif isinstance(source_references, SourceReferences):
        if source_references.all is not None:
            all_refs = source_references.all
        if source_references.any is not None:
            any_refs = source_references.any

    if not all_refs and not any_refs:
        return True

    all_remaining = {ref.encode(SOURCE_ENCODING) for ref in all_refs}
    any_remaining = {ref.encode(SOURCE_ENCODING) for ref in any_refs}

    for file in solution.files:
        encoded_content = file.content.encode(SOURCE_ENCODING)
        language = get_most_likely_language(file.path, encoded_content)
        if language is not None:
            for ref, _ in language.find_matching_subtrees(encoded_content, all_remaining | any_remaining):
                all_remaining.discard(ref)
                if ref in any_remaining:
                    any_remaining.clear()
                if not all_remaining and not any_remaining:
                    return True

    return not all_remaining and not any_remaining


@contextmanager
def _work_dir_context(base_dir: str | None = None):
    with tempfile.TemporaryDirectory(dir=base_dir) as tmpdir:
        yield Path(tmpdir)


class SolutionEvaluator(ABC):
    """
    Abstract base class for evaluating CUDA solutions.

    Provides common validation and workspace setup logic, with execution strategy
    delegated to concrete implementations (Podman, local subprocess, etc.).
    """

    def __init__(
        self,
        ctk_major: int,
        ctk_minor: int,
        gpu_info: GpuInfo | None = None,
        base_dir: str | None = None,
        profiler: PerformanceProfiler | None = None,
    ):
        """
        Initialize evaluator.

        Args:
            ctk_major: Major CUDA toolkit version supported
            ctk_minor: Minor CUDA toolkit version supported
            gpu_info: Optional GPU info for compute capability checks
            base_dir: Optional base directory for temporary workspaces
            profiler: Optional performance profiler
        """
        self._ctk_major = ctk_major
        self._ctk_minor = ctk_minor
        self._gpu_info = gpu_info
        self._base_dir = base_dir
        self._profiler = profiler

        # For session-only caching, compute capability is sufficient to differentiate GPU systems
        self._system_hash = self._compute_system_hash()

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        problem: Problem,
        profiler: PerformanceProfiler | None,
    ) -> "SolutionEvaluator":
        """
        Factory method to create an evaluator from configuration.

        Each subclass implements this to handle its specific setup requirements.

        Args:
            problem: Problem being evaluated (for language detection, etc.)
            profiler: Optional profiler for performance analysis

        Returns:
            Configured evaluator instance
        """
        pass

    @abstractmethod
    def evaluator_runtime(self, workdir: Path) -> EvaluatorRuntime:
        """
        Build the execution function for running commands in the evaluator's context.

        Args:
            workdir: The working directory path

        Returns:
            ExecutionFn callable
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Teardown any resources held by the evaluator.
        """
        pass

    def evaluate_solutions(
        self,
        problem: Problem,
        solutions: list[Solution],
        check_source_references: bool = True,
        sanitizer_tools: tuple[SanitizerTool, ...] = (),
    ) -> list[GradedSolution]:
        """
        Main entry point for evaluating multiple solutions to a problem.

        Args:
            problem: The problem specification
            solutions: List of solutions to evaluate
            check_source_references: If False, skip source reference validation
            sanitizer_tools: Compute Sanitizer tools to run before benchmarking

        Returns:
            List of GradedSolution with results
        """
        # First evaluate all solutions
        graded_solutions = [
            self._evaluate_internal(problem, solution, check_source_references, sanitizer_tools)
            for solution in solutions
        ]

        # Check if at least one solution passed
        any_passed = any(graded.passed for graded in graded_solutions)
        any_benchmarkable = any(
            graded.passed
            and (graded.sanitizer_result is None or graded.sanitizer_result.status == SanitizerStatus.PASSED)
            for graded in graded_solutions
        )

        # Only collect baseline if at least one solution passed and we have the necessary configuration.
        # Custom timing mode doesn't require a profiler; other modes do.
        has_timing_support = isinstance(problem.timing_mode, CustomTimingMode) or self._profiler is not None
        collect_baseline = (
            any_passed
            and any_benchmarkable
            and has_timing_support
            and problem.benchmark_command
            and problem.baseline_solution is not None
        )

        if collect_baseline:
            # Baselines are validated when the corpus is authored or released.
            # Re-running the selected sanitizer tools here would repeat that
            # work for every grading run; only submitted solutions use the
            # opt-in sanitizer gate.
            baseline_graded = self._evaluate_internal(
                problem,
                problem.baseline_solution,
                check_source_references,
                (),
            )

            if not baseline_graded.passed or baseline_graded.benchmark_output is not None:
                baseline_error = (
                    baseline_graded.benchmark_output
                    or baseline_graded.test_output
                    or baseline_graded.build_output
                    or "baseline evaluation failed"
                )
                print(f"Baseline benchmark {problem.task_id} failed:\n{baseline_error}")
                print(f"Skipping baseline comparison and marking as failed for all solutions to {problem.task_id}.")

                for graded in graded_solutions:
                    graded.solution_metrics = None
                    graded.solution_time = None
                    graded.speedup = None
                    graded.benchmark_output = f"Baseline evaluation failed: {baseline_error}"
            else:
                for graded in graded_solutions:
                    if graded.passed:
                        graded.baseline_metrics = baseline_graded.solution_metrics
                        graded.baseline_time = baseline_graded.solution_time
                        if (
                            graded.solution_time is not None
                            and graded.solution_time > 0
                            and graded.baseline_time is not None
                            and graded.baseline_time > 0
                        ):
                            graded.speedup = graded.baseline_time / graded.solution_time

        return graded_solutions

    def _evaluate_internal(
        self,
        problem: Problem,
        solution: Solution,
        check_source_references: bool = True,
        sanitizer_tools: tuple[SanitizerTool, ...] = (),
    ) -> GradedSolution:
        # Validate environment.
        # First check if we're configured with a valid min CTK (if applicable).
        if not self._validate_ctk(problem):
            return GradedSolution(
                task_id=solution.task_id,
                solution=solution,
                problem=problem,
                passed=False,
                skipped=True,
            )

        # Next check that we have the requested compute capability (if applicable).
        if not self._validate_cc(problem):
            return GradedSolution(
                task_id=solution.task_id,
                solution=solution,
                problem=problem,
                passed=False,
                skipped=True,
            )

        # Validate solution
        solution_valid, solution_error = self._validate_solution(problem, solution)
        if not solution_valid:
            return GradedSolution(
                task_id=solution.task_id,
                solution=solution,
                problem=problem,
                passed=False,
                skipped=False,
                build_output=solution_error,
            )

        with _work_dir_context(self._base_dir) as workdir_path:
            # Setup workspace with all necessary files
            self._setup_workspace(workdir_path, problem, solution)

            # Validate source references before execution
            if check_source_references and not verify_source_references(solution, problem.source_references):
                return GradedSolution(
                    task_id=solution.task_id,
                    solution=solution,
                    problem=problem,
                    passed=False,
                    skipped=False,
                    test_output="[VALIDATION ERROR] Solution does not include required source references.",
                )

            try:
                exec_fn: EvaluatorRuntime = self.evaluator_runtime(workdir_path)

                def _wrapped_exec_fn(
                    command: str,
                    timeout_seconds: float,
                    tag: str | None,
                ) -> tuple[bool, str]:
                    exit_code, timed_out, out = exec_fn(command, timeout_seconds, tag)
                    return exit_code or timed_out, out

                # Run build command (if set)
                build_output = None
                if problem.build_command:
                    build_failed, build_output = _wrapped_exec_fn(
                        problem.build_command,
                        problem.timeout_seconds or 60.0,
                        "build",
                    )
                    if build_failed:
                        return GradedSolution(
                            task_id=problem.task_id,
                            solution=solution,
                            problem=problem,
                            passed=False,
                            skipped=False,
                            build_output=build_output,
                        )

                test_failed, test_output = _wrapped_exec_fn(
                    problem.test_command,
                    problem.timeout_seconds or 60.0,
                    "test",
                )
                if test_failed:
                    return GradedSolution(
                        task_id=problem.task_id,
                        solution=solution,
                        problem=problem,
                        passed=False,
                        skipped=False,
                        build_output=build_output,
                        test_output=test_output,
                    )

                sanitizer_result = None
                if sanitizer_tools:
                    sanitizer_result = run_compute_sanitizer(
                        tools=sanitizer_tools,
                        test_command=problem.test_command,
                        workdir=workdir_path,
                        timeout_seconds=max(
                            problem.timeout_seconds or 0.0,
                            DEFAULT_SANITIZER_TIMEOUT_SECONDS,
                        ),
                        exec_fn=exec_fn,
                    )
                    if sanitizer_result.status != SanitizerStatus.PASSED:
                        return GradedSolution(
                            task_id=problem.task_id,
                            solution=solution,
                            problem=problem,
                            passed=sanitizer_result.status == SanitizerStatus.INVALID,
                            skipped=False,
                            build_output=build_output,
                            test_output=test_output,
                            sanitizer_result=sanitizer_result,
                        )

                perf_metrics, solution_time, bench_out = self._benchmark(problem, workdir_path, exec_fn)

                return GradedSolution(
                    task_id=problem.task_id,
                    solution=solution,
                    problem=problem,
                    passed=True,
                    skipped=False,
                    build_output=build_output,
                    test_output=test_output,
                    benchmark_output=bench_out,
                    solution_metrics=perf_metrics,
                    solution_time=solution_time,
                    sanitizer_result=sanitizer_result,
                )
            finally:
                self.teardown()

    def _benchmark(
        self,
        problem: Problem,
        workdir_path: Path,
        exec_fn: EvaluatorRuntime,
    ) -> tuple[PerformanceMetrics | None, float | None, str | None]:
        """
        Run the benchmark command with profiling and extract performance metrics and timing.

        Returns:
            Tuple of (performance_metrics, solution_time, error_message)
        """
        if problem.benchmark_command is None:
            return None, None, None

        if isinstance(problem.timing_mode, CustomTimingMode):
            # Custom timing: run benchmark directly and parse self-reported time from STDOUT
            code, timed_out, benchmark_output = exec_fn(
                problem.benchmark_command,
                problem.timeout_seconds or 300.0,
                "benchmark",
            )

            if code or timed_out:
                error_msg = f"Benchmark failed. Exit code {code}, timeout={timed_out}.\nOutput: {benchmark_output}"
                return None, None, error_msg
            else:
                try:
                    solution_time = problem.timing_mode.extract_from_output(benchmark_output)
                    return None, solution_time, None
                except Exception as e:
                    error_msg = f"Failed to extract timing from benchmark output: {str(e)}\nOutput: {benchmark_output}"
                    return None, None, error_msg

        if self._profiler is None:
            return None, None, None

        try:
            perf_metrics, err = self._profiler.profile(
                test_command=problem.benchmark_command,
                workdir_path=workdir_path,
                timeout_seconds=problem.timeout_seconds or 300.0,
                execution_fn=exec_fn,
            )
            if perf_metrics is None:
                return None, None, f"Benchmark profiling failed: {err}"

            solution_time = problem.timing_mode.extract(perf_metrics)
            return perf_metrics, solution_time, None
        except Exception as e:
            return None, None, f"Exception during benchmarking: {str(e)}"

    def _validate_ctk(self, problem: Problem) -> bool:
        """
        Validate that the environment has the required CUDA toolkit version.

        Args:
            problem: The problem specification

        Returns:
            is_valid
        """
        required_ctk = problem.min_cuda_toolkit
        if required_ctk is None:
            return True

        required_ctk_major, required_ctk_minor, _ = parse_semver(problem.min_cuda_toolkit)
        return (self._ctk_major, self._ctk_minor) >= (required_ctk_major, required_ctk_minor)

    def _validate_cc(self, problem: Problem) -> bool:
        """
        Validate that the environment has the required compute capability.

        Args:
            problem: The problem specification

        Returns:
            is_valid
        """
        # If there is no compute capability requirement then we assert that there cannot be a datacenter GPU requirement
        if problem.compute_capability is None:
            return True

        required_cc = parse_semver(problem.compute_capability)
        if required_cc is None:
            return True

        required_cc_major, required_cc_minor, _ = required_cc

        if self._gpu_info is None:
            return False

        gpu_cc_major, gpu_cc_minor = self._gpu_info.compute_capability or (0, 0)
        if (gpu_cc_major, gpu_cc_minor) < (required_cc_major, required_cc_minor):
            return False

        return not problem.requires_datacenter_gpu or self._gpu_info.is_datacenter_gpu

    def _compute_system_hash(self) -> str | None:
        """
        Compute a system hash for baseline caching based on GPU info.

        For session-only caching, compute capability is sufficient to differentiate systems.
        Returns None if GPU info is not available.
        """
        if self._gpu_info is None or self._gpu_info.compute_capability is None:
            return None

        import hashlib

        cc_major, cc_minor = self._gpu_info.compute_capability
        hash_input = f"cc_{cc_major}.{cc_minor}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    @staticmethod
    def _validate_solution(problem: Problem, solution: Solution) -> tuple[bool, str | None]:
        """
        Validate that the solution meets basic preconditions.

        Args:
            problem: The problem specification
            solution: The solution to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if solution.task_id != problem.task_id:
            return False, "[VALIDATION ERROR] Solution failed validation checks."

        if isinstance(solution, FileSolution):
            if not solution.files:
                return False, "[VALIDATION ERROR] Solution failed validation checks."
            try:
                solution_paths = {resolve_workspace_path(Path("/tmp"), file.path) for file in solution.files}
                test_paths = {resolve_workspace_path(Path("/tmp"), file.path) for file in problem.test_files}
            except ValueError:
                return False, "[VALIDATION ERROR] Solution failed validation checks."
            if len(solution_paths) != len(solution.files) or solution_paths & test_paths:
                return False, "[VALIDATION ERROR] Solution failed validation checks."
        elif isinstance(solution, PatchSolution):
            raise NotImplementedError("Patch solutions are not supported by this evaluator.")

        return True, None

    @staticmethod
    def _setup_workspace(workdir_path: Path, problem: Problem, solution: Solution) -> None:
        """
        Set up the workspace with context files, test files, and solution files.

        Args:
            workdir_path: Path to the working directory
            problem: The problem specification
            solution: The solution to apply
        """
        # Write context files (public) from Problem to workdir
        for cf in problem.context_files:
            file_path = resolve_workspace_path(workdir_path, cf.path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(cf.content)

        # Write test files (private) from Problem to workdir
        for tf in problem.test_files:
            file_path = resolve_workspace_path(workdir_path, tf.path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(tf.content)

        # Apply the Solution to the workdir. Note that these may intentionally overwrite context files.
        if isinstance(solution, FileSolution):
            for file in solution.files:
                file_path = resolve_workspace_path(workdir_path, file.path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(file.content)
