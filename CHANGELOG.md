# Changelog

## 2026.2

### Dataset

- **591 problems** (up from 566), now organized into 9 domain groups (up from 7)
- Added **CUTLASS** group (10 problems): CUTLASS C++ templates, collectives, schedulers, epilogues, and integration patterns
- Added **CuTe DSL** group (15 problems): CuTe DSL layouts, tensors, copy/MMA atoms, pipelines, and custom kernels
- Introduced **repair** problems — 14 of the new problems supply a defective or under-performing implementation to fix, rather than a blank slate. Every prior problem is a generation task.
- The new groups pin their evaluation stack: CUDA Toolkit 13.1 with CUTLASS 4.6.1 (C++) and `nvidia-cutlass-dsl` 4.6.1 (Python)

### Evaluation Harness

- Added an opt-in **Compute Sanitizer** gate (`--sanitizer_mode`): run `memcheck`, `racecheck`, `synccheck`, and `initcheck` — individually, as a subset, or `all` — before benchmarking. Disabled by default. Any finding, including racecheck warnings, fails the gate; every selected tool still runs so the graded result carries a complete diagnostic record. Results are reported per-solution in a new `sanitizer_result` field on graded solutions.
- Sanitizer runs require the problem's held-out test to confirm it exercised the intended workload, via the `COMPUTE_EVAL_SANITIZER_REPORT` / `COMPUTE_EVAL_SANITIZER_TOKEN` side channel. Problems that do not implement it report `invalid` rather than a false clean result: their correctness result is preserved, but benchmarking is blocked.
- Added the optional `base_image` field to problem specs: a problem can name the container image it is evaluated in, overriding the default. Honored by the Podman evaluator; ignored by the local evaluator.
- Replaced the Docker evaluator with **Podman** (`--mode=podman`, the recommended mode)
- Fixed CUPTI profiler under-counting of kernels in anonymous namespaces
- Restructured for PyPI: split `eval` / `gen` extras, moved evaluation code under `compute_eval/evaluation/`
- Ship a pure-Python wheel; the CUPTI profiler is now built on demand
- Default evaluation image resolves to `localhost/compute-eval:<CTK_VERSION>`

### Packaging

- **Breaking:** minimum Python is now **3.12** (was 3.10)
- New CLI entry points: `compute_eval_generate`, `compute_eval_harness`, and
  `compute_eval_build_profiler` (previously a single `compute_eval` command)

### Generation

- Added **agent-based generation**: `compute_eval_harness harbor generate` runs a coding agent
  (`claude-code` or `opencode`) against each problem in a sandboxed container, so the model can read
  files, compile, and iterate rather than answering a single prompt. Per-run token usage and full
  agent trajectories are written alongside the solutions datapack.
- Added `harbor optimize`, which starts the agent from the known-good baseline solution with full
  problem context and scores it on speedup rather than initial correctness
- Replaced the zero-shot generator with a unified few-shot/zero-shot generator. `compute_eval_generate`
  is retained as a zero-shot alias; `compute_eval_harness` adds `--num_shots` / `--demonstrations_dir`,
  raw text-completion endpoints (`--api_mode completion`), and custom stop sequences.

---

## 2026.1

### Dataset

- **566 problems** (up from 405), now organized into 7 domain groups
- Added **cuDNN** group (113 problems): convolutions, attention, matmul, normalization via cuDNN Graph API
- Added **CUDA Python** support with the **cutile** group (48 problems): tile-based kernels for matmul, attention, normalization, and element-wise ops (SM 10.0+)
- Introduced **problem groups** as a first-class concept — every problem now belongs to exactly one domain group

### Performance Measurement

- Added performance benchmarking as a new evaluation dimension alongside functional correctness
- Problems can now declare a `benchmark_command` for GPU workload profiling and a `baseline_solution` for speedup comparison — 499 of 566 problems ship with performance workloads
- New `timing_mode` field with 5 extraction strategies: `process`, `kernels`, `region` (NVTX), `gpu`, and `custom` (self-reported)
- CUPTI and Nsight Compute (`ncu`) profiler backends, selectable via `--profile_mode`
- Performance data model: `PerformanceMetrics`, `KernelMetrics`, `NvtxRangeMetrics`, `MemoryTransferMetrics` with SM/DRAM throughput tracking
- GPU isolation: concurrent workers are assigned to individual GPUs to avoid contention during profiling
- NVTX injection for reliable capture of default domain-scoped ranges
- NCU profiler tuned with `--set none` and `--clock-control boost` for lighter, more consistent metric collection

### Evaluation Harness

- Added `--include` / `--exclude` flags to filter problems by group during generation
- Added `--release` parameter to `evaluate_functional_correctness` (release version is now explicit rather than inferred)
- `solutions_datapack` now accepts a directory of datapacks for batch evaluation of multiple models at once
- Benchmark error handling: failures in baseline or solution benchmarking are tracked and reported per-solution
- `results_file` parameter removed; output filenames are now auto-generated per datapack

---

## 2025.3

### Dataset

- **405 problems** (up from 231)
- Added **mathlibs** group (101 problems): cuSPARSE, cuSOLVER, cuFFT, cuRAND
- Expanded **cublas** group from 8 to 81 problems

### Evaluation Harness

- Major rewrite: migrated from loose JSONL files to versioned **datapacks** (`.tar.gz` archives with `metadata.json` and `problems.jsonl`)
- Introduced structured data model with Pydantic-based problem and solution schemas
- Added `source_references` validation: problems can declare required API calls, types, or language constructs that solutions must use, verified statically via tree-sitter grammar-based symbol matching (complements functional correctness testing)
- Added Docker-based evaluation mode (`--mode=docker`) with automatic CUDA container image selection
- Added local evaluation mode (`--mode=local`)
- Added per-group metrics reporting in evaluation output
- Migrated from Poetry to uv for package management
- Added Dockerfile for containerized evaluation

---

## 2025.2

### Dataset

- **231 problems** (up from 127)
- Expanded **cuda-kernels** group from 59 to 114 problems
- Expanded **cccl** group from 49 to 80 problems
- Expanded **cuda-runtime** group from 18 to 29 problems
- Added initial **cublas** problems (8)

### Evaluation Harness

- Refactored data loading and generation pipeline
- Improved prompt formatting

---

## 2025.1

Initial public release.

### Dataset

- **127 problems** across 4 groups: cuda-kernels (59), cccl (49), cuda-runtime (18), cublas (1)
- All problems are CUDA C++ code generation tasks
- Each problem includes a prompt, context files, held-out test harness, and reference solution

### Evaluation Harness

- Solution generation via OpenAI-compatible APIs (including NVIDIA NIM)
- Functional correctness evaluation with pass@k metrics
- YAML configuration file support
