# ComputeEval

A benchmark for evaluating LLM-generated CUDA code on **correctness** and **performance**.

ComputeEval provides a growing set of handcrafted CUDA programming challenges — spanning kernels, runtime APIs, GPU libraries, and template/DSL kernel authoring — along with tooling to generate, compile, and evaluate solutions from any LLM. Each problem includes a held-out test harness for functional correctness and can optionally include a performance benchmark that measures GPU execution time against a baseline solution.

The benchmark is under active development with frequent updates. New problems, domain groups, and evaluation capabilities are added in each release — see the [changelog](CHANGELOG.md) for details.

For periodic technical write-ups on what we've built, what we've learned, and where we're headed, see the [Engineering Diaries](docs/papers/README.md).

## Benchmark Structure and Evaluation

### Problem Organization

Each problem in ComputeEval is stored as a directory under `data`, containing:

```
CUDA-0/
├── problem-spec.yaml      # Problem metadata and configuration
├── context/               # Files visible to the tested model/system (headers, helpers)
│   ├── include/
│   │   └── kernel.h       # Interface contract to implement
│   └── helpers/
│       └── helpers.cu     # Optional helper utilities
├── solution/              # Reference implementation (not shown to tested model/system)
│   └── solution.cu
└── test/                  # Test harness (not shown to tested model/system)
    └── test/
        └── test_main.cu
```

### Problem Groups

Problems are organized into **groups** by domain. Each problem belongs to exactly one group, which determines the CUDA APIs and programming concepts it tests. The current groups are:

| Group | Language | Description |
|-------|----------|-------------|
| `cuda-runtime` | C++ | Kernel launch, memory management, streams, events, CUDA Graphs, cluster launch, occupancy |
| `cuda-kernels` | C++ | Shared memory, warp intrinsics, reductions, scans, stencils, tensor cores, cooperative groups |
| `cccl` | C++ | Thrust, CUB, libcu++ |
| `cublas` | C++ | BLAS levels 1-3, extensions, applications |
| `mathlibs` | C++ | cuSPARSE, cuSOLVER, cuFFT, cuRAND |
| `cudnn` | C++ | Convolutions, attention, matmul, normalization via cuDNN Graph API |
| `cutlass` | C++ | CUTLASS templates, collectives, schedulers, epilogues, and integration patterns |
| `cutile` | Python | Tile-based kernels: matmul, attention, normalization, element-wise ops (SM 10.0+) |
| `cute-dsl` | Python | CuTe DSL layouts, tensors, copy/MMA atoms, pipelines, and custom kernels |

The `cutlass` and `cute-dsl` groups are evaluated against a pinned stack — CUDA Toolkit 13.1 with
CUTLASS 4.6.1 (C++) and `nvidia-cutlass-dsl` 4.6.1 (Python). Their problem specs declare
`base_image: compute-eval:13.1.0`, which the Podman evaluator honors automatically. The local
evaluator ignores `base_image`, so `--mode=local` only works for these groups if you reproduce
that stack yourself. See [Evaluation Container Image](#evaluation-container-image).

You can use `--include` or `--exclude` to filter by group when generating solutions. For a full coverage map and domain backlog, see [`DOMAIN_MAP.md`](DOMAIN_MAP.md).

#### Problem Specification Format

The `problem-spec.yaml` file defines each problem's metadata and configuration:

```yaml
task_id: "CUDA/0"                     # Unique identifier (generally matches directory name)
date: "2024-12-19"                    # Problem creation date
problem_type: cuda_cpp                # Type: cuda_cpp or cuda_python
group: cuda-kernels                   # Domain group (see Problem Groups above)
prompt: "Implement a CUDA kernel..."  # Problem description shown to model

# Build and test configuration
build_command: "nvcc -I include -o test.out solution.cu test/*.cu"
test_command: "./test.out"
timeout_seconds: 30.0

# Optional: override the container image used by the Podman evaluator.
# Defaults to null (use localhost/compute-eval:<CTK_VERSION>).
# Ignored by the local evaluator.
base_image: null

# Requirements
min_cuda_toolkit: "13.1"             # Minimum CUDA version required

# Optional metadata
metadata:
  difficulty: medium                 # Problem difficulty level
  tags: [kernels, memory]            # Classification tags
  releases: [2025-1, 2025-2]         # Which releases include this problem
  do_not_release: false              # Internal-only flag to skip CI

source_references: null              # Optional: required API calls/symbols to verify
                                     #  - string: single item must be present
                                     #  - list of strings: all must be present
                                     #  - {any: [...]} at least one must be present
                                     #  - {all: [...]} all must be present
                                     #  - {all: [...], any: [...]} combines both
```

Example with source references requiring specific CUDA APIs:

```yaml
source_references:
  all: [cudaMalloc, cudaFree]        # Must use both malloc and free
  any: [cudaMemcpy, cudaMemcpyAsync] # Must use at least one copy method
```

### Evaluation Rules of Engagement

ComputeEval follows a strict separation between what systems/models see during generation versus what is used during evaluation:

**What the system/model sees (generation time):**
- Problem `prompt` - describes the task and requirements
- `context_files` - headers defining interfaces, optional helper utilities
- `build_command` - compilation instructions and flags
- Minimum CUDA toolkit version and architecture requirements

**What the system/model does NOT see:**
- `test_files` - held-out test harness that validates correctness
- `solution` - reference implementation directory

**During evaluation:**
1. A temporary workspace is created
2. `context_files` are written to the workspace
3. `test_files` are written to the workspace (now visible)
4. The model-generated solution files are written to the workspace
5. The `build_command` is executed to compile the unified workspace
6. If compilation succeeds, the `test_command` is executed
7. Test exit code determines pass/fail

This ensures models cannot overfit to test cases and must solve problems based solely on the problem description and interface contracts.

### Continuous Integration Validation

Every problem in the repository includes a known-good reference solution. Our CI pipeline continuously validates the integrity of the benchmark by:

1. Running the evaluation procedure on each problem's reference solution
2. Verifying that build commands compile successfully
3. Ensuring test harnesses execute correctly and pass
4. Validating that problem specifications are well-formed

This guarantees that all released problems are solvable and correctly specified.

### Performance Measurement

Problems can opt into performance measurement by declaring a `benchmark_command` in their `problem-spec.yaml`. This enables comparison of LLM-generated solutions against a known baseline on real GPU workloads.

#### Opting In

Add `benchmark_command` and optionally `timing_mode` to your problem spec:

```yaml
# Functional correctness (required)
test_command: "./test.out"

# Performance measurement (optional)
benchmark_command: "./benchmark.out"
timing_mode:
  type: region
  include: ["matmul*"]
```

The `benchmark_command` is fundamentally different from `test_command`:

- **`test_command`** validates correctness — it tests edge cases, boundary conditions, and error handling
- **`benchmark_command`** exercises a typical workload — it simulates realistic GPU work that can be profiled and compared against a baseline solution

The benchmark command runs only after the solution passes all functional tests. If a `baseline_solution` is provided for the problem, the framework computes speedup as `baseline_time / solution_time`.

#### Timing Modes

The `timing_mode` field controls how performance timing is extracted. All modes report values in **milliseconds**.

| Mode | Type | Description |
|------|------|-------------|
| `process` | Default | Total application wall-clock time (`application_duration_ms` from profiler summary) |
| `kernels` | Profiler | Sum of GPU kernel execution time. Supports `include`/`exclude` glob patterns to filter by kernel name |
| `region` | Profiler | Timing from NVTX-annotated code ranges. Supports `include`/`exclude` globs and `time_type` (`"kernel"` or `"wall_clock"`) |
| `gpu` | Profiler | Total GPU time: kernel execution + memory transfers |
| `custom` | Self-reported | The benchmark program prints its own timing to STDOUT (see below) |

The profiler-based modes (`process`, `kernels`, `region`, `gpu`) require a `--profile_mode` to be specified at evaluation time (see [Profiling Modes](#profiling-modes)). The `custom` mode does not require a profiler.

**Timing mode examples in `problem-spec.yaml`:**

```yaml
# Default: total application wall-clock time
# (timing_mode defaults to "process" if omitted)
timing_mode:
  type: process

# Only count specific kernels
timing_mode:
  type: kernels
  include: ["matmul_*", "*_optimized"]
  exclude: ["debug_*"]

# NVTX region kernel time (time_type defaults to "kernel")
timing_mode:
  type: region
  include: ["PerfTest*"]

# NVTX region wall-clock time (overrides the default time_type)
timing_mode:
  type: region
  include: ["PerfTest*"]
  time_type: wall_clock

# Total GPU time (kernels + memory transfers)
timing_mode:
  type: gpu

# Self-reported timing from STDOUT
timing_mode:
  type: custom
```

#### Region Timing with NVTX

The `region` timing mode uses [NVTX](https://nvidia.github.io/NVTX/) (NVIDIA Tools Extension) to measure annotated code ranges. Problem authors wrap the performance-critical section of their benchmark with NVTX push/pop calls, and the profiler attributes kernel execution and wall-clock time to those ranges.

**C++ (NVTX is included in the CUDA Toolkit):**

```cpp
#include <nvToolsExt.h>

// In your benchmark harness:
nvtxRangePushA("matmul_benchmark");
matmul_kernel<<<grid, block>>>(A, B, C, N);
cudaDeviceSynchronize();
nvtxRangePop();
```

**Python (the `nvtx` package is pre-installed in evaluation containers):**

```python
import nvtx

# As a context manager:
with nvtx.annotate("matmul_benchmark"):
    result = my_matmul(A, B)
    torch.cuda.synchronize()

# Or as a decorator:
@nvtx.annotate("matmul_benchmark")
def run_benchmark():
    return my_matmul(A, B)
```

#### Custom Timing

The `custom` timing mode lets the benchmark program report its own wall-clock time. This is useful when you need full control over timing (e.g., using CUDA events, excluding warmup iterations, or timing host-side logic).

The benchmark program must print a line to STDOUT matching this format:

```
COMPUTE_EVAL_TIME_MS: <value>
```

The value must be in **milliseconds**. If multiple matching lines are printed (e.g., warmup iterations), the last one is used.

**C++:**

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
my_kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
printf("COMPUTE_EVAL_TIME_MS: %f\n", ms);
```

**Python:**

```python
import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
result = my_function(A, B)
end.record()
torch.cuda.synchronize()

elapsed_ms = start.elapsed_time(end)
print(f"COMPUTE_EVAL_TIME_MS: {elapsed_ms}")
```

#### Profiling Modes

When evaluating solutions, the `--profile_mode` flag controls which profiler is used to collect GPU metrics. This applies to all timing modes except `custom`.

| Mode | Description |
|------|-------------|
| *(not set)* | No performance profiling. Functional correctness only. |
| `cupti` | Lightweight profiler using CUPTI via `LD_PRELOAD`. Collects kernel timing, memory transfers, and NVTX ranges. Lower overhead, suitable for most workloads. |
| `ncu` | NVIDIA Nsight Compute profiler. Collects detailed metrics including SM throughput and DRAM throughput percentages. Requires GPU profiling permissions. Higher overhead — reduce `n_workers` to avoid contention. |

### Problem Distribution

Released problems are distributed as **datapacks** — versioned archives hosted on the [nvidia/compute-eval](https://huggingface.co/datasets/nvidia/compute-eval) HuggingFace dataset. The evaluation harness downloads and caches them automatically; no manual download is required.

Each datapack contains:
- **`metadata.json`** — Release version, creation timestamp, problem count, and per-problem integrity hashes
- **`problems.jsonl`** — One JSON object per line, with all fields from `problem-spec.yaml` plus embedded `context_files` and `test_files`

This format provides:
- **Stable meaning** — A release always contains the same problems, asking the same questions. We fix defects; we never redefine a problem or change a release's problem set. See [Release Support](#release-support).
- **Integrity** — `metadata.json` records an MD5 per problem, so you can tell exactly which problems changed between two releases
- **Portability** — Self-contained archives that work offline once cached

#### Release Support

ComputeEval follows a continuous delivery model. We are committed to **permanently supporting all previous releases** so model developers can:
- Track progress over time against a fixed baseline
- Compare results with published benchmarks
- Ensure reproducibility of evaluation results

What supporting a release means in practice:

- **The problem set is fixed.** A shipped release never gains or loses problems. New problems land in the next release; retired problems simply stop appearing in new ones.
- **Each problem's intent is fixed.** If a problem should become harder or ask something different, it ships as a *new* problem; the original is left alone.
- **Defects still get fixed.** Ambiguous prompts, buggy headers, and flawed tests are corrected in place and flow into republished archives, so a release measures its stated intent more accurately over time. See [Versioning and Maintenance](DATASET_CARD.md#versioning-and-maintenance).


## Setup

### Prerequisites

- Python 3.12+
- NVIDIA GPU
- [Podman](https://podman.io/) — required for `--mode=podman`, the recommended evaluation mode
- CUDA Toolkit 13.1 or greater — required only for `--mode=local`; Podman mode provides the toolkit inside the container image

### Installation

```bash
# Evaluation harness (CLI + evaluation framework)
pip install nvidia-compute-eval[eval]

# Solution generator (CLI + generation framework)
pip install nvidia-compute-eval[gen]

# Both
pip install nvidia-compute-eval[all]

# Core data models and datapack API only
pip install nvidia-compute-eval
```

For development from source:

```bash
uv sync --extra eval
uv sync --extra gen
# or both:
uv sync --all-extras
```

### Evaluation Modes

The harness runs generated code in one of two modes, selected with the `--mode` flag:

- **`podman` (recommended)** — Executes each solution inside an isolated container built from `docker/Dockerfile.eval`. The image ships the CUDA Toolkit and a prebuilt CUPTI profiler, so functional correctness and CUPTI profiling work with no additional setup. This mode also provides the strongest isolation for untrusted model-generated code.
- **`local`** — Executes solutions directly on the host. Requires a local CUDA Toolkit installation and, for CUPTI profiling, a compiled profiler library (see [Local CUPTI Profiling](#local-cupti-profiling)).

### Local CUPTI Profiling

CUPTI profiling in `--mode=local` requires the `libcuda_profile.so` shared library. Podman mode builds this automatically inside the container image; for local mode, build it once after installation.

If ComputeEval is installed as a package, build the library in place with the bundled CLI:

```bash
compute_eval_build_profiler
```

This compiles `libcuda_profile.so` next to the packaged profiler sources, where the harness locates it automatically. A C++ compiler and the CUDA Toolkit (including the CUPTI headers and libraries) must be available. Set `CUDA_HOME` if the toolkit is not at `/usr/local/cuda`.

When working from a source checkout, build directly with the provided `Makefile`:

```bash
cd compute_eval/cuda_tools && make
```

To use a profiler library built elsewhere, point the harness at it explicitly:

```bash
export CUPTI_PROFILER_LIB=/path/to/libcuda_profile.so
```

### Evaluation Container Image

Podman mode executes solutions inside an isolated container built from
`docker/Dockerfile.eval`. Build it locally from the repo root:

```bash
podman build -f docker/Dockerfile.eval -t compute-eval:13.1.0 .
```

The harness defaults to `localhost/compute-eval:<CTK_VERSION>`, which
resolves to the image you just built in Podman's local store — no registry
login required.

If you prefer to build with Docker, transfer the image to Podman afterwards:

```bash
docker build -f docker/Dockerfile.eval -t compute-eval:13.1.0 .
docker save compute-eval:13.1.0 | podman load
```

To use a different toolkit version or pull from a remote registry, set the
environment variables before running the harness:

```bash
# Use a different CUDA toolkit version
export CTK_VERSION="13.2.0"

# Pull from a remote registry instead of building locally
export IMAGE_REGISTRY="registry.example.com/my-org"
```

The full image reference used is `<IMAGE_REGISTRY>/compute-eval:<CTK_VERSION>`.

The image also pins the CUTLASS stack that the `cutlass` and `cute-dsl` problems build
against: CUTLASS C++ headers at `/opt/cutlass` (exposed as `CUTLASS_PATH`) and the
`nvidia-cutlass-dsl` wheel, both at 4.6.1. Problems in those groups set
`base_image: compute-eval:13.1.0` and will not build under `--mode=local` unless you
reproduce that stack yourself.

### Pre-commit Hooks

Set up pre-commit hooks for code quality:

```bash
uv sync --group dev
uv run pre-commit install
```

### API Keys

To query an LLM, you must first obtain an API key from the respective service.

#### NVIDIA NIM

To use ComputeEval with NVIDIA-hosted models, you need an API key from
[build.nvidia.com](https://build.nvidia.com).

1. Go to [build.nvidia.com](https://build.nvidia.com)
1. Sign in with your account
1. Verify that you have sufficient credits to call hosted models
1. Navigate to the desired model and click on it
1. Click on `Get API Key`
1. Copy the generated API key
1. Export it as an environment variable:

```bash
export NEMO_API_KEY="<your-nvidia-key>"
```

#### OpenAI

Follow the instructions in the [OpenAI docs](https://openai.com/index/openai-api),
then:

```bash
export OPENAI_API_KEY="<your-openai-key>"
```

#### Anthropic (Claude)

Follow instruction on [Anthropic docs](https://www.anthropic.com/api), then:

```bash
export ANTHROPIC_API_KEY="<your-anthropic-key>"
```

## Usage

**Note:** This repository executes machine-generated CUDA code.
While it's unlikely that the code is malicious, it could still pose potential risks.
Therefore, all code execution requires the `--mode` flag to be set explicitly to `podman` or `local`.
We strongly recommend using Podman mode or a sandbox environment (e.g., a virtual machine) when running generated code to minimize security risks.

### Generating Solutions

Generate solutions for all problems in a release and write a solutions datapack ready for evaluation.

```bash
# Set your API key (OpenAI, Anthropic, or NVIDIA NIM)
export OPENAI_API_KEY="<your-key>"

compute_eval_generate generate_samples \
  --model gpt-5 \
  --solutions_per_problem 5 \
  --n_workers 8
```

For NVIDIA NIM or any other OpenAI-compatible endpoint, pass `--base_url`:

```bash
export NEMO_API_KEY="<your-nvidia-key>"

compute_eval_generate generate_samples \
  --model nvidia/nemotron-3-super-120b-a12b \
  --base_url https://integrate.api.nvidia.com/v1 \
  --solutions_per_problem 5 \
  --n_workers 8
```

This will:
- Load problems for the specified release from the HuggingFace dataset (cached locally after first download)
- Generate `solutions_per_problem` completions per problem using the specified model
- Write a solutions datapack (e.g., `2026-2-gpt-5-solutions.tar.gz`) ready for evaluation

Generation is resumable — if you interrupt it, re-running with the same arguments picks up where it left off using intermediate files in `--temp_dir`.

Use `--include` or `--exclude` to generate for a subset of groups:

```bash
compute_eval_generate generate_samples \
  --model gpt-5 \
  --include '["thrust", "cub"]'
```

Parameters can also be supplied via a YAML config file (CLI arguments take precedence):

```yaml
# gen-config.yaml
release: "2026-2"
model: gpt-5
solutions_per_problem: 5
n_workers: 8
```

```bash
compute_eval_generate generate_samples --config_file=gen-config.yaml
```

#### Few-shot generation

`compute_eval_generate` is a zero-shot alias over the unified generator. For
few-shot prompting — where the model is shown completed example problems before
the target — use the `compute_eval_harness` CLI, which exposes the same
arguments plus `--num_shots` and `--demonstrations_dir`:

```bash
compute_eval_harness generate_samples \
  --model gpt-5 \
  --num_shots 3 \
  --demonstrations_dir path/to/demonstrations \
  --solutions_per_problem 5
```

The bundled demonstrations shipped with the package can be used directly; omit
`--demonstrations_dir` to keep `--num_shots 0` (zero-shot). The unified CLI also
supports raw text-completion endpoints (`--api_mode completion`) and custom stop
sequences (`--stop`). It writes both the solutions datapack and the scored
target-subset problems datapack (named with the seed and problem count).

#### Agent-based generation (`harbor`)

The generators above issue a single prompt per problem. To instead let a coding
agent work a problem inside a sandboxed container — reading files, compiling,
and iterating — use the `harbor` subcommand:

```bash
compute_eval_harness harbor generate \
  --agent claude-code \
  --model <model-id> \
  --docker-image compute-eval-harbor-agent:latest \
  --attempts 1 \
  --concurrency 4
```

`--docker-image` is required and is used verbatim, so the image must ship the
full solve toolchain (compilers, profilers, and the agent CLI). Build
`docker/harbor-agent/Dockerfile` for a working default. Supported agents are
`claude-code` (default) and `opencode`; OpenCode additionally accepts
`--opencode-config` and requires a model.

Passing `--allow-host` puts the task on Harbor's allowlist network policy,
restricting agent egress to the hostnames, wildcards, IPs, or CIDRs you name
(repeatable). Without it no allowlist is applied, so the agent inherits the
container runtime's default network access. Use `--forward-env` to pass named
environment variables through to the agent — useful for credentials a mounted
`--skills-dir` needs.

Useful execution flags: `--attempts` (agent attempts per problem),
`--concurrency` (parallel trials), `--timeout` (seconds per agent, default
1800), `--gpus` (per task; `0` for CPU-only debugging), `--max-retries` with
`--retry-include`, `--limit`, and `--resume` to continue an interrupted run.
Problems come from `--release` by default, or from `--problems` to point at a
local datapack or YAML source directory.

The run writes `<release>-<model>-solutions.tar.gz` alongside per-run
`-harbor-usage.jsonl` and `-harbor-trajectories.tar.gz` artifacts, so token
usage and full agent transcripts are recoverable after the fact. The solutions
datapack is evaluated exactly like any other.

##### Optimize mode

`harbor optimize` targets performance rather than initial correctness. The agent
receives the full problem context, including test files, and starts from the
known-good baseline solution already placed in its workspace; its job is to make
it faster. Problems without a `benchmark_command` are skipped, since there is
nothing to measure a speedup against. Output is suffixed `-optimize-solutions.tar.gz`.

```bash
compute_eval_harness harbor optimize \
  --agent claude-code \
  --model <model-id> \
  --docker-image compute-eval-harbor-agent:latest
```

### Evaluating Solutions

```bash
compute_eval evaluate_functional_correctness \
  --release=2026-2 \
  --solutions_datapack=2026-2-gpt-5-solutions.tar.gz \
  --mode=podman \
  --k='(1, 3)' \
  --n_workers=4
```

**Security Note:** You must pass `--mode=podman` (or `--mode=local`) to run the evaluation. As described in the Evaluation Rules of Engagement section, this executes untrusted model-generated code, so use appropriate sandboxing. Podman mode is recommended.

This will:
- Load problems for the specified release from the HuggingFace dataset (cached locally after first download)
- Build and execute each solution in an isolated workspace with the test harness
- Output structured JSON with `pass@k` metrics and problem count
- Write results to a graded solutions file (auto-named per datapack, e.g., `2026-2-gpt-5-graded-solutions.jsonl`)

**Note:** The `k` parameter can be a single integer (`--k=1`) or a tuple (`--k='(1, 3)'`). For accurate pass@k estimates, ensure `max(k) <= solutions_per_problem`.

### Batch Evaluation

Pass a directory instead of a single file to evaluate multiple solution datapacks at once:

```bash
compute_eval evaluate_functional_correctness \
  --release=2026-2 \
  --solutions_datapack=./solutions/ \
  --mode=podman \
  --k=1 \
  --n_workers=4
```

All `*-solutions.tar.gz` files in the directory will be evaluated and each produces a separate graded-solutions file.

### Using a Config File

Parameters can be provided via a YAML config file (CLI arguments take precedence):

```yaml
# eval-config.yaml
release: "2026-2"
mode: podman
k: 1
n_workers: 4
```

```bash
compute_eval evaluate_functional_correctness \
  --config_file=eval-config.yaml \
  --solutions_datapack=2026-2-gpt-5-solutions.tar.gz
```

## Command Reference

### `evaluate_functional_correctness`

Evaluates the functional correctness of generated solutions by compiling and executing them against held-out test suites. Outputs structured JSON with `pass@k` metrics.

#### Configuration Parameters

All parameters can be specified in a YAML config file or passed as CLI arguments (CLI arguments take precedence).

- `release` (str): Release version to evaluate solutions for (e.g., `"2025-3"`). Default: `"2026-2"`
- `solutions_datapack` (str): Path to a solutions datapack file or a directory containing multiple `*-solutions.tar.gz` files for batch evaluation. **Required.**
- `problems_source` (str | None): Where to load problems from. When not set, problems are loaded from the [HuggingFace dataset](https://huggingface.co/datasets/nvidia/compute-eval) and cached locally. A `.tar.gz` path opens that datapack directly. Any other path is treated as a directory of YAML problem specs. Default: `None`
- `mode` (str | None): Evaluation execution mode. Must be set to `"podman"` or `"local"` to allow execution. Default: `None`
- `k` (int | tuple[int, ...]): K value(s) for pass@k evaluation. Default: `1`
- `n_workers` (int): Number of worker threads. Default: `4`
- `profile_mode` (str | None): Profiling mode for performance analysis. `"cupti"` for lightweight CUPTI profiling, `"ncu"` for NVIDIA Nsight Compute, or `None` to disable profiling. Only affects problems that declare a `benchmark_command`. Default: `None`
- `sanitizer_mode` (str | list[str]): Optional Compute Sanitizer gate. Use `"none"` (default), `"all"`, one tool, or a comma-separated/list subset of `memcheck`, `racecheck`, `synccheck`, and `initcheck`.

#### Compute Sanitizer

Compute Sanitizer is disabled by default. Enable memcheck alone:

```bash
compute_eval evaluate_functional_correctness \
  --release=2026-1 \
  --solutions_datapack=2026-1-my-solutions.tar.gz \
  --mode=podman \
  --sanitizer_mode=memcheck
```

Run any subset with a comma-separated selector, or use `all` for all four tools:

```bash
--sanitizer_mode=memcheck,racecheck
--sanitizer_mode=all
```

A YAML config may use a list:

```yaml
sanitizer_mode:
  - memcheck
  - synccheck
```

Selected tools always run in the order `memcheck`, `racecheck`, `synccheck`, `initcheck`, regardless of input order. A solution must first pass its normal `test_command`, which then runs under every selected tool. Findings from any tool, including racecheck warnings, fail the sanitizer gate. All selected tools still run so the graded result contains a complete diagnostic record.

The sanitizer gate applies to submitted solutions only. Timing baselines still run their normal correctness and benchmark commands; gold/baseline sanitation belongs in corpus authoring and release validation rather than being repeated for every grading run.

To report a clean sanitizer result, the hidden test must also prove that it exercised the intended workload. During each sanitized run, ComputeEval provides `COMPUTE_EVAL_SANITIZER_REPORT` and `COMPUTE_EVAL_SANITIZER_TOKEN` to the hidden test. After confirming the target launches, the hidden test writes JSON to the report path:

```json
{"token": "<value from COMPUTE_EVAL_SANITIZER_TOKEN>", "target_launches": 1}
```

The hidden harness should not pass either variable to candidate-controlled child processes. Candidate stdout is never accepted as coverage proof. Problems that do not yet implement this side channel receive an `invalid` sanitizer result rather than a false clean result: their ordinary correctness result is preserved, but performance benchmarking remains blocked. Missing or malformed coverage, missing sanitizer summaries or logs, and timeouts are all invalid runs. After every tool passes, ComputeEval reruns the normal correctness test before benchmarking.

#### Performance Profiling

To enable performance profiling during evaluation, pass `--profile_mode`:

```bash
compute_eval evaluate_functional_correctness \
  --release=2026-2 \
  --solutions_datapack=2026-2-gpt-5-solutions.tar.gz \
  --mode=podman \
  --profile_mode=cupti \
  --n_workers=2
```

When profiling is enabled, problems with a `benchmark_command` will be profiled after passing functional tests. The results include timing, throughput, and optional speedup metrics against baseline solutions.

In `--mode=local`, `--profile_mode=cupti` additionally requires the profiler library to be built (see [Local CUPTI Profiling](#local-cupti-profiling)).

**Using Nsight Compute (`ncu`):**

The `ncu` profiler collects detailed per-kernel metrics (SM throughput, DRAM throughput) but requires GPU profiling permissions and has higher overhead. Reduce `n_workers` (e.g., to 2) to avoid GPU contention.

```bash
compute_eval evaluate_functional_correctness \
  --release=2026-2 \
  --solutions_datapack=2026-2-my-solutions.tar.gz \
  --mode=local \
  --profile_mode=ncu \
  --n_workers=2
```

**Example Output:**

```json
{
  "pass_at_k": {
    "skipped": 0.0,
    "pass@1": 1.0
  },
  "problem_count": 5,
  "performance_analysis": {
    "invalid_skipped": 1,
    "avg_solution_time_ms": 0.065,
    "avg_sm_throughput_pct": 45.2,
    "avg_dram_throughput_pct": 23.8
  }
}
```

---

### `generate_samples`

Generates solutions for all problems in a release and writes a solutions datapack ready for evaluation with `compute_eval evaluate_functional_correctness`.

Requires `nvidia-compute-eval[gen]` to be installed.

#### Configuration Parameters

All parameters can be specified in a YAML config file or passed as CLI arguments (CLI arguments take precedence).

- `release` (str): Release version to generate solutions for. Default: `"2026-2"`
- `model` (str): Model name. **Required.** Examples: `"gpt-5"`, `"nvidia/nemotron-3-super-120b-a12b"`
- `base_url` (str | None): Base URL for the model API. Defaults to the OpenAI API. Default: `None`
- `problems_source` (str | None): Where to load problems from. When not set, problems are loaded from the [HuggingFace dataset](https://huggingface.co/datasets/nvidia/compute-eval) and cached locally. A `.tar.gz` path opens that datapack directly. Any other path is treated as a directory of YAML problem specs. Default: `None`
- `solutions_per_problem` (int): Number of solutions to generate per problem. Default: `1`
- `n_workers` (int): Number of parallel worker threads. Default: `4`
- `system_prompt` (str): System prompt for the model. Defaults to the built-in CUDA engineer prompt.
- `reasoning` (str | None): Reasoning effort level. One of `"low"`, `"medium"`, `"high"`. Mutually exclusive with `thinking`. Default: `None`
- `thinking` (bool | None): Enable extended thinking for models that support it. Mutually exclusive with `reasoning`. Default: `None`
- `temperature` (float | None): Sampling temperature. Default: `None`
- `top_p` (float | None): Nucleus sampling probability. Default: `None`
- `max_tokens` (int | None): Maximum tokens to generate. Default: `None`
- `temp_dir` (str | None): Directory for intermediate per-problem result files (enables resume). Defaults to the model name. Default: `None`
- `output_dir` (str | None): Directory for the final solutions datapack. Defaults to the current directory. Default: `None`
- `debug` (bool): Include raw prompts and completions in the output for debugging. Default: `False`
- `include` (list | None): Only generate for problems in these groups. Default: `None`
- `exclude` (list | None): Skip problems in these groups. Default: `None`

## Dataset

For more information about the dataset see [`DATASET_CARD.md`](DATASET_CARD.md).
For a full coverage map and ecosystem backlog see [`DOMAIN_MAP.md`](DOMAIN_MAP.md).

## License

The code in this repository is licensed under [Apache 2.0](LICENSE).

The dataset (everything under `data/`) is licensed under the [NVIDIA Evaluation
Dataset License Agreement](data/LICENSE). This license permits use of the dataset
**solely for evaluation and benchmarking of AI models**. In particular, the
dataset **may not be used for training AI models** (Section 3.1). You may publish
or otherwise disclose evaluation results.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development instructions.
