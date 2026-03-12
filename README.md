# ComputeEval

A benchmark for evaluating LLM-generated CUDA code on **correctness** and **performance**.

ComputeEval provides a growing set of handcrafted CUDA programming challenges — spanning kernels, runtime APIs, and GPU libraries — along with tooling to generate, compile, and evaluate solutions from any LLM. Each problem includes a held-out test harness for functional correctness and can optionally include a performance benchmark that measures GPU execution time against a baseline solution.

The benchmark is under active development with frequent updates. New problems, domain groups, and evaluation capabilities are added in each release — see the [changelog](CHANGELOG.md) for details.

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
| `cutile` | Python | Tile-based kernels: matmul, attention, normalization, element-wise ops (SM 10.0+) |

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

# Requirements
min_cuda_toolkit: "12.0"             # Minimum CUDA version required

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

### Release Datapacks

For production use, ComputeEval distributes problems as **datapacks** - versioned, immutable releases stored as compressed tarballs (`.tar.gz`):

```
data/releases/
├── 2025-1-problems.tar.gz
├── 2025-2-problems.tar.gz
├── 2025-3-problems.tar.gz
├── 2026-1-problems.tar.gz
```

#### Datapack Structure

Each datapack contains:
- **`metadata.json`** - Release version, creation timestamp, problem count, and integrity hashes
- **`problems.jsonl`** or **`solutions.jsonl`** - One JSON object per line representing each problem/solution

Problems in datapacks are serialized as JSON objects rather than directories. Each problem includes:
- All fields from `problem-spec.yaml`
- Embedded `context_files` (list of `{path, content}` objects)
- Embedded `test_files` (held-out, for evaluation only)

This format provides:
- **Immutability** - Released benchmarks never change
- **Integrity** - MD5 hashes verify problem consistency
- **Portability** - Self-contained archives easy to distribute
- **Versioning** - Clear separation between releases

#### Release Support

ComputeEval follows a continuous delivery model. New problems and improvements are released regularly as versioned datapacks.

We are committed to **permanently supporting all previous releases**. Model developers can benchmark against any release version to:
- Track progress over time against a fixed baseline
- Compare results with published benchmarks
- Ensure reproducibility of evaluation results


## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA Toolkit 12 or greater (for evaluation)

### Installation

Install the package using uv:

```bash
uv sync
```

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
Therefore, all code execution requires the `--mode` flag to be set explicitly to `docker` or `local`.
We strongly recommend using Docker mode or a sandbox environment (e.g., a virtual machine) when running generated code to minimize security risks.

### Using Preset NIM Models

To generate solutions using NVIDIA-hosted models:

```bash
uv run compute_eval generate_samples \
  --release=2026-1 \
  --base_url=https://integrate.api.nvidia.com/v1 \
  --model=openai/gpt-oss-120b \
  --solutions_per_problem=3 \
  --n_workers=10
```

**Note:** Set `NEMO_API_KEY` environment variable when using preset NIM models.

This will:
- Read problems from the 2026-1 release datapack
- Generate 3 solutions per problem using the `openai/gpt-oss-120b` model
- Write all solutions to: `2026-1-openai-gpt-oss-120b-solutions.tar.gz`

You can find the list of available models at [NVIDIA NIM Model Catalog](https://build.nvidia.com/models).

### Using OpenAI-Compatible APIs

For models with OpenAI-compatible API endpoints:

```bash
uv run compute_eval generate_samples \
  --release=2026-1 \
  --model=gpt-5 \
  --solutions_per_problem=3 \
  --n_workers=10
```

**Note:** Set `OPENAI_API_KEY` environment variable when using custom OpenAI-compatible endpoints.

This will:
- Read problems from the 2026-1 release datapack
- Generate 3 solutions per problem using the `gpt-5` model
- Write all solutions to: `2026-1-gpt-5-solutions.tar.gz`

### Using Configuration Files

You can also use YAML configuration files for convenience:

```yaml
# config.yaml
release: 2026-1
model: gpt-5
solutions_per_problem: 3
n_workers: 10
```

```bash
uv run compute_eval generate_samples --config_file=config.yaml
```

CLI arguments override config file values.

### Generating and Evaluating Solutions

After generating solutions (see examples above), evaluate them with:

```bash
uv run compute_eval evaluate_functional_correctness \
  --release=2026-1 \
  --solutions_datapack=2026-1-gpt-5-solutions.tar.gz \
  --mode=docker \
  --k='(1, 3)' \
  --n_workers=4
```

**Security Note:** You must pass `--mode=docker` (or `--mode=local`) to run the evaluation. As described in the Evaluation Rules of Engagement section, this executes untrusted model-generated code, so use appropriate sandboxing. Docker mode is recommended.

This will:
- Read the problems and solutions datapacks
- Build and execute each solution in an isolated workspace with the test harness
- Output structured JSON with `pass@k` metrics and problem count
- Write results to a graded solutions file (auto-named per datapack, e.g., `2026-1-gpt-5-graded-solutions.jsonl`)

**Note:** The `k` parameter can be a single integer (`--k=1`) or a tuple (`--k='(1, 3)'`). For accurate pass@k estimates, ensure `max(k) <= solutions_per_problem`.

## Command Reference

### `generate_samples`

Generates solutions for all problems in a release datapack using a specified model or custom API endpoint.

#### Configuration Parameters

All parameters can be specified in a YAML config file or passed as CLI arguments (CLI arguments take precedence).

- `release` (str): Release version to generate solutions for (e.g., "2025-3") (default: "2026-1")
- `include` (list[str] | None): Comma-separated list of groups to include (e.g., `"cuda-kernels,cublas"`). Mutually exclusive with `exclude`. If not set, all groups are included. (default: None)
- `exclude` (list[str] | None): Comma-separated list of groups to exclude. Mutually exclusive with `include`. If not set, no groups are excluded. (default: None)
- `problems_datapack_dir` (str): Directory where released problem datapacks are stored (default: "data/releases/")
- `solutions_per_problem` (int): Number of solutions to generate per problem (default: 1)
- `n_workers` (int): Number of worker threads to use (default: 10)
- `system_prompt` (str): System prompt for the model (default: predefined CUDA programming prompt)
- `model` (str): Model to use (use an appropriate NIM or use an OpenAI model name) (required)
- `base_url` (str | None): Custom API base URL (default: None)
- `reasoning` (str | None): Reasoning level for OpenAI models (e.g., "low", "medium", "high") (default: None)
- `temperature` (float): Sampling temperature for generation (default: 1.0)
- `top_p` (float): Nucleus sampling parameter (default: Model dependent)
- `max_tokens` (int | None): Maximum tokens to generate (default: None, model dependent)
- `temp_dir` (str | None): Temporary directory for intermediate results (default: None)
- `debug` (bool): Include system prompt, prompt, and completion in output for debugging (default: False)

**Note**: `model` must be specified.

### `evaluate_functional_correctness`

Evaluates the functional correctness of generated solutions by compiling and executing them against held-out test suites. Outputs structured JSON with `pass@k` metrics.

#### Configuration Parameters

All parameters can be specified in a YAML config file or passed as CLI arguments (CLI arguments take precedence).

- `release` (str): Release version to evaluate solutions for (e.g., "2025-3") (default: "2026-1")
- `solutions_datapack` (str): Path to a solutions datapack file or a directory containing multiple `*-solutions.tar.gz` files for batch evaluation (required)
- `problems_datapack_dir` (str): Directory where released problem datapacks are stored (default: "data/releases/")
- `mode` (str | None): Evaluation execution mode. Must be set to `"docker"` or `"local"` to allow execution (default: None)
- `k` (int | tuple[int, ...]): K value(s) for pass@k evaluation (default: 1)
- `n_workers` (int): Number of worker threads (default: 4)
- `profile_mode` (str | None): Profiling mode for performance analysis. `"cupti"` for lightweight CUPTI profiling, `"ncu"` for NVIDIA Nsight Compute, or `None` (default) to disable profiling. Only affects problems that declare a `benchmark_command`.

#### Performance Profiling

To enable performance profiling during evaluation, pass `--profile_mode`:

```bash
uv run compute_eval evaluate_functional_correctness \
  --release=2026-1 \
  --solutions_datapack=2026-1-gpt-5-solutions.tar.gz \
  --mode=local \
  --profile_mode=cupti \
  --n_workers=2
```

When profiling is enabled, problems with a `benchmark_command` will be profiled after passing functional tests. The results include timing, throughput, and optional speedup metrics against baseline solutions.

**Using Nsight Compute (`ncu`):**

The `ncu` profiler collects detailed per-kernel metrics (SM throughput, DRAM throughput) but requires GPU profiling permissions and has higher overhead. Reduce `n_workers` (e.g., to 2) to avoid GPU contention.

```bash
uv run compute_eval evaluate_functional_correctness \
  --release=2026-1 \
  --solutions_datapack=data/releases/2026-1-baseline-solutions.tar.gz \
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
