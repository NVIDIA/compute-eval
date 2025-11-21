# compute-eval

ComputeEval: Evaluating Large Language Models for CUDA Code Generation

ComputeEval is a framework designed to generate and evaluate CUDA code from Large Language Models.
It features:

- A set of handcrafted CUDA programming challenges designed to evaluate an LLM's capability at writing reliable CUDA code
- Utilities for generating multiple solutions to each challenge
- Utilities for functional correctness evaluation of generated solutions

ComputeEval is currently in Alpha. We plan to refine the evaluation framework
and make frequent updates to the dataset with additional problems spanning all
aspects of CUDA development.

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

#### Problem Specification Format

The `problem-spec.yaml` file defines each problem's metadata and configuration:

```yaml
task_id: "CUDA/0"                     # Unique identifier (generally matches directory name)
date: "2024-12-19"                    # Problem creation date
problem_type: cuda_cpp                # Type: cuda_cpp or cuda_python
prompt: "Implement a CUDA kernel..."  # Problem description shown to model

# Build and test configuration
build_command: "nvcc -I include -o test.out solution.cu test/*.cu"
test_command: "./test.out"
timeout_seconds: 30.0

# Requirements
min_cuda_toolkit: "12.0"             # Minimum CUDA version required
arch_list: []                        # Optional: specific GPU architectures

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

### Release Datapacks

For production use, ComputeEval distributes problems as **datapacks** - versioned, immutable releases stored as compressed tarballs (`.tar.gz`):

```
data/releases/
├── 2025-1-problems.tar.gz
├── 2025-2-problems.tar.gz
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

#### Release Strategy

ComputeEval follows a regular release schedule:

- **2025.1** (Released) - Initial benchmark with 231 problems
- **2025.2** (Released) - Second release with expanded coverage
- **2025.3** (Upcoming) - Third release scheduled November 2025

We are committed to **permanently supporting all previous releases**. Model developers can benchmark against any release version to:
- Track progress over time against a fixed baseline
- Compare results with published benchmarks
- Ensure reproducibility of evaluation results


## Setup

### Prerequisites

- Python 3.11+
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

**Note:** This repository executes machine-generated CUDA C++ code.
While it's unlikely that the code is malicious, it could still pose potential risks.
Therefore, all code execution requires the `--allow_execution` flag.
We strongly recommend using a sandbox environment (e.g., a Docker container or virtual machine) when running generated code to minimize security risks.

### Using Preset NIM Models

To generate solutions using NVIDIA-hosted models:

```bash
uv run compute_eval generate_samples \
  --release=2025-2 \
  --base_url=https://integrate.api.nvidia.com/v1 \
  --model=openai/gpt-oss-120b \
  --solutions_per_problem=3 \
  --n_workers=10
```

**Note:** Set `NEMO_API_KEY` environment variable when using preset NIM models.

This will:
- Read problems from the 2025-2 release datapack
- Generate 3 solutions per problem using the `openai/gpt-oss-120b` model
- Write all solutions to: `2025-2-openai-gpt-oss-120b-solutions.tar.gz`

You can find the list of available models at [NVIDIA NIM Model Catalog](https://build.nvidia.com/models).

### Using OpenAI-Compatible APIs

For models with OpenAI-compatible API endpoints:

```bash
uv run compute_eval generate_samples \
  --release=2025-2 \
  --model=gpt-5 \
  --solutions_per_problem=3 \
  --n_workers=10
```

**Note:** Set `OPENAI_API_KEY` environment variable when using custom OpenAI-compatible endpoints.

This will:
- Read problems from the 2025-2 release datapack
- Generate 3 solutions per problem using the `gpt-5` model
- Write all solutions to: `2025-2-gpt-5-solutions.tar.gz`

### Using Configuration Files

You can also use YAML configuration files for convenience:

```yaml
# config.yaml
release: 2025-2
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
  --solutions_datapack=2025-2-gpt-5-solutions.tar.gz \
  --allow_execution=true \
  --k='(1, 3)' \
  --n_workers=4
```

**Security Note:** You must pass `--allow_execution=true` to run the evaluation. As described in the Evaluation Rules of Engagement section, this executes untrusted model-generated code, so use appropriate sandboxing.

This will:
- Read the problems and solutions datapacks
- Build and execute each solution in an isolated workspace with the test harness
- Output structured JSON with `pass@k` metrics and problem count
- Write results to a graded solutions file (e.g., `2025-2-graded-solutions.jsonl`)

**Note:** The `k` parameter can be a single integer (`--k=1`) or a tuple (`--k='(1, 3)'`). For accurate pass@k estimates, ensure `max(k) <= solutions_per_problem`.

## Command Reference

### `generate_samples`

Generates solutions for all problems in a release datapack using a specified model or custom API endpoint.

#### Configuration Parameters

All parameters can be specified in a YAML config file or passed as CLI arguments (CLI arguments take precedence).

- `release` (str): Release version to generate solutions for (e.g., "2025-2") (default: "2025-2")
- `problems_datapack_dir` (str): Directory where released problem datapacks are stored (default: "data/releases/")
- `solutions_per_problem` (int): Number of solutions to generate per problem (default: 1)
- `n_workers` (int): Number of worker threads to use (default: 10)
- `system_prompt` (str): System prompt for the model (default: predefined CUDA programming prompt)
- `model` (str): Model to use (use an appropriate NIM or use an OpenAI model name) (required)
- `base_url` (str | None): Custom API base URL (default: None)
- `reasoning` (str | None): Reasoning level for OpenAI models (e.g., "low", "medium", "high") (default: None)
- `temperature` (float): Sampling temperature for generation (default: 1.0)
- `top_p` (float): Nucleus sampling parameter (default: Model dependent)
- `max_tokens` (int): Maximum tokens to generate (default: Model dependent)
- `temp_dir` (str | None): Temporary directory for intermediate results (default: None)
- `debug` (bool): Include system prompt, prompt, and completion in output for debugging (default: False)

**Note**: `model` must be specified.

### `evaluate_functional_correctness`

Evaluates the functional correctness of generated solutions by compiling and executing them against held-out test suites. Outputs structured JSON with `pass@k` metrics.

#### Configuration Parameters

All parameters can be specified in a YAML config file or passed as CLI arguments (CLI arguments take precedence).

- `solutions_datapack` (str): Path to the solutions datapack file (required)
- `problems_datapack_dir` (str): Directory where released problem datapacks are stored (default: "data/releases/")
- `allow_execution` (bool): Whether to allow execution of untrusted code - must be set to True (default: False)
- `k` (int | tuple[int, ...]): K value(s) for pass@k evaluation (default: 1)
- `n_workers` (int): Number of worker threads (default: 4)
- `results_file` (str | None): Path to output results file (default: auto-generated from release name)

## Dataset

For more information about the dataset see [`DATASET_CARD.md`](DATASET_CARD.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development instructions.
