# Dataset Card for ComputeEval

**ComputeEval** is a benchmark for evaluating LLM-generated CUDA code on **correctness** and **performance**. Each problem provides a self-contained programming challenge — spanning kernels, runtime APIs, memory management, parallel algorithms, and GPU libraries — with a held-out test harness for functional validation and optional performance benchmarks for measuring GPU execution time against a baseline.

**Homepage:** [github.com/NVIDIA/compute-eval](https://github.com/NVIDIA/compute-eval)

---

## Dataset Overview

| | |
|---|---|
| **Dataset Owner** | NVIDIA Corporation |
| **Dataset Creation Date** | 2025-04-02 |
| **License** | NVIDIA Evaluation Dataset License Agreement (data). Apache 2.0 (evaluation harness code). |
| **Intended Usage** | Researchers and developers evaluating the CUDA code generation capabilities of large language models. Use cases include benchmarking LLM correctness and performance on GPU programming tasks, comparing models across CUDA domains, and identifying areas for improvement in LLM training for systems programming. |

This dataset is for research and development only.

---

## Dataset Characterization

**Data Collection Method**
- Hybrid: Human, Synthetic — Problems were authored by NVIDIA engineers and in some cases generated or modified with assistance from frontier language models (OpenAI, Anthropic). All problems were reviewed and validated by human engineers.

**Labeling Method**
- Hybrid: Human, Synthetic — Test harnesses, build/test commands, and domain group assignments were authored by NVIDIA engineers. Difficulty ratings and tags were augmented with assistance from frontier language models. All labels were reviewed and validated by human engineers.

---

## Dataset Quantification

- **Record Count:** 566 problems across 4 versioned datapacks (2025.1: 127, 2025.2: 231, 2025.3: 405, 2026.1: 566)
- **Feature Count:** 19 fields per problem (task_id, type, prompt, group, metadata, context_files, test_files, source_references, build_command, test_command, benchmark_command, timing_mode, baseline_solution, min_cuda_toolkit, compute_capability, requires_datacenter_gpu, timeout_seconds, date, schema_version)
- **Total Data Storage:** ~2.7 MB compressed across all datapacks (problems + baseline solutions)

---

## Problem Groups

Problems are organized into **groups** by domain. Each problem belongs to exactly one group, which determines the CUDA APIs and programming concepts it tests.

### CUDA C++

| Group | Description |
|-------|-------------|
| `cuda-runtime` | Kernel launch, memory management, streams, events, CUDA Graphs, cluster launch, occupancy |
| `cuda-kernels` | Shared memory, warp intrinsics, reductions, scans, stencils, tensor cores, cooperative groups |
| `cccl` | Thrust, CUB, libcu++ |
| `cublas` | BLAS levels 1-3, extensions, applications |
| `mathlibs` | cuSPARSE, cuSOLVER, cuFFT, cuRAND |
| `cudnn` | Convolutions, attention, matmul, normalization via cuDNN Graph API |

### CUDA Python

| Group | Description |
|-------|-------------|
| `cutile` | Tile-based kernels: matmul, attention, normalization, element-wise ops (SM 10.0+) |

For a full coverage map and domain backlog, see [`DOMAIN_MAP.md`](DOMAIN_MAP.md).

---

## Dataset Format

**Modality:** Text (structured JSONL)

Problems are distributed as **datapacks** - versioned releases stored as compressed tarballs (`.tar.gz`). Each datapack contains:

- **`metadata.json`** - Release version, creation timestamp, problem count, and integrity hashes
- **`problems.jsonl`** - One JSON object per line representing each problem

**Encoding:** UTF-8

---

## Data Schema

### Problem Structure

Each problem is a JSON object with the following schema:

#### Core Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | `string` | ✓ | Unique identifier (e.g., `"CUDA/0"`, `"CUDA/42"`) |
| `type` | `string` | ✓ | Problem type: `"cuda_cpp"` or `"cuda_python"` |
| `group` | `string` | ✓ | Domain group (see [Problem Groups](#problem-groups) above) |
| `schema_version` | `integer` | ✓ | Data schema version (currently `2`) |
| `date` | `string` | ✓ | Problem creation date (ISO 8601 format: `"YYYY-MM-DD"`) |
| `prompt` | `string` | ✓ | Natural language instruction for the programming task |
| `context_files` | `array` | ✓ | Files visible to the model/system (headers, stubs, helpers) |
| `test_files` | `array` | ✓ | Held-out test harness for evaluation (not shown to model/system) |
| `build_command` | `string` | — | Command to compile the solution (e.g., `"nvcc -I include ..."`) |
| `test_command` | `string` | ✓ | Command to execute tests (e.g., `"./test.out"`) |
| `benchmark_command` | `string` | — | Command to run performance benchmark (separate from tests) |
| `timing_mode` | `object` | — | How to extract performance timing (default: `{"type": "process"}`). See [Timing Modes](#timing-modes) |
| `baseline_solution` | `object` | — | Reference solution for speedup comparison |
| `min_cuda_toolkit` | `string` | — | Minimum CUDA Toolkit version required (e.g., `"12.0"`) |
| `compute_capability` | `string` | — | Minimum GPU compute capability required (default: `"8.0"`) |
| `requires_datacenter_gpu` | `boolean` | — | Whether the problem requires a datacenter-class GPU (default: `false`) |
| `timeout_seconds` | `float` | — | Maximum execution time allowed for tests (in seconds) |
| `source_references` | `string`, `array`, or `object` | — | Required API calls to verify in solution. Supports `{any: [...], all: [...]}` semantics |
| `python_version` | `string` | — | Required Python version (`cuda_python` problems only) |
| `metadata` | `object` | — | Problem metadata: `difficulty`, `tags`, `releases`, `do_not_release` |

#### File Objects (context_files & test_files)

Each file in `context_files` and `test_files` is an object with:

| Field | Type | Description |
|-------|------|-------------|
| `path` | `string` | Relative file path (e.g., `"include/kernel.h"`) |
| `content` | `string` | Complete file contents (UTF-8 encoded) |

#### Timing Modes

Problems that declare a `benchmark_command` can specify a `timing_mode` to control how performance timing is extracted. All modes report values in milliseconds.

| Type | Description |
|------|-------------|
| `process` | Default. Total application wall-clock time. |
| `kernels` | Sum of GPU kernel execution time. Supports `include`/`exclude` glob patterns. |
| `region` | Timing from NVTX-annotated code ranges. Supports `include`/`exclude` globs and `time_type` (`"kernel"` or `"wall_clock"`). |
| `gpu` | Total GPU time: kernel execution + memory transfers. |
| `custom` | The benchmark program prints its own timing to STDOUT as `COMPUTE_EVAL_TIME_MS: <value>`. |

---

## Evaluation Protocol

ComputeEval follows a strict separation between generation and evaluation:

### What Models/Systems See (Generation Time)

- Problem `prompt` - describes the task and requirements
- `context_files` - interface definitions and helper utilities
- `build_command` - compilation instructions (if provided)
- Minimum CUDA toolkit version and architecture requirements

### What Models/Systems Do NOT See

- `test_files` - held-out test harness
- Reference solutions

### Evaluation Process

1. Create temporary workspace
2. Write `context_files` to workspace
3. Write model-generated solution files to workspace
4. Write `test_files` to workspace (now visible)
5. Execute `build_command` to compile (if provided)
6. If compilation succeeds (or no build step required), execute `test_command`
7. Test exit code determines pass/fail (exit code 0 = pass)
8. If tests pass and `benchmark_command` is declared, run performance measurement using the configured `timing_mode` and profiler

This ensures models cannot overfit to test cases and must solve problems based solely on the natural language description and interface contracts.

---

## Versioning and Maintenance

ComputeEval maintains all previous release versions to enable longitudinal tracking of model progress. Users can benchmark against any release version to track improvements over time.

**Important:** We are committed to maintaining backward compatibility, but not bit-for-bit immutability. If we discover bugs in problems (e.g., unsolvable test cases, incorrect specifications), we reserve the right to fix them and update the corresponding datapacks in future releases. For exact historical versions, users can download specific releases from the git repository history.

This approach ensures users can continue using previous benchmark versions while benefiting from bug fixes and improvements.

---

## Example Problem

```json
{
  "task_id": "CUDA/3",
  "type": "cuda_cpp",
  "group": "cuda-runtime",
  "date": "2025-10-31",
  "prompt": "Implement a function called `launch` that launches a kernel function named `kernel` without using triple chevrons. The x, y, z grid and block dimensions will be provided as parameters to the `launch` function.\n\nThe function signature is defined in `include/kernel_launch.h`:\n```cuda\nvoid launch(int gridSizeX, int blockSizeX, int gridSizeY = 1, int blockSizeY = 1,\n            int gridSizeZ = 1, int blockSizeZ = 1);\n```\n\nThe `kernel` function is already defined with the following signature:\n```cuda\n__global__ void kernel(int *output, const int *input);\n```\n\nYour implementation should use the CUDA runtime API to launch the kernel with the specified grid and block dimensions.",
  "context_files": [
    {
      "path": "include/kernel_launch.h",
      "content": "#pragma once\n\nvoid launch(int gridSizeX, int blockSizeX, int gridSizeY = 1, int blockSizeY = 1,\n            int gridSizeZ = 1, int blockSizeZ = 1);\n"
    },
    {
      "path": "src/kernel.cu",
      "content": "#include <cuda_runtime.h>\n\n__global__ void kernel(int *output, const int *input) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    output[idx] = input[idx] * 2;\n}\n"
    }
  ],
  "test_files": [
    {
      "path": "test/test_main.cu",
      "content": "#include <cassert>\n#include <cuda_runtime.h>\n#include \"../include/kernel_launch.h\"\n\nint main() {\n    // Test implementation\n    int *d_input, *d_output;\n    cudaMalloc(&d_input, 256 * sizeof(int));\n    cudaMalloc(&d_output, 256 * sizeof(int));\n    \n    launch(4, 64);  // Launch with 4 blocks, 64 threads each\n    \n    cudaFree(d_input);\n    cudaFree(d_output);\n    return 0;\n}\n"
    }
  ],
  "build_command": "nvcc -I include -o test.out solution.cu src/kernel.cu test/*.cu -arch=sm_80",
  "test_command": "./test.out",
  "min_cuda_toolkit": "12.0",
  "timeout_seconds": 30.0,
  "source_references": {"any": null, "all": ["cudaLaunchKernelEx"]}
}
```

---

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse. Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

---

## License

**SPDX-FileCopyrightText:** Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
**SPDX-License-Identifier:** LicenseRef-NVIDIA-Evaluation

This dataset is licensed under the **NVIDIA Evaluation Dataset License Agreement**.
The license permits use of the dataset **solely for evaluation and benchmarking of
AI models**. In particular:

- **No training:** The dataset may not be used for training AI models or machine
  learning algorithms (Section 3.1).
- **No redistribution:** The dataset may not be sold, sublicensed, distributed,
  or hosted (Section 3.2).
- **Results may be published:** You may publish or otherwise disclose evaluation
  and benchmarking results.

See the full [license text](LICENSE) for details.

---

## Citation

If you use ComputeEval in your research, please cite:

```bibtex
@misc{computeeval2025,
  title={ComputeEval: A Benchmark for Evaluating Large Language Models on CUDA Code Generation},
  author={NVIDIA Corporation},
  year={2025},
  url={https://github.com/NVIDIA/compute-eval}
}
```
