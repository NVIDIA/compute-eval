# Dataset Card for ComputeEval

**ComputeEval** is a benchmark dataset for evaluating Large Language Models on **CUDA code generation** tasks. Each problem provides a self-contained programming challenge designed to test various aspects of CUDA development, including kernel launches, memory management, parallel algorithms, and CUDA libraries (Thrust, CUB, etc.).

**Homepage:** [github.com/NVIDIA/compute-eval](https://github.com/NVIDIA/compute-eval)

---

## Dataset Format

Problems are distributed as **datapacks** - versioned releases stored as compressed tarballs (`.tar.gz`). Each datapack contains:

- **`metadata.json`** - Release version, creation timestamp, problem count, and integrity hashes
- **`problems.jsonl`** - One JSON object per line representing each problem

**Storage Format:** JSON Lines (`.jsonl`)
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
| `schema_version` | `integer` | ✓ | Data schema version (currently `2`) |
| `date` | `string` | ✓ | Problem creation date (ISO 8601 format: `"YYYY-MM-DD"`) |
| `prompt` | `string` | ✓ | Natural language instruction for the programming task |
| `context_files` | `array` | ✓ | Files visible to the model/system (headers, stubs, helpers) |
| `test_files` | `array` | ✓ | Held-out test harness for evaluation (not shown to model/system) |
| `build_command` | `string` | — | Command to compile the solution (e.g., `"nvcc -I include ..."`) |
| `test_command` | `string` | ✓ | Command to execute tests (e.g., `"./test.out"`) |
| `min_cuda_toolkit` | `string` | — | Minimum CUDA Toolkit version required (e.g., `"12.0"`) |
| `timeout_seconds` | `float` | — | Maximum execution time allowed for tests |
| `source_references` | `string` or `array` | — | Required API calls to verify in solution (e.g., `["cudaMalloc", "cudaFree"]`) |
| `metadata` | `object` | — | Additional problem metadata |
| `arch_list` | `array` | — | GPU architectures required (e.g., `["sm_80", "sm_89"]`) |

#### File Objects (context_files & test_files)

Each file in `context_files` and `test_files` is an object with:

| Field | Type | Description |
|-------|------|-------------|
| `path` | `string` | Relative file path (e.g., `"include/kernel.h"`) |
| `content` | `string` | Complete file contents (UTF-8 encoded) |

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
  "arch_list": ["sm_80", "sm_89", "sm_90"],
  "source_references": ["cudaLaunchKernelEx"]
}
```

---

## License

**SPDX-FileCopyrightText:** Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
**SPDX-License-Identifier:** LicenseRef-NVIDIA-Evaluation

This dataset is licensed under the **NVIDIA Evaluation Dataset License Agreement**.
See the full license text in [`data/LICENSE`](data/LICENSE).

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

---

## Additional Resources

- **Full Documentation:** [README.md](README.md)
- **Contributing Guidelines:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Issue Tracker:** [GitHub Issues](https://github.com/NVIDIA/compute-eval/issues)
