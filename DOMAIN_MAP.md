# ComputeEval Domain Map

566 problems across 7 groups.

---

## What We Cover

### CUDA C++ (518 problems)

| Group | Problems | Covers |
|-------|----------|--------|
| `cuda-runtime` | 29 | Kernel launch, memory management, streams, events, CUDA Graphs, cluster launch, occupancy |
| `cuda-kernels` | 114 | Shared memory, warp intrinsics, reductions, scans, stencils, tensor cores, cooperative groups |
| `cccl` | 80 | Thrust, CUB, libcu++ |
| `cublas` | 81 | BLAS levels 1–3, extensions, applications |
| `mathlibs` | 101 | cuSPARSE, cuSOLVER, cuFFT, cuRAND |
| `cudnn` | 113 | Convolutions, attention, matmul, normalization via cuDNN Graph API |

### CUDA Python (48 problems)

| Group | Problems | Covers |
|-------|----------|--------|
| `cutile` | 48 | Tile-based kernels: matmul, attention, normalization, element-wise ops (SM 10.0+) |

Every problem today is **generation** — write a function or kernel from a specification.

---

## What We Don't Cover

### Domains

NVIDIA technologies not yet in ComputeEval.

**Libraries**

| Domain | One-liner |
|--------|-----------|
| cuBLASLt | Lightweight matmul API with epilogue fusion and algorithm selection |
| CUTLASS | Templated GEMM/convolution kernels, CuTe layout algebra |
| cuTENSOR | Tensor contractions, reductions, element-wise ops |
| cuBLASDx / cuFFTDx | Device-side BLAS and FFT (inside your own kernels) |
| cuSPARSELt | 2:4 structured sparsity for Ampere+ |
| cuDSS | Direct sparse solver |
| cuCollections | GPU hash maps and concurrent data structures |
| TensorRT | Network definition, plugins, ONNX import, quantization |
| Transformer Engine | FP8/FP4 GEMMs, delayed scaling, fused attention |
| Video Codec SDK | Hardware-accelerated video encode and decode |
| nvJPEG / NPP | Image and signal processing |
| nvCOMP | GPU compression (LZ4, Snappy, zstd) |
| OptiX | Ray tracing with RT cores |
| cuQuantum / cuPQC | Quantum circuit simulation, post-quantum cryptography |
| RAPIDS | cuDF, cuML, cuGraph, cuVS — GPU-accelerated data science |

**CUDA programming model**

| Domain | One-liner |
|--------|-----------|
| Tensor Memory Accelerator | Hopper+ async bulk copy engine |
| Virtual memory management | cuMemCreate / cuMemMap — fine-grained memory control |
| CUDA Math API | Device-side intrinsics and special functions |

**Multi-GPU & communication**

| Domain | One-liner |
|--------|-----------|
| NCCL | Collective and point-to-point communication |
| NVSHMEM | GPU-initiated communication (partitioned global address space) |
| GPUDirect | Direct GPU-to-NIC and GPU-to-storage transfers |
| CUDA-Aware MPI | GPU buffers in MPI calls |

**Python GPU programming**

| Domain | One-liner |
|--------|-----------|
| Triton | GPU kernel language with automatic optimization |
| CuPy | GPU arrays, custom kernels, array protocols |
| PyTorch CUDA extensions | Custom ops, cpp_extension, custom autograd |
| Numba CUDA | JIT-compiled CUDA kernels in Python |
| Warp | Differentiable GPU programming |
| JAX custom kernels | Pallas, custom_call, XLA integration |
| CUTLASS Python DSL | Python-based kernel authoring via cutlass.emit |

### Problem shapes

| Shape | Description |
|-------|-------------|
| Optimization | Given correct code, make it faster |
| Repair | Given broken code, find and fix the defect |
| Translation | Given working code, port it to a different API, language, or architecture |
