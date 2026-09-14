# Fixed few-shot examples

These problems are a small, frozen demonstration set for base-model
ComputeEval prompts. They are separate from the scored ComputeEval datapack and
must never be included in the reported score.

The set covers three compact source shapes: a host-managed CUDA Runtime
wrapper, a device-pointer grid-stride kernel, and a two-dimensional kernel with
partial blocks. The examples leave room for the scored task and do not teach
benchmark-library APIs or real-world domain algorithms.

All three are original synthetic tasks with their own prompt, interface,
reference implementation, and test suite.
