from compute_eval.data.data_model import Problem, SourceFile

SYSTEM_PROMPT = """
You are a senior CUDA/C/C++ engineer. Produce complete, compilable solutions from a structured problem specification. Follow these rules:

General
- You will be given: a problem description, context files (editable), and build environment details (e.g., build command).
- Hidden tests exist but are not shown. Do not mention tests, do not write test code, and do not add I/O used only for testing.
- Use only the APIs and contracts specified in the problem and context files. Preserve all provided function signatures exactly.
- Prefer using only headers already present in the provided codebase. Avoid adding new headers unless strictly necessary and supported by the build command. Do not introduce third-party dependencies.

Context files policy
- You may modify provided context files when necessary. If you include any file in your solution output (new or modified), emit its full and final contents; your output will overwrite the provided version.
- Only emit files you add or modify. Do not output files that are unchanged, and do not include placeholder blocks saying "no changes" or similar.

Build command
- You should pay careful attention to the build command or any context files about the build process.
- The build command and/or context build files may include important hints about required files or expected project structure.  This likely includes the name of the expected solution file, important macros, standards, or linked libraries.
- Pay special attention to -I or -isystem flags -- they indicate important include paths.  Remember, if a -I or -isystem flag is present you do not need to include the relative path in your #include statements.


Output format
- Output only source files needed for the solution. No explanations or commentary.
- Each file must be in its own fenced code block, with the first line indicating its path as a comment.
  Example:
  ```
  // file: geodistance.cu
  #include "geodistance.h"
  ...
  ```

Code quality and constraints

The solution must compile cleanly with the provided build command and target architectures.
Avoid unnecessary heap allocations, environment access, and global mutable state. Keep deterministic behavior.
Honor all contracts, constants, and macros defined in provided headers.

For CUDA:
Implement kernels with correct global signatures and parameter types.
Bounds-check all memory accesses; consider grid-stride loops when appropriate for scalability.
Favor coalesced memory access and avoid undefined behavior.
Apply appropriate numerical stability practices when needed (e.g., clamp arguments before acos/asin).

Reasoning discipline

Think through edge cases and performance internally, but output only the final code files, no analysis or explanations. 
"""

_USER_PROMPT = """
Produce the complete solution as one or more source files that compile with the provided build command. Do not output anything except the code files.

Problem
Description: 
{prompt}

Build command: 
{build_command}

Context files:
{context_files_block}

Output requirements

Emit only the source files necessary to satisfy the problem (new or modified).
Only emit files you add or modify. Do not output files that are unchanged, and do not include placeholder blocks saying "no changes" or similar.
Do not include any test code or references to tests.
If an interface header is provided (e.g., declares functions to implement), place implementations in a corresponding .cu/.cc source file and include that header.
Begin your response with the first code block. 
"""

_CONTEXT_FILES_BLOCK_TEMPLATE = """
--- file: {path}
```{fence}
{content}
```
"""


def _fence_for_path(path: str) -> str:
    p = path.lower()
    if p.endswith((".cu", ".cuh")):
        return "cuda"
    if p.endswith((".cc", ".cpp", ".cxx")):
        return "cpp"
    if p.endswith(".c"):
        return "c"
    if p.endswith(".h") or p.endswith(".hpp"):
        return "h"
    # Default to plaintext if unknown
    return ""


def _format_context_files_block(context_files: list[SourceFile]) -> str:
    blocks: list[str] = []
    for source in context_files:
        fence = _fence_for_path(source.path)
        blocks.append(_CONTEXT_FILES_BLOCK_TEMPLATE.format(path=source.path, fence=fence, content=source.content))
    return "".join(blocks)


def to_user_message(problem: Problem) -> str:
    return _USER_PROMPT.format(
        prompt=problem.prompt,
        build_command=problem.build_command,
        context_files_block=_format_context_files_block(problem.context_files),
    )
