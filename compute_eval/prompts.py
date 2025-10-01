# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from compute_eval.data import Problem

SYSTEM_PROMPT = """You are a CUDA programming expert capable of generating high-quality, efficient CUDA code with best practices, optimized for performance and clarity.
Instructions:
1. Implement the body of the function(s). Do not include any additional code outside the function(s).
2. Wrap the completed function code, including the provided signatures, inside a single ```cuda markdown code block.
3. Make sure to use the precise function signature in your response if it's provided in the query.
"""

USER_PROMPT_TEMPLATE = """{user_prompt}\n{header_files_prompt}"""

HEADER_FILES_PROMPT_TEMPLATE = """
The following headers are already defined and should not be included in the response:
```cuda
{header_files}
```
Please do not include any additional headers in your response.
"""


def extract_header_files_from_problem(problem: Problem) -> str:
    declaration = problem.declaration
    header_files = []
    for header_file in declaration.split("\n"):
        header_file = header_file.strip()
        # header file starts with #include
        if header_file.startswith("#include"):
            header_files.append(header_file)
    return "\n".join(header_files)


def generate_user_prompt(problem: Problem, include_header_files: bool = False) -> str:
    header_files = extract_header_files_from_problem(problem)
    header_files_prompt = ""
    if include_header_files and header_files:
        header_files_prompt = HEADER_FILES_PROMPT_TEMPLATE.format(header_files=header_files)

    return USER_PROMPT_TEMPLATE.format(user_prompt=problem.prompt, header_files_prompt=header_files_prompt)
