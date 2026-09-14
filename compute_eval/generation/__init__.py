# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-file completion generation for ComputeEval (few-shot + zero-shot)."""

try:
    from compute_eval.generation.model import ModelInterface
    from compute_eval.generation.openai import OpenAIModel
except ImportError as e:
    raise ImportError(
        f"compute_eval.generation requires optional dependencies: {e}. "
        "Install them with: pip install nvidia-compute-eval[gen]"
    ) from e

try:
    from compute_eval.generation.internal import MODEL_CLASSES
except ModuleNotFoundError as error:
    if error.name != f"{__name__}.internal":
        raise
    MODEL_CLASSES = {}


def get_model_class(model: str) -> type[ModelInterface]:
    return MODEL_CLASSES.get(model, OpenAIModel)


# Imported after get_model_class is defined: generate.py depends on it via
# ``from . import ModelInterface, get_model_class``.
from compute_eval.generation.generate import CompletionConfig, generate_samples  # noqa: E402

__all__ = [
    "CompletionConfig",
    "ModelInterface",
    "OpenAIModel",
    "generate_samples",
    "get_model_class",
]
