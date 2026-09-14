# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy ``compute_eval_generate`` entry point.

Zero-shot alias over the unified generator. Accepts the historical
``generate_samples`` argument surface and runs the unified generator with
``num_shots=0``. New few-shot and raw-completion options live on the
``compute_eval_harness`` CLI (compute_eval.generation.main).
"""

import yaml

from compute_eval.generation.generate import CompletionConfig, generate_samples


def _build_config(config_file: str | None, cli_kwargs: dict) -> CompletionConfig:
    """Merge a YAML config file and CLI args, with CLI taking precedence."""
    if config_file:
        with open(config_file) as f:
            config_data = yaml.safe_load(f) or {}
    else:
        config_data = {}
    config_data.update({k: v for k, v in cli_kwargs.items() if v is not None})
    # Zero-shot alias: few-shot demonstrations are never configured here.
    config_data["num_shots"] = 0
    config_data.pop("demonstrations_dir", None)
    return CompletionConfig(**config_data)


def generate_samples_with_config(config_file: str | None = None, **cli_kwargs):
    """Generate zero-shot solutions for all problems in a release.

    Example::

        compute_eval_generate generate_samples \\
            --model nvidia/nemotron-3-super-120b-a12b \\
            --base_url https://integrate.api.nvidia.com/v1 \\
            --solutions_per_problem 5
    """
    config = _build_config(config_file, cli_kwargs)
    return generate_samples(config)


def main():
    import fire

    fire.Fire({"generate_samples": generate_samples_with_config})


if __name__ == "__main__":
    main()
