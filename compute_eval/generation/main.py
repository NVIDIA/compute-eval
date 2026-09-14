# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fire
import yaml

from .generate import CompletionConfig, generate_samples


def _build_config(config_file: str | None, cli_kwargs: dict) -> CompletionConfig:
    if config_file:
        with open(config_file) as config_stream:
            config_data = yaml.safe_load(config_stream) or {}
    else:
        config_data = {}
    config_data.update({key: value for key, value in cli_kwargs.items() if value is not None})
    return CompletionConfig(**config_data)


def _run(config_file: str | None = None, **cli_kwargs):
    config = _build_config(config_file, cli_kwargs)
    return generate_samples(config)


def main(argv: list[str] | None = None) -> None:
    fire.Fire({"generate_samples": _run}, command=argv)


if __name__ == "__main__":
    main()
