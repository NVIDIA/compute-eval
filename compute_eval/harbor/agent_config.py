import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

from harbor.models.trial.config import AgentConfig

from compute_eval.harbor import opencode

CLAUDE_CODE_MAX_TURNS = "500"
OPENCODE_OUTPUT_TOKEN_MAX = "128000"


def _forward_requested_env(args: Namespace, agent_env: dict[str, str]) -> None:
    # Pass named launch-env variables through to the agent so a mounted skill can
    # reach whatever external service it needs. The harness stays agnostic about
    # which vars these are: the operator names them via --forward-env, and only
    # vars that are actually set in the launch environment are forwarded.
    for name in args.forward_env or []:
        value = os.environ.get(name)
        if value:
            agent_env[name] = value


def _copy_env(agent_env: dict[str, str], *names: str) -> None:
    for name in names:
        if value := os.environ.get(name):
            agent_env[name] = value


def _load_opencode_config(config_value: str | None) -> dict[str, Any]:
    if config_value is None:
        return {}
    config_path = Path(config_value).expanduser()
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"OpenCode config must be a JSON object: {config_path}")
    return config


def resolve_agent_config(args: Namespace) -> AgentConfig:
    agent_env: dict[str, str] = {}
    agent_kwargs: dict[str, Any] = {}
    model_name = args.model
    agent_import_path: str | None = None
    if args.opencode_config is not None and args.agent != "opencode":
        raise ValueError("--opencode-config requires --agent opencode")
    if args.agent == "claude-code":
        agent_kwargs["max_turns"] = CLAUDE_CODE_MAX_TURNS
    elif args.agent == "opencode":
        opencode_config = _load_opencode_config(args.opencode_config)
        configured_model = opencode_config.get("model")
        if configured_model is not None and not isinstance(configured_model, str):
            raise ValueError("OpenCode config model must be a string")
        model_name = model_name or configured_model
        if model_name is None:
            raise ValueError("OpenCode requires --model or a model in --opencode-config")
        if opencode_config:
            agent_kwargs["opencode_config"] = opencode_config
            _copy_env(agent_env, "OPENAI_API_KEY", "OPENAI_BASE_URL")
        provider = model_name.partition("/")[0]
        if provider == "openai":
            _copy_env(agent_env, "OPENAI_API_KEY", "OPENAI_BASE_URL")
        elif provider == "anthropic":
            _copy_env(agent_env, "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL")
        agent_env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = os.environ.get(
            "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", OPENCODE_OUTPUT_TOKEN_MAX
        )
        agent_import_path = opencode.AGENT_IMPORT_PATH

    _forward_requested_env(args, agent_env)

    agent_name = None if agent_import_path else args.agent
    # Deliberately leave override_timeout_sec unset: the task.toml [agent]
    # timeout_sec (written by task.py) is the single source of truth and is
    # what Harbor's viewer surfaces. Harbor falls back to it when there is no
    # per-trial override.
    return AgentConfig(
        name=agent_name,
        import_path=agent_import_path,
        model_name=model_name,
        kwargs=agent_kwargs,
        env=agent_env,
    )
