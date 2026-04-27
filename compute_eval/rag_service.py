"""
Optional RAG (Retrieval-Augmented Generation) integration.

Queries an HTTP RAG endpoint to retrieve reference documentation before
generation. Retrieved content is injected into the prompt as supplementary
context.

When no RAG endpoint is configured, this module is never imported and
the generation pipeline behaves identically to before.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RAGServerConfig(BaseModel):
    """Configuration for a single RAG server to query for reference docs."""

    url: str = Field(..., description="RAG endpoint URL (e.g. https://host/v1/rag-context)")
    api_key: str | None = Field(
        default_factory=lambda: os.environ.get("RAG_API_KEY"),
        description="Optional API key sent as Bearer token. Defaults to RAG_API_KEY env var if set.",
    )
    timeout: float = Field(60.0, description="HTTP timeout in seconds")


class RAGConfig(BaseModel):
    """Configuration for RAG retrieval."""

    servers: list[RAGServerConfig] = Field(..., description="List of RAG servers to query")
    max_context_chars: int = Field(50000, description="Maximum characters of retrieved content to inject")


def resolve_rag_config(value: str) -> RAGConfig:
    """Resolve a rag CLI value into a RAGConfig.

    Args:
        value: Either a URL (http:// or https://) for single-server
            config with defaults, or a path to a YAML config file.

    Returns:
        Validated RAGConfig.

    Raises:
        FileNotFoundError: If the value is treated as a path but doesn't exist.
        ValueError: If the YAML file is invalid.
    """
    if value.startswith(("http://", "https://")):
        return RAGConfig(servers=[RAGServerConfig(url=value)])

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"RAG config file not found: {value}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"RAG config file must contain a YAML mapping, got {type(data).__name__}")

    return RAGConfig(**data)


def retrieve_reference_docs(query: str, rag_config: RAGConfig) -> str | None:
    """Query configured RAG servers for reference documentation.

    Args:
        query: The search query to send to the RAG endpoint.
        rag_config: RAG configuration specifying which servers to query.

    Returns:
        Concatenated reference documentation text, or None if no results or all queries failed.
    """
    results: list[str] = []

    for server in rag_config.servers:
        try:
            result = _query_rag_server(server, query)
            if result:
                results.append(result)
        except Exception:
            logger.exception("Failed to query RAG server %s", server.url)

    if not results:
        return None

    combined = "\n\n".join(results)
    if len(combined) > rag_config.max_context_chars:
        combined = combined[: rag_config.max_context_chars]
        logger.info("Truncated RAG context to %d characters", rag_config.max_context_chars)

    return combined


def _query_rag_server(config: RAGServerConfig, query: str) -> str | None:
    """Query a RAG server's HTTP endpoint for reference docs.

    Expects the server to accept POST with {"message": query}
    and return {"context": "..."}.

    Args:
        config: RAG server configuration.
        query: The search query to send.

    Returns:
        Retrieved text content, or None if the call failed or returned no content.
    """
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    response = requests.post(
        config.url,
        headers=headers,
        json={"message": query},
        timeout=config.timeout,
    )
    response.raise_for_status()

    data = response.json()
    context = data.get("context")
    if not context:
        logger.warning("RAG server %s returned empty context", config.url)
        return None

    return context
