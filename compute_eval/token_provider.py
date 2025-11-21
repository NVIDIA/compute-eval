import os
from abc import ABC, abstractmethod


class TokenProvider(ABC):
    """Abstract base class for token providers."""

    @abstractmethod
    def get_token(self, base_url: str) -> str | None:
        """Get authentication token for the given base URL."""
        pass

    @abstractmethod
    def handles_url(self, base_url: str) -> bool:
        """Check if this provider handles the given URL."""
        pass


class EnvTokenProvider(TokenProvider):
    """Default provider that reads from environment variables."""

    def __init__(self, env_var: str):
        self.env_var = env_var

    def get_token(self, base_url: str) -> str | None:
        return os.getenv(self.env_var)

    def handles_url(self, base_url: str) -> bool:
        # Handles all URLs by default
        return True


# Global registry for token providers
_token_providers: list[TokenProvider] = []


def register_token_provider(provider: TokenProvider):
    """Register a custom token provider (checked in order of registration)."""
    _token_providers.insert(0, provider)  # Prepend so custom providers take precedence


def get_token_for_url(base_url: str, default_env_var: str) -> str | None:
    """Get token from the first provider that handles this URL."""
    for provider in _token_providers:
        if provider.handles_url(base_url):
            token = provider.get_token(base_url)
            if token:
                return token

    # Fallback to environment variable
    return os.getenv(default_env_var)
