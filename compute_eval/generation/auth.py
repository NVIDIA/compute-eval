import os
from collections.abc import Callable

TokenProvider = Callable[[str], str | None]

_TOKEN_PROVIDERS: list[TokenProvider] = []


def register_token_provider(provider: TokenProvider) -> None:
    _TOKEN_PROVIDERS.insert(0, provider)


def get_api_token(base_url: str, env_var: str) -> str | None:
    for provider in _TOKEN_PROVIDERS:
        if token := provider(base_url):
            return token
    return os.getenv(env_var)
