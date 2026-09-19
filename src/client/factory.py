"""Simple client factory: build an OpenAI or Gemini client by name.

Both clients expose the same ``chat_completion(messages, stream=True)``
interface, so callers can stay provider-neutral after construction.

Routing is explicit:

- ``create_client("gemini")``  -> pick a provider in code
- ``create_default()``         -> pick a provider from the ``LLM_PROVIDER`` env var
"""
from __future__ import annotations

import os
from typing import Any, Union

from client.gemini import GeminiLLMClient
from client.llmclient import LLMClient

CLIENTS = {
    "openai": LLMClient,
    "gemini": GeminiLLMClient,
    
}

ClientType = Union[LLMClient, GeminiLLMClient]


class ClientFactory:
    """Create a chat client by provider name."""

    @staticmethod
    def create_client(provider: str = "openai", **kwargs: Any) -> ClientType:
        """Return the client for ``provider`` (``"openai"`` or ``"gemini"``).

        Keyword arguments are forwarded to the client constructor
        (e.g. ``api_key``, ``model``, ``max_retries``; ``base_url`` for OpenAI).
        """
        try:
            client_cls = CLIENTS[provider]
        except KeyError as exc:
            raise ValueError(
                f"unsupported llm provider: {provider!r} "
                f"(available: {', '.join(sorted(CLIENTS))})"
            ) from exc
        return client_cls(**kwargs)

    @staticmethod
    def create_default(**kwargs: Any) -> ClientType:
        """Create the client named by the ``LLM_PROVIDER`` env var.

        Defaults to ``"openai"`` when the variable is unset.
        """
        provider = os.getenv("LLM_PROVIDER", "openai")
        return ClientFactory.create_client(provider, **kwargs)


__all__ = ["ClientFactory", "ClientType", "CLIENTS"]
