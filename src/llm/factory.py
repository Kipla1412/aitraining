"""Provider registry/factory: build a provider from a name + config."""
from __future__ import annotations

from typing import Any

from llm.base import ChatMessage, ToolResultMessage, ToolSpec
from llm.gemini.provider import GeminiProvider, GeminiProviderConfig
from llm.openai.provider import OpenAIProvider, OpenAIProviderConfig
from llm.response import StreamEvent

_PROVIDERS: dict[str, Any] = {
    "openai": (OpenAIProvider, OpenAIProviderConfig),
    "gemini": (GeminiProvider, GeminiProviderConfig),
}


class LLMClientFactory:
    """Create providers and their configs by name.

    The factory only ever returns a provider (``generate`` -> ``StreamEvent``
    async generator). The caller remains provider-neutral.
    """

    @staticmethod
    def create_provider(provider: str, config: Any) -> Any:
        try:
            provider_cls, _config_cls = _PROVIDERS[provider]
        except KeyError as exc:
            raise ValueError(
                f"unsupported llm provider: {provider!r} "
                f"(available: {', '.join(sorted(_PROVIDERS))})"
            ) from exc
        return provider_cls(config)

    @staticmethod
    def create_config(provider: str, **kwargs: Any) -> Any:
        try:
            _provider_cls, config_cls = _PROVIDERS[provider]
        except KeyError as exc:
            raise ValueError(
                f"unsupported llm provider: {provider!r} "
                f"(available: {', '.join(sorted(_PROVIDERS))})"
            ) from exc
        return config_cls(**kwargs)


__all__ = [
    "LLMClientFactory",
    "ChatMessage",
    "ToolResultMessage",
    "ToolSpec",
    "StreamEvent",
]
