"""OpenAI chat provider.

Responsible only for API communication: client lifecycle, request/tool
building, streaming, retries and error normalization. Response parsing is
delegated to :class:`~llm.openai.parser.OpenAIStreamParser`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Sequence

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llm.base import ChatMessage, ToolSpec, retry_on_transient
from llm.openai.parser import OpenAIStreamParser
from llm.response import StreamEvent, StreamEventType


@dataclass
class OpenAIProviderConfig:
    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    timeout: float = 120.0
    max_retries: int = 3
    extra_body: dict[str, Any] = field(default_factory=dict)


class OpenAIProvider:
    """Minimal OpenAI-compatible chat provider (OpenAI, Groq, OpenRouter...)."""

    def __init__(self, config: OpenAIProviderConfig) -> None:
        self._config = config
        self._client: AsyncOpenAI | None = None
        self._parser = OpenAIStreamParser()

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------
    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                max_retries=0,  # Retries are handled in generate().
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "OpenAIProvider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_generate(messages, tools):
                        yield event
                else:
                    async for event in self._non_stream_generate(messages, tools):
                        yield event
                return
            except (APIConnectionError, APIStatusError) as exc:
                if not retry_on_transient(exc) or attempt >= self._config.max_retries:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"openai: {exc}",
                    )
                    return
                last_error = exc

        yield StreamEvent(
            type=StreamEventType.ERROR,
            error=f"openai: {last_error}",
        )

    async def _stream_generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        stream: AsyncIterator[ChatCompletionChunk] = await self.client.chat.completions.create(
            model=self._config.model,
            messages=[m.to_openai_message() for m in messages],
            tools=self._build_tools(tools),
            stream=True,
            **self._config.extra_body,
        )

        try:
            async with stream:
                async for chunk in stream:
                    for event in self._parser.stream_chunk(chunk):
                        yield event
        finally:
            await stream.close()

    async def _non_stream_generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        response: ChatCompletion = await self.client.chat.completions.create(
            model=self._config.model,
            messages=[m.to_openai_message() for m in messages],
            tools=self._build_tools(tools),
            stream=False,
            **self._config.extra_body,
        )
        for event in self._parser.parse_response(response):
            yield event

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------
    @staticmethod
    def _build_tools(tools: Sequence[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]
