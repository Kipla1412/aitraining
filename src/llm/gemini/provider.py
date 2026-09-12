"""Gemini chat provider.

Responsible only for API communication: client lifecycle, request/tool
building, streaming, retries and error normalization. Response parsing is
delegated to :class:`~llm.gemini.parser.GeminiStreamParser`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Sequence

from google import genai
from google.genai import types as genai_types

from llm.base import ChatMessage, ToolSpec, retry_on_transient
from llm.gemini.parser import GeminiStreamParser
from llm.response import StreamEvent, StreamEventType


@dataclass
class GeminiProviderConfig:
    api_key: str | None = None
    model: str = "gemini-2.0-flash"
    max_retries: int = 3
    extra_config: dict[str, Any] = field(default_factory=dict)


class GeminiProvider:
    """Google Gemini chat provider built on the ``google-genai`` SDK."""

    def __init__(self, config: GeminiProviderConfig) -> None:
        self._config = config
        self._client: genai.Client | None = None
        self._parser = GeminiStreamParser()

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------
    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self._config.api_key)
        return self._client

    async def close(self) -> None:
        # google-genai's Client.close() is synchronous and tears down the
        # underlying httpx resources.
        if self._client is not None:
            self._client.close()
            self._client = None

    async def __aenter__(self) -> "GeminiProvider":
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
            except Exception as exc:  # noqa: BLE001 - normalize any API error
                if not retry_on_transient(exc) or attempt >= self._config.max_retries:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"gemini: {exc}",
                    )
                    return
                last_error = exc

        yield StreamEvent(
            type=StreamEventType.ERROR,
            error=f"gemini: {last_error}",
        )

    async def _stream_generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        # google-genai exposes async generate_content_stream directly; its
        # sync counterpart is an iterator. The async call returns an async
        # iterator of chunk responses.
        system_instruction, contents = self._split_messages(messages)
        stream = await self.client.aio.models.generate_content_stream(
            model=self._config.model,
            contents=contents,
            config=self._build_config(tools, system_instruction),
        )
        async for chunk in stream:
            for event in self._parser.stream_chunk(chunk):
                yield event

    async def _non_stream_generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        system_instruction, contents = self._split_messages(messages)
        response = await self.client.aio.models.generate_content(
            model=self._config.model,
            contents=contents,
            config=self._build_config(tools, system_instruction),
        )
        for event in self._parser.parse_response(response):
            yield event

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------
    @staticmethod
    def _split_messages(
        messages: Sequence[ChatMessage],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Separate system prompts from contents.

        Gemini's generate_content does not accept a "system" role in
        ``contents``; system text must be provided via
        ``config.system_instruction`` instead.
        """
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            if getattr(message, "normalized_role", None) == "system":
                content = getattr(message, "content", None)
                if content:
                    system_parts.append(content)
                continue
            contents.append(message.to_gemini_content())

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    def _build_config(
        self,
        tools: Sequence[ToolSpec] | None,
        system_instruction: str | None = None,
    ) -> genai_types.GenerateContentConfig:
        merged: dict[str, Any] = dict(self._config.extra_config)
        if system_instruction:
            merged["system_instruction"] = system_instruction
        if tools:
            merged["tools"] = [
                genai_types.Tool(
                    function_declarations=[
                        genai_types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description or "",
                            parameters=tool.parameters,
                        )
                        for tool in tools
                    ]
                )
            ]
        return genai_types.GenerateContentConfig(**merged)
