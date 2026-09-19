"""Gemini chat client (text only, no tool calling).

Mirrors the simple OpenAI client in ``client.llmclient``: same lifecycle, same
``chat_completion`` interface, same ``StreamEvent`` output -- only the
provider-specific API logic is Gemini's.

It only ever emits:

- ``TEXT_DELTA``       for streamed text
- ``MESSAGE_COMPLETE`` with finish_reason and usage
- ``ERROR``            on failures
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from llm.response import StreamEvent, StreamEventType, TextDelta, TokenUsage

load_dotenv()

# Gemini finish reasons -> common framework finish reasons.
_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "SPII": "content_filter",
}

# HTTP statuses worth retrying (rate limit / transient server errors).
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class GeminiLLMClient:
    """Async Gemini client yielding common ``StreamEvent`` objects."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        # Fall back to env vars when not provided.
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._model = (
            model
            or os.getenv("GEMINI_MODEL")
            or os.getenv("GEMINI_DEFAULT_MODEL")
            or "gemini-3.6-flash"
        )
        self._max_retries = max_retries

        self._client: genai.Client | None = None

    # client lifecycle

    def get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def close(self) -> None:
        # google-genai's Client.close() is synchronous.
        if self._client is not None:
            self._client.close()
            self._client = None

    async def __aenter__(self) -> "GeminiLLMClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # public API

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield ``StreamEvent``s for a plain chat completion."""
        client = self.get_client()
        contents, config = self._build_request(messages)

        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_response(client, contents, config):
                        yield event
                else:
                    async for event in self._non_stream_response(client, contents, config):
                        yield event
                return

            except errors.APIError as e:
                # 429 / 5xx -> retry, other API errors fail fast.
                if self._is_retryable(e) and attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"Gemini error: {e}",
                )
                return

            except Exception as e:  # connection / transport errors
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"Connection error: {e}",
                )
                return

    # streaming response

    async def _stream_response(
        self,
        client: genai.Client,
        contents: List[types.Content],
        config: types.GenerateContentConfig,
    ) -> AsyncGenerator[StreamEvent, None]:
        stream = await client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        )

        finish_reason: str | None = None
        usage: TokenUsage | None = None

        async for chunk in stream:
            if chunk.candidates:
                candidate = chunk.candidates[0]

                reason = self._to_finish_reason(candidate.finish_reason)
                if reason:
                    finish_reason = reason

                content = self._extract_text(candidate)
                if content:
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        text_delta=TextDelta(content),
                    )

            if chunk.usage_metadata is not None:
                usage = self._to_usage(chunk.usage_metadata)

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )

    # non-streaming response

    async def _non_stream_response(
        self,
        client: genai.Client,
        contents: List[types.Content],
        config: types.GenerateContentConfig,
    ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        finish_reason: str | None = None
        usage: TokenUsage | None = None

        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = self._to_finish_reason(candidate.finish_reason)

            content = self._extract_text(candidate)
            if content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(content),
                )

        if response.usage_metadata is not None:
            usage = self._to_usage(response.usage_metadata)

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )

    # Gemini request building / response helpers

    def _build_request(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[types.Content], types.GenerateContentConfig]:
        """Convert OpenAI-style dict messages into Gemini contents + config.

        Gemini has no "system" role in ``contents``; system text goes into
        ``config.system_instruction`` instead.
        """
        contents: List[types.Content] = []
        system_parts: List[str] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content") or ""

            if role == "system":
                system_parts.append(content)
                continue

            contents.append(
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[types.Part(text=content)],
                )
            )

        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(system_parts) if system_parts else None,
            # This client is text-only; disable Gemini's automatic function
            # calling so it never tries to invoke tools.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        return contents, config

    @staticmethod
    def _extract_text(candidate: Any) -> str:
        """Join the text of a candidate's non-thought parts."""
        if candidate.content is None or not candidate.content.parts:
            return ""

        chunks: List[str] = []
        for part in candidate.content.parts:
            if getattr(part, "text", None) and not getattr(part, "thought", False):
                chunks.append(part.text)
        return "".join(chunks)

    @staticmethod
    def _to_finish_reason(reason: Any) -> str | None:
        if reason is None:
            return None
        name = getattr(reason, "name", None) or str(reason)
        if name == "FINISH_REASON_UNSPECIFIED":
            return None
        return _FINISH_REASON_MAP.get(name, name.lower())

    @staticmethod
    def _to_usage(metadata: Any) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(metadata, "candidates_token_count", 0) or 0,
            total_tokens=getattr(metadata, "total_token_count", 0) or 0,
            cached_tokens=getattr(metadata, "cached_content_token_count", 0) or 0,
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        return isinstance(code, int) and code in _RETRYABLE_STATUS
