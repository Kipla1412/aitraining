"""OpenAI-compatible chat client (text only, no tool calling).

Connects to the LLM and normalizes streaming/non-streaming responses into the
project's common ``StreamEvent`` objects. It only ever emits:

- ``TEXT_DELTA``     for streamed text
- ``MESSAGE_COMPLETE`` with finish_reason and usage
- ``ERROR``          on failures

There is intentionally no tool / function-calling support.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator, Dict, List

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    RateLimitError,
)

from llm.response import StreamEvent, StreamEventType, TextDelta, TokenUsage

load_dotenv()


class LLMClient:
    """Async OpenAI-compatible client yielding common ``StreamEvent`` objects."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        # Fall back to env vars when not provided.
        self._api_key = (
            api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._model = (
            model
            or os.getenv("OPENAI_MODEL")
            or os.getenv("OPENAI_DEFAULT_MODEL")
            or "gpt-4o-mini"
        )
        self._max_retries = max_retries

        self._client: AsyncOpenAI | None = None

    # client lifecycle

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "LLMClient":
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

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }

        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    async for event in self._non_stream_response(client, kwargs):
                        yield event
                return

            except RateLimitError as e:
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"Rate limit exceeded: {e}",
                )
                return

            except APIConnectionError as e:
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"Connection error: {e}",
                )
                return

            except APIError as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"API error: {e}",
                )
                return

    # streaming response

    async def _stream_response(
        self,
        client: AsyncOpenAI,
        kwargs: Dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)

        finish_reason: str | None = None
        usage: TokenUsage | None = None

        try:
            async for chunk in response:
                if getattr(chunk, "usage", None):
                    cached_tokens = getattr(
                        getattr(chunk.usage, "prompt_tokens_details", None),
                        "cached_tokens",
                        0,
                    )
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cached_tokens=cached_tokens or 0,
                    )

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                content = choice.delta.content
                if content:
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        text_delta=TextDelta(content),
                    )
        finally:
            # Release the underlying HTTP stream deterministically.
            await response.close()

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )

    # non-streaming response
   
    async def _non_stream_response(
        self,
        client: AsyncOpenAI,
        kwargs: Dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        if message.content:
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text_delta=TextDelta(message.content),
            )

        usage = None
        if response.usage:
            cached_tokens = getattr(
                getattr(response.usage, "prompt_tokens_details", None),
                "cached_tokens",
                0,
            )
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=cached_tokens or 0,
            )

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
