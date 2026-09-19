"""Live integration tests for the simple OpenAI and Gemini clients.

These call the real APIs and consume tokens. They are skipped automatically
when the matching API key is not present:

- OpenAI : OPENAI_KEY (preferred) / OPENAI_API_KEY
- Gemini : GEMINI_API_KEY

Run with:
    .venv\\Scripts\\python -m pytest src/tests/test_live_clients.py -v -m integration
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from client import ClientFactory
from llm.response import StreamEventType

load_dotenv()

pytestmark = pytest.mark.integration

MESSAGES = [{"role": "user", "content": "Reply with exactly the word: pong"}]

needs_openai = pytest.mark.skipif(
    not (os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="OPENAI_KEY not set in .env",
)
needs_gemini = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set in .env",
)


async def _collect(client, messages, stream):
    """Run one completion and collapse the events into plain values."""
    text = ""
    finish_reason = None
    usage = None
    error = None

    async for event in client.chat_completion(messages, stream=stream):
        if event.type is StreamEventType.TEXT_DELTA and event.text_delta:
            text += event.text_delta.content
        elif event.type is StreamEventType.MESSAGE_COMPLETE:
            finish_reason = event.finish_reason
            usage = event.usage
        elif event.type is StreamEventType.ERROR:
            error = event.error

    return text, finish_reason, usage, error


async def _assert_completion_ok(provider: str, stream: bool) -> None:
    client = ClientFactory.create_client(provider)
    async with client:
        text, finish_reason, usage, error = await _collect(client, MESSAGES, stream)

    assert error is None, error
    assert text.strip(), "expected a non-empty response"
    assert finish_reason == "stop", finish_reason

    # Both providers report usage for non-streaming. For streaming, OpenAI
    # omits usage unless stream_options.include_usage is requested, so only
    # require it on the non-streaming path.
    if not stream:
        assert usage is not None and usage.total_tokens > 0, usage


# --- OpenAI ---------------------------------------------------------------


@needs_openai
async def test_openai_non_streaming_live():
    await _assert_completion_ok("openai", stream=False)


@needs_openai
async def test_openai_streaming_live():
    await _assert_completion_ok("openai", stream=True)


# --- Gemini ---------------------------------------------------------------


@needs_gemini
async def test_gemini_non_streaming_live():
    await _assert_completion_ok("gemini", stream=False)


@needs_gemini
async def test_gemini_streaming_live():
    await _assert_completion_ok("gemini", stream=True)
