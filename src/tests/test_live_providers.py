"""Live integration tests against the real OpenAI / Gemini APIs.

These call real endpoints and consume tokens. They are skipped automatically
when the matching API key is not present:

- OpenAI : OPENAI_API_KEY (or OPENAI_KEY)
- Gemini : GEMINI_API_KEY

Run with:  .venv\\Scripts\\python -m pytest src/tests/test_live_providers.py -v
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from llm.agent import AgentRuntime
from llm.base import ChatMessage, ToolSpec
from llm.factory import LLMClientFactory
from llm.response import StreamEvent, StreamEventType

load_dotenv()

pytestmark = pytest.mark.integration

OPENAI_KEY = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")

needs_openai = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set in .env")
needs_gemini = pytest.mark.skipif(not GEMINI_KEY, reason="GEMINI_API_KEY not set in .env")

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. Paris"},
        },
        "required": ["city"],
    },
)


def _openai_provider(**extra):
    return LLMClientFactory.create_provider(
        "openai",
        LLMClientFactory.create_config(
            "openai",
            api_key=OPENAI_KEY,
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=OPENAI_MODEL,
            extra_body=extra,
        ),
    )


def _gemini_provider(**extra_config):
    return LLMClientFactory.create_provider(
        "gemini",
        LLMClientFactory.create_config(
            "gemini",
            api_key=GEMINI_KEY,
            model=GEMINI_MODEL,
            extra_config=extra_config,
        ),
    )


def _consume(events) -> dict:
    """Collapse a StreamEvent stream into plain aggregates."""
    text: list[str] = []
    thinking: list[str] = []
    tool_calls = []
    usage = None
    finish_reason = None
    for e in events:
        if e.type is StreamEventType.ERROR:
            raise AssertionError(f"provider error event: {e.error}")
        elif e.type is StreamEventType.TEXT_DELTA and e.text_delta:
            text.append(e.text_delta.content)
        elif e.type is StreamEventType.THINKING_DELTA and e.thinking_delta:
            thinking.append(e.thinking_delta.content)
        elif e.type is StreamEventType.TOOL_CALL_COMPLETE and e.tool_call:
            tool_calls.append(e.tool_call)
        elif e.type is StreamEventType.MESSAGE_COMPLETE:
            if e.finish_reason:
                finish_reason = e.finish_reason
            if e.usage:
                usage = e.usage
    return {
        "text": "".join(text),
        "thinking": "".join(thinking),
        "tool_calls": tool_calls,
        "usage": usage,
        "finish_reason": finish_reason,
    }


async def _weather(call) -> str:
    city = call.arguments.get("city", "unknown")
    return f"Sunny, 24C in {city}."


# ===========================================================================
# OpenAI (live)
# ===========================================================================
@needs_openai
async def test_openai_text_streaming_live():
    provider = _openai_provider(stream_options={"include_usage": True})
    async with provider:
        out = _consume(
            [
                e
                async for e in provider.generate(
                    messages=[ChatMessage(role="user", content="Say 'hello' once.")],
                    stream=True,
                )
            ]
        )

    assert "hello" in out["text"].lower()
    assert out["finish_reason"] == "stop"
    assert out["usage"] is not None and out["usage"].total_tokens > 0


@needs_openai
async def test_openai_non_streaming_live():
    provider = _openai_provider()
    async with provider:
        out = _consume(
            [
                e
                async for e in provider.generate(
                    messages=[ChatMessage(role="user", content="What is 2+2? Answer with a number.")],
                    stream=False,
                )
            ]
        )

    assert out["text"].strip().startswith("4")
    assert out["usage"] is not None and out["usage"].total_tokens > 0


@needs_openai
async def test_openai_tool_loop_live():
    """Real streaming tool call: the model must call get_weather, the runtime
    executes it, then the model answers with the tool result."""
    provider = _openai_provider()
    async with provider:
        runtime = AgentRuntime(
            provider,
            system_prompt=(
                "You MUST call get_weather to answer weather questions. "
                "After the tool returns, reply using only its output."
            ),
            max_turns=3,
        )
        result = await runtime.run(
            "What is the weather in Paris?",
            tools=[WEATHER_TOOL],
            executor=_weather,
        )

    assert result.error is None, result.error
    assert result.tool_calls, "expected at least one tool call"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments.get("city")
    assert "sunny" in result.content.lower() or "24c" in result.content.lower(), result.content
    assert result.finish_reason == "stop"


# ===========================================================================
# Gemini (live)
# ===========================================================================
@needs_gemini
async def test_gemini_text_streaming_live():
    provider = _gemini_provider()
    async with provider:
        out = _consume(
            [
                e
                async for e in provider.generate(
                    messages=[ChatMessage(role="user", content="Say 'hello' once.")],
                    stream=True,
                )
            ]
        )

    assert "hello" in out["text"].lower()
    assert out["usage"] is not None and out["usage"].total_tokens > 0


@needs_gemini
async def test_gemini_non_streaming_live():
    provider = _gemini_provider()
    async with provider:
        out = _consume(
            [
                e
                async for e in provider.generate(
                    messages=[ChatMessage(role="user", content="What is 2+2? Answer with a number.")],
                    stream=False,
                )
            ]
        )

    assert out["text"].strip().startswith("4")
    assert out["usage"] is not None and out["usage"].total_tokens > 0


@needs_gemini
async def test_gemini_tool_loop_live():
    provider = _gemini_provider()
    async with provider:
        runtime = AgentRuntime(
            provider,
            system_prompt=(
                "You MUST call get_weather to answer weather questions. "
                "After the tool returns, reply using only its output."
            ),
            max_turns=3,
        )
        result = await runtime.run(
            "What is the weather in Paris?",
            tools=[WEATHER_TOOL],
            executor=_weather,
        )

    assert result.error is None, result.error
    assert result.tool_calls, "expected at least one tool call"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments.get("city")
    assert "sunny" in result.content.lower() or "24c" in result.content.lower(), result.content
