"""Live integration test for the tool-free Agent (client backend).

Calls the real OpenAI API (via env config). Skipped when no key is present:
    OPENAI_KEY (preferred) / OPENAI_API_KEY

Run with:
    .venv\\Scripts\\python -m pytest src/tests/test_live_agent_loop.py -v -m integration
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from agent.agent import Agent
from agent.event import AgentEventType
from client.llmclient import LLMClient

load_dotenv()

pytestmark = pytest.mark.integration

KEY = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
needs_key = pytest.mark.skipif(not KEY, reason="OPENAI_KEY not set in .env")


async def _collect(agent: Agent, message: str):
    return [event async for event in agent.run(message)]


@needs_key
async def test_agent_loop_streams_text_live():
    client = LLMClient()
    async with client:
        agent = Agent(client, system_prompt="You are terse.")
        events = await _collect(agent, "Say hello in exactly three words.")

    types = [e.type for e in events]
    assert types[0] is AgentEventType.AGENT_START
    assert types[-1] is AgentEventType.AGENT_END
    assert AgentEventType.TEXT_COMPLETE in types

    text = "".join(
        e.data["content"]
        for e in events
        if e.type is AgentEventType.TEXT_DELTA
    )
    assert text.strip(), "expected streamed text"

    end = events[-1].data
    assert end["response"] == text
    assert end["finish_reason"] == "stop"


@needs_key
async def test_agent_loop_multi_turn_history_live():
    client = LLMClient()
    async with client:
        agent = Agent(
            client,
            system_prompt="Remember facts the user tells you.",
        )
        await _collect(agent, "My favourite number is 7. Reply with just OK.")
        second = await _collect(
            agent, "What is my favourite number? Reply with just the digit."
        )

    # system + user + assistant + user + assistant
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"]

    answer = second[-1].data["response"]
    assert "7" in answer, answer


@needs_key
async def test_agent_loop_reset_live():
    client = LLMClient()
    async with client:
        agent = Agent(client, system_prompt="sys")
        await _collect(agent, "hi")
        assert len(agent.messages) == 3  # system + user + assistant

        agent.reset()
        assert [m["role"] for m in agent.messages] == ["system"]
