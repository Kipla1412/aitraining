"""Unit tests for the simple, tool-free Agent (client backend)."""
from agent.agent import Agent
from agent.event import AgentEventType
from llm.response import StreamEvent, StreamEventType, TextDelta, TokenUsage


class FakeClient:
    """Minimal stand-in for client.llmclient.LLMClient."""

    def __init__(self, reply: str = "hi there", usage: TokenUsage | None = None):
        self.reply = reply
        self.usage = usage
        self.calls: list[list[dict]] = []

    async def chat_completion(self, messages, stream=True):
        self.calls.append(list(messages))
        for word in self.reply.split(" "):
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(word + " ")
            )
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason="stop",
            usage=self.usage,
        )


class ErrorClient:
    def __init__(self, error: str = "boom"):
        self.error = error

    async def chat_completion(self, messages, stream=True):
        yield StreamEvent(type=StreamEventType.ERROR, error=self.error)


async def _collect(agent: Agent, message: str):
    return [event async for event in agent.run(message)]


async def test_event_sequence_and_content():
    client = FakeClient(reply="hello world", usage=TokenUsage(1, 2, 3))
    agent = Agent(client)

    events = await _collect(agent, "hey")

    types = [e.type for e in events]
    assert types[0] is AgentEventType.AGENT_START
    assert types[-1] is AgentEventType.AGENT_END
    assert AgentEventType.TEXT_COMPLETE in types

    text = "".join(
        e.data["content"] for e in events if e.type is AgentEventType.TEXT_DELTA
    )
    assert text == "hello world "
    assert events[-1].data["response"] == text
    assert events[-1].data["finish_reason"] == "stop"
    assert events[-1].data["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
        "cached_tokens": 0,
    }


async def test_history_is_plain_dicts_and_persists():
    client = FakeClient(reply="ok")
    agent = Agent(client, system_prompt="be terse")

    await _collect(agent, "first")
    await _collect(agent, "second")

    assert len(client.calls) == 2
    assert client.calls[1] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok "},
        {"role": "user", "content": "second"},
    ]
    assert agent.messages == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok "},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ok "},
    ]


async def test_error_event_is_surfaced():
    agent = Agent(ErrorClient("kaboom"))
    events = await _collect(agent, "hi")

    assert events[1].type is AgentEventType.AGENT_ERROR
    assert events[1].data["error"] == "kaboom"
    assert all(e.type is not AgentEventType.AGENT_END for e in events)


async def test_reset_clears_history_but_keeps_system_prompt():
    agent = Agent(FakeClient(), system_prompt="sys")

    await _collect(agent, "hello")
    assert agent.messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there "},
    ]

    agent.reset()
    assert agent.messages == [{"role": "system", "content": "sys"}]
