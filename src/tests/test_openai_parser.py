"""Unit tests for OpenAIStreamParser -> common StreamEvents."""
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as CompletionChoice, CompletionUsage
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall

from llm.openai.parser import OpenAIStreamParser
from llm.response import StreamEventType, TokenUsage, ToolCall


def _chunk(**kwargs):
    defaults = dict(id="chunk_1", model="gpt-4o-mini", created=1, object="chat.completion.chunk")
    defaults.update(kwargs)
    return ChatCompletionChunk(**defaults)


def _choice(delta, finish_reason=None):
    return ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)


def _delta(**kwargs):
    return ChoiceDelta(**kwargs)


def _tool_call(index=0, id=None, name=None, arguments=None):
    fn = None
    if name or arguments:
        fn = {"name": name, "arguments": arguments} if name else {"arguments": arguments}
    return ChoiceDeltaToolCall(index=index, id=id, type="function", function=fn)

# ---------------------------------------------------------------------------
# Text streaming
# ---------------------------------------------------------------------------
def test_text_streaming_emits_text_delta_events():
    parser = OpenAIStreamParser()
    chunk = _chunk(choices=[_choice(_delta(content="hello"))])
    chunk2 = _chunk(choices=[_choice(_delta(content=" world"))])
    events = parser.stream_chunk(chunk) + parser.stream_chunk(chunk2)

    text = [e for e in events if e.type is StreamEventType.TEXT_DELTA]
    assert [e.text_delta.content for e in text] == ["hello", " world"]


# ---------------------------------------------------------------------------
# Tool-call streaming
# ---------------------------------------------------------------------------
def test_tool_call_streaming_emits_start_delta_complete():
    parser = OpenAIStreamParser()
    # First chunk: id + name arrives (start).
    c1 = _chunk(choices=[_choice(_delta(tool_calls=[_tool_call(id="call_1", name="get_weather")]))])
    # Middle chunks: argument fragments.
    c2 = _chunk(choices=[_choice(_delta(tool_calls=[_tool_call(arguments='{"city":')]))])
    c3 = _chunk(choices=[_choice(_delta(tool_calls=[_tool_call(arguments='"Paris"}')]))])
    # Finish chunk.
    c4 = _chunk(choices=[_choice(_delta(), finish_reason="tool_calls")])

    events = []
    for c in (c1, c2, c3, c4):
        events += parser.stream_chunk(c)

    types = [e.type for e in events]
    assert types == [
        StreamEventType.TOOL_CALL_START,
        StreamEventType.TOOL_CALL_DELTA,
        StreamEventType.TOOL_CALL_DELTA,
        StreamEventType.TOOL_CALL_COMPLETE,
        StreamEventType.MESSAGE_COMPLETE,
    ]

    complete = events[3].tool_call
    assert complete == ToolCall(
        call_id="call_1",
        name="get_weather",
        arguments={"city": "Paris"},
        raw_arguments='{"city":"Paris"}',
    )


def test_tool_arguments_split_across_chunks_are_accumulated():
    parser = OpenAIStreamParser()
    events = []
    events += parser.stream_chunk(
        _chunk(choices=[_choice(_delta(tool_calls=[_tool_call(id="c1", name="f")]))])
    )
    for frag in ['{"a":', "1,", '"b":', '[1,', "2,3]}"]:
        events += parser.stream_chunk(
            _chunk(choices=[_choice(_delta(tool_calls=[_tool_call(arguments=frag)]))])
        )
    events += parser.stream_chunk(_chunk(choices=[_choice(_delta(), finish_reason="tool_calls")]))

    deltas = [e.tool_call_delta for e in events if e.type is StreamEventType.TOOL_CALL_DELTA]
    assert "".join(d.arguments_delta for d in deltas) == '{"a":1,"b":[1,2,3]}'

    complete = [e for e in events if e.type is StreamEventType.TOOL_CALL_COMPLETE][0].tool_call
    assert complete.arguments == {"a": 1, "b": [1, 2, 3]}


def test_multiple_tool_calls_in_one_chunk():
    parser = OpenAIStreamParser()
    chunk = _chunk(
        choices=[
            _choice(
                _delta(
                    tool_calls=[
                        _tool_call(index=0, id="a", name="f1", arguments="{}"),
                        _tool_call(index=1, id="b", name="f2", arguments="{}"),
                    ]
                )
            )
        ]
    )
    events = parser.stream_chunk(chunk)
    events += parser.stream_chunk(_chunk(choices=[_choice(_delta(), finish_reason="tool_calls")]))

    starts = [e for e in events if e.type is StreamEventType.TOOL_CALL_START]
    completes = [e for e in events if e.type is StreamEventType.TOOL_CALL_COMPLETE]

    assert [s.tool_call_delta.call_id for s in starts] == ["a", "b"]
    assert [c.tool_call.call_id for c in completes] == ["a", "b"]
    assert {c.tool_call.name for c in completes} == {"f1", "f2"}


# ---------------------------------------------------------------------------
# Usage + finish reason
# ---------------------------------------------------------------------------
def test_usage_events():
    parser = OpenAIStreamParser()
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    chunk = _chunk(choices=[_choice(_delta())], usage=usage)
    events = parser.stream_chunk(chunk)

    usage_events = [e for e in events if e.usage is not None]
    assert len(usage_events) == 1
    assert usage_events[0].usage == TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)


def test_finish_reason_mapping():
    parser = OpenAIStreamParser()
    for raw, expected in [("stop", "stop"), ("length", "length"), ("tool_calls", "tool_calls")]:
        events = parser.stream_chunk(_chunk(choices=[_choice(_delta(), finish_reason=raw)]))
        completes = [e for e in events if e.type is StreamEventType.MESSAGE_COMPLETE]
        assert completes and completes[0].finish_reason == expected


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------
def test_non_streaming_response():
    parser = OpenAIStreamParser()
    message = ChatCompletionMessage(
        role="assistant",
        content="Let me check.",
        tool_calls=[
            {
                "id": "call_9",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"London"}'},
            }
        ],
    )
    choice = CompletionChoice(index=0, message=message, finish_reason="tool_calls")
    usage = CompletionUsage(prompt_tokens=5, completion_tokens=15, total_tokens=20)
    response = ChatCompletion(
        id="resp_1",
        model="gpt-4o-mini",
        created=1,
        object="chat.completion",
        choices=[choice],
        usage=usage,
    )

    events = parser.parse_response(response)

    assert events[0].type is StreamEventType.TEXT_DELTA
    assert events[0].text_delta.content == "Let me check."

    starts = [e for e in events if e.type is StreamEventType.TOOL_CALL_START]
    completes = [e for e in events if e.type is StreamEventType.TOOL_CALL_COMPLETE]
    assert len(starts) == 1 and starts[0].tool_call_delta.call_id == "call_9"
    assert completes[0].tool_call.arguments == {"city": "London"}

    final = events[-1]
    assert final.type is StreamEventType.MESSAGE_COMPLETE
    assert final.finish_reason == "tool_calls"
    assert final.usage == TokenUsage(prompt_tokens=5, completion_tokens=15, total_tokens=20)
