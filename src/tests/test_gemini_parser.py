"""Unit tests for GeminiStreamParser -> common StreamEvents."""
from google.genai import types as t

from llm.gemini.parser import GeminiStreamParser
from llm.response import (
    StreamEvent,
    StreamEventType,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
)


def _chunk(parts=None, finish_reason=None, usage=None):
    candidate = None
    if parts is not None or finish_reason is not None:
        candidate = t.Candidate(
            content=t.Content(parts=parts or [], role="model"),
            finish_reason=finish_reason,
        )
    return t.GenerateContentResponse(candidates=[candidate] if candidate else None, usage_metadata=usage)


def _text_part(text):
    return t.Part(text=text)


def _thought_part(text):
    # In google-genai, a thinking part carries its reasoning text in `text`
    # and sets the boolean `thought` flag to True.
    return t.Part(text=text, thought=True)


def _fc_part(name, args=None, id=None):
    return t.Part(function_call=t.FunctionCall(name=name, args=args or {}, id=id))


def _usage_metadata(prompt=0, candidates=0, total=0, cached=0):
    return t.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=total,
        cached_content_token_count=cached,
    )


# ---------------------------------------------------------------------------
# Text streaming
# ---------------------------------------------------------------------------
def test_text_streaming_emits_text_delta_events():
    parser = GeminiStreamParser()
    events = parser.stream_chunk(_chunk(parts=[_text_part("Hello")]))
    events += parser.stream_chunk(_chunk(parts=[_text_part(" world")]))

    text = [e for e in events if e.type is StreamEventType.TEXT_DELTA]
    assert [e.text_delta.content for e in text] == ["Hello", " world"]


# ---------------------------------------------------------------------------
# Function calls
# ---------------------------------------------------------------------------
def test_function_call_emits_start_and_complete():
    parser = GeminiStreamParser()
    events = parser.stream_chunk(
        _chunk(parts=[_fc_part("get_weather", {"city": "Paris"}, id="fc_1")])
    )

    types = [e.type for e in events]
    assert types == [
        StreamEventType.TOOL_CALL_START,
        StreamEventType.TOOL_CALL_COMPLETE,
    ]
    start = events[0].tool_call_delta
    complete = events[1].tool_call

    assert start.call_id == "fc_1"
    assert start.name == "get_weather"
    assert complete.call_id == "fc_1"
    assert complete.name == "get_weather"
    assert complete.arguments == {"city": "Paris"}


def test_function_call_accumulates_args_as_raw_string_until_complete():
    """Gemini delivers a whole functionCall in one part, so it should
    produce TOOL_CALL_COMPLETE with structured args in one shot."""
    parser = GeminiStreamParser()
    events = parser.stream_chunk(
        _chunk(parts=[_fc_part("search", {"q": "ai training", "limit": 5}, id="fc_9")])
    )

    completes = [e for e in events if e.type is StreamEventType.TOOL_CALL_COMPLETE]
    assert len(completes) == 1
    assert completes[0].tool_call.name == "search"
    assert completes[0].tool_call.arguments == {"q": "ai training", "limit": 5}
    # raw_arguments is the JSON string form.
    assert '"q"' in completes[0].tool_call.raw_arguments


def test_duplicate_function_call_id_is_not_replayed():
    parser = GeminiStreamParser()
    first = parser.stream_chunk(_chunk(parts=[_fc_part("f", {"a": 1}, id="dup")]))
    second = parser.stream_chunk(_chunk(parts=[_fc_part("f", {"a": 1}, id="dup")]))

    assert len([e for e in first if e.type is StreamEventType.TOOL_CALL_START]) == 1
    assert second == []


# ---------------------------------------------------------------------------
# Usage + finish reason
# ---------------------------------------------------------------------------
def test_usage_metadata_is_mapped():
    parser = GeminiStreamParser()
    usage = _usage_metadata(prompt=10, candidates=20, total=30, cached=4)
    events = parser.stream_chunk(_chunk(usage=usage))

    usage_events = [e for e in events if e.usage is not None]
    assert len(usage_events) == 1
    assert usage_events[0].usage == TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cached_tokens=4)


def test_finish_reason_normalization():
    parser = GeminiStreamParser()
    cases = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
    }
    for raw, expected in cases.items():
        events = parser.stream_chunk(_chunk(finish_reason=raw))
        completes = [e for e in events if e.type is StreamEventType.MESSAGE_COMPLETE]
        assert completes and completes[0].finish_reason == expected, raw


def test_no_candidates_is_ok():
    parser = GeminiStreamParser()
    events = parser.stream_chunk(t.GenerateContentResponse())
    assert events == []


# ---------------------------------------------------------------------------
# Thinking
# ---------------------------------------------------------------------------
def test_thinking_part_emits_thinking_delta():
    parser = GeminiStreamParser()
    chunk = _chunk(parts=[_thought_part("Let me reason step by step..."), _text_part("Answer!")])
    events = parser.stream_chunk(chunk)

    types = [e.type for e in events]
    assert types == [StreamEventType.THINKING_DELTA, StreamEventType.TEXT_DELTA]

    thinking = events[0].thinking_delta
    text = events[1].text_delta
    assert thinking.content == "Let me reason step by step..."
    assert text.content == "Answer!"


def test_models_without_thinking_still_work():
    """A model that never emits thought parts should stream normally."""
    parser = GeminiStreamParser()
    events = parser.stream_chunk(_chunk(parts=[_text_part("plain answer")]))
    assert [e.type for e in events] == [StreamEventType.TEXT_DELTA]


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------
def test_non_streaming_response_uses_same_shape():
    parser = GeminiStreamParser()
    response = t.GenerateContentResponse(
        candidates=[
            t.Candidate(
                content=t.Content(
                    parts=[_text_part("Final"), _fc_part("finish", {"ok": True}, id="fc_x")],
                    role="model",
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=_usage_metadata(prompt=1, candidates=2, total=3),
    )

    events = parser.parse_response(response)

    types = [e.type for e in events]
    assert StreamEventType.TEXT_DELTA in types
    assert StreamEventType.TOOL_CALL_START in types
    assert StreamEventType.TOOL_CALL_COMPLETE in types
    assert types[-1] == StreamEventType.MESSAGE_COMPLETE

    final = events[-1]
    assert final.finish_reason == "stop"
    assert final.usage == TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
