"""Parse OpenAI wire objects into common :mod:`llm.response` StreamEvents."""
from __future__ import annotations

from typing import Any, Iterable

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llm.response import (
    StreamEvent,
    StreamEventType,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    parse_tool_call_arguments,
)

# OpenAI -> common finish reasons.
_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "function_call": "tool_calls",
}


class OpenAIStreamParser:
    """Normalize OpenAI chat responses/chunks into ``StreamEvent`` objects."""

    def __init__(self) -> None:
        self._pending_tool_calls: dict[int, ToolCallDelta] = {}

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def stream_chunk(self, chunk: ChatCompletionChunk) -> list[StreamEvent]:
        """Convert one streaming chunk into a list of events."""
        events: list[StreamEvent] = []
        choice = chunk.choices[0] if chunk.choices else None

        if choice is not None and choice.delta is not None:
            delta = choice.delta
            events.extend(self._handle_content(delta.content))
            events.extend(self._handle_tool_calls(delta.tool_calls))

            if choice.finish_reason is not None:
                self._flush_pending_tool_calls(events)
                events.append(
                    StreamEvent(
                        type=StreamEventType.MESSAGE_COMPLETE,
                        finish_reason=_FINISH_REASON_MAP.get(
                            choice.finish_reason, choice.finish_reason
                        ),
                    )
                )

        if chunk.usage is not None:
            events.append(
                StreamEvent(
                    type=StreamEventType.MESSAGE_COMPLETE,
                    usage=self._to_usage(chunk.usage),
                )
            )

        return events

    def _handle_content(self, content: str | None) -> list[StreamEvent]:
        if not content:
            return []
        return [
            StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text_delta=TextDelta(content=content),
            )
        ]

    def _handle_tool_calls(
        self, tool_calls: Iterable[Any] | None
    ) -> list[StreamEvent]:
        """Detect new tool calls, track them by index, emit deltas.

        OpenAI streams a tool call's id/name on the first fragment and its
        arguments across many following fragments, so we accumulate by index
        instead of assuming the arguments arrive in a single chunk.
        """
        events: list[StreamEvent] = []
        if not tool_calls:
            return events

        for tc in tool_calls:
            idx = tc.index
            name = tc.function.name if tc.function else None

            if tc.id:
                # New tool call -> (re)start tracking it.
                self._pending_tool_calls[idx] = ToolCallDelta(
                    call_id=tc.id, name=name
                )
                events.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call_delta=ToolCallDelta(call_id=tc.id, name=name),
                    )
                )
            elif idx not in self._pending_tool_calls:
                # Continuation arriving before we ever saw the id/name.
                call_id = f"call_{idx}"
                self._pending_tool_calls[idx] = ToolCallDelta(
                    call_id=call_id, name=name
                )
                events.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call_delta=ToolCallDelta(
                            call_id=call_id, name=name
                        ),
                    )
                )

            fragment = tc.function.arguments if tc.function else None
            if fragment:
                pending = self._pending_tool_calls[idx]
                pending.arguments_delta += fragment
                events.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_DELTA,
                        tool_call_delta=ToolCallDelta(
                            call_id=pending.call_id,
                            name=pending.name,
                            arguments_delta=fragment,
                        ),
                    )
                )

        return events

    def _flush_pending_tool_calls(self, events: list[StreamEvent]) -> None:
        """Emit TOOL_CALL_COMPLETE for calls the finish chunk closes out."""
        for pending in self._pending_tool_calls.values():
            events.append(
                StreamEvent(
                    type=StreamEventType.TOOL_CALL_COMPLETE,
                    tool_call=ToolCall(
                        call_id=pending.call_id,
                        name=pending.name,
                        arguments=parse_tool_call_arguments(
                            pending.arguments_delta
                        ),
                        raw_arguments=pending.arguments_delta,
                    ),
                )
            )
        self._pending_tool_calls.clear()

    @staticmethod
    def _to_usage(usage: Any) -> TokenUsage:
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            cached_tokens=(
                getattr(prompt_tokens_details, "cached_tokens", 0) or 0
            ),
        )

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    def parse_response(self, response: ChatCompletion) -> list[StreamEvent]:
        """Convert a complete (non-streamed) chat completion into events."""
        events: list[StreamEvent] = []
        choice = response.choices[0]
        message = choice.message

        if message.content:
            events.append(
                StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(content=message.content),
                )
            )

        if message.tool_calls:
            for tc in message.tool_calls:
                fn = tc.function
                raw = fn.arguments if fn else ""
                events.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call_delta=ToolCallDelta(
                            call_id=tc.id, name=fn.name if fn else None
                        ),
                    )
                )
                events.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_COMPLETE,
                        tool_call=ToolCall(
                            call_id=tc.id,
                            name=fn.name if fn else None,
                            arguments=parse_tool_call_arguments(raw),
                            raw_arguments=raw,
                        ),
                    )
                )

        events.append(
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                finish_reason=_FINISH_REASON_MAP.get(
                    choice.finish_reason, choice.finish_reason
                ),
                usage=self._to_usage(response.usage)
                if response.usage is not None
                else None,
            )
        )
        return events
