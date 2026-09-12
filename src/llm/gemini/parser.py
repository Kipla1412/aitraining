"""Parse Gemini wire objects into common :mod:`llm.response` StreamEvents."""
from __future__ import annotations

import json
from typing import Any, Sequence

from google.genai.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    Part,
)

from llm.response import (
    StreamEvent,
    StreamEventType,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    parse_tool_call_arguments,
)

# Gemini finish reasons -> common framework values.
_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "MALFORMED_FUNCTION_CALL": "tool_calls",
    "OTHER": "other",
}


def _finish_reason_name(reason: Any) -> str | None:
    """Get a stable string name from a gemini FinishReason enum member."""
    if reason is None:
        return None
    if isinstance(reason, str):
        return reason
    # SDK enum members expose .value (the raw int) and .name (the string).
    return getattr(reason, "name", None) or str(reason)


class GeminiStreamParser:
    """Normalize Gemini ``GenerateContentResponse`` objects into StreamEvents.

    Works for both streaming chunks and a final non-streamed response since
    the SDK uses the same model for both.
    """

    def __init__(self) -> None:
        self._finished_calls: set[str] = set()

    # ------------------------------------------------------------------
    # Response / chunk entry point
    # ------------------------------------------------------------------
    def stream_chunk(
        self, chunk: GenerateContentResponse
    ) -> list[StreamEvent]:
        """Convert one Gemini response/chunk into a list of events."""
        events: list[StreamEvent] = []

        finish_reason: str | None = None
        if chunk.candidates:
            candidate = chunk.candidates[0]
            self._parse_content_parts(candidate.content, events)
            raw_finish = _finish_reason_name(candidate.finish_reason)
            if raw_finish:
                finish_reason = _FINISH_REASON_MAP.get(raw_finish, raw_finish)

        usage = self._to_usage(chunk.usage_metadata) if chunk.usage_metadata is not None else None

        if finish_reason or usage is not None:
            events.append(
                StreamEvent(
                    type=StreamEventType.MESSAGE_COMPLETE,
                    finish_reason=finish_reason,
                    usage=usage,
                )
            )

        return events

    # ------------------------------------------------------------------
    # Parts
    # ------------------------------------------------------------------
    def _parse_content_parts(
        self, content: Content | None, events: list[StreamEvent]
    ) -> None:
        if content is None:
            return

        for part in content.parts or []:
            events.extend(self._handle_part(part))

    def _handle_part(self, part: Part) -> list[StreamEvent]:
        # Thinking/reasoning parts carry their text in `text` and set the
        # boolean `thought` flag; not every model produces them.
        if part.thought and part.text:
            return self._handle_thinking(part.text)
        if part.text:
            return self._handle_text(part.text)
        if part.function_call is not None:
            return self._handle_function_call(part.function_call)
        # Other part kinds (file_data, inline_data, ...) are not chat text.
        return []

    # ------------------------------------------------------------------
    # Text & thinking
    # ------------------------------------------------------------------
    @staticmethod
    def _handle_text(text: str) -> list[StreamEvent]:
        if not text:
            return []
        return [
            StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text_delta=TextDelta(content=text),
            )
        ]

    @staticmethod
    def _handle_thinking(thought: str) -> list[StreamEvent]:
        if not thought:
            return []
        return [
            StreamEvent(
                type=StreamEventType.THINKING_DELTA,
                thinking_delta=ThinkingDelta(content=thought),
            )
        ]

    # ------------------------------------------------------------------
    # Function calls
    # ------------------------------------------------------------------
    def _handle_function_call(self, fc: Any) -> list[StreamEvent]:
        name = getattr(fc, "name", None)
        call_id = getattr(fc, "id", None) or f"call_{name}"

        if call_id in self._finished_calls:
            # A re-sent/replayed function call for a completed invocation.
            return []

        self._finished_calls.add(call_id)
        events: list[StreamEvent] = [
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                tool_call_delta=ToolCallDelta(call_id=call_id, name=name),
            )
        ]

        raw_arguments = self._extract_arguments(fc)
        events.append(
            StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=parse_tool_call_arguments(raw_arguments),
                    raw_arguments=raw_arguments,
                ),
            )
        )
        return events

    @staticmethod
    def _extract_arguments(fc: Any) -> str:
        """Return the function call's args as a JSON string.

        Prefers the structured ``args`` dict when present, falling back to
        ``partial_args`` (streamed fragments) for providers that send it.
        """
        args = getattr(fc, "args", None)
        if args is not None:
            # args may arrive as a plain dict or a JSON string depending on
            # the SDK/transport. Normalize both to a stable string.
            if isinstance(args, str):
                return args
            return json.dumps(args, ensure_ascii=False)

        partial = getattr(fc, "partial_args", None)
        if partial:
            if isinstance(partial, str):
                return partial
            return json.dumps(partial, ensure_ascii=False)

        return ""

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------
    @staticmethod
    def _to_usage(metadata: Any) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
            completion_tokens=(
                getattr(metadata, "candidates_token_count", 0) or 0
            ),
            total_tokens=getattr(metadata, "total_token_count", 0) or 0,
            cached_tokens=(
                getattr(metadata, "cached_content_token_count", 0) or 0
            ),
        )

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    def parse_response(
        self, response: GenerateContentResponse
    ) -> list[StreamEvent]:
        """Convert a complete (non-streamed) response into events."""
        # The SDK reuses the same object shape for both paths.
        return self.stream_chunk(response)
