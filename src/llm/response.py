"""Common, provider-neutral framework events.

Every LLM provider/parser converts its own native wire format into these
models so the agent runtime never has to know about OpenAI, Gemini, etc.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    MESSAGE_COMPLETE = "message_complete"
    ERROR = "error"


@dataclass
class TextDelta:
    content: str

    def __str__(self) -> str:
        return self.content


@dataclass
class ThinkingDelta:
    content: str

    def __str__(self) -> str:
        return self.content


@dataclass
class ToolCallDelta:
    """Incremental fragment of an in-flight tool call (streaming)."""

    call_id: str
    name: str | None = None
    arguments_delta: str = ""


@dataclass
class ToolCall:
    """A completed tool call with its final arguments."""

    call_id: str
    name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: TokenUsage) -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass
class StreamEvent:
    type: StreamEventType
    text_delta: TextDelta | None = None
    thinking_delta: ThinkingDelta | None = None
    tool_call_delta: ToolCallDelta | None = None
    tool_call: ToolCall | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    error: str | None = None


def parse_tool_call_arguments(raw_arguments: str) -> dict[str, Any]:
    """Parse a streamed tool-call argument string into structured data.

    Malformed JSON is never silently dropped: the raw string is preserved
    under a ``raw_arguments`` key so the caller can surface or log it.
    """
    if not raw_arguments:
        return {}

    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"raw_arguments": raw_arguments}

    if isinstance(parsed, dict):
        return parsed

    # Function tools expect a JSON object; wrap anything else explicitly.
    return {"value": parsed, "raw_arguments": raw_arguments}
