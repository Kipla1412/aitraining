"""Agent-level events for the no-tool streaming agent.

These mirror the shape of the fuller agent's events but intentionally omit
any tool-calling events, since this agent has no tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llm.response import TokenUsage


class AgentEventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"


@dataclass
class AgentEvent:
    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def agent_start(cls, message: str) -> "AgentEvent":
        return cls(type=AgentEventType.AGENT_START, data={"message": message})

    @classmethod
    def agent_end(
        cls,
        response: str,
        usage: TokenUsage | None = None,
        finish_reason: str | None = None,
    ) -> "AgentEvent":
        return cls(
            type=AgentEventType.AGENT_END,
            data={
                "response": response,
                "usage": usage.__dict__ if usage else None,
                "finish_reason": finish_reason,
            },
        )

    @classmethod
    def agent_error(
        cls, error: str, details: dict[str, Any] | None = None
    ) -> "AgentEvent":
        return cls(
            type=AgentEventType.AGENT_ERROR,
            data={"error": error, "details": details or {}},
        )

    @classmethod
    def text_delta(cls, content: str) -> "AgentEvent":
        return cls(type=AgentEventType.TEXT_DELTA, data={"content": content})

    @classmethod
    def text_complete(cls, content: str) -> "AgentEvent":
        return cls(type=AgentEventType.TEXT_COMPLETE, data={"content": content})
