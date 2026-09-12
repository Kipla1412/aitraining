"""Shared abstractions for LLM providers.

``ChatMessage`` / ``ToolResultMessage`` expose small provider-facing adapter
hooks (``to_openai_message`` / ``to_gemini_content``) so the provider modules
stay free of chat-history conversion logic. The models themselves carry no
provider-specific runtime behavior beyond those hooks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from llm.response import ToolCall

# HTTP statuses that warrant a retry (transient server/rate-limit errors).
_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})

# Normalize OpenAI/Gemini role names to a common set.
_ROLE_ALIASES: dict[str, Literal["user", "model", "system", "tool"]] = {
    "assistant": "model",
    "developer": "system",
}


@dataclass
class ChatMessage:
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @property
    def normalized_role(self) -> str:
        return _ROLE_ALIASES.get(self.role, self.role)

    def to_openai_message(self) -> dict[str, Any]:
        role = "assistant" if self.normalized_role == "model" else self.normalized_role

        if role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content or "",
            }

        if role == "system":
            return {"role": "system", "content": self.content or ""}

        message: dict[str, Any] = {"role": role, "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name or "",
                        "arguments": tc.raw_arguments or "",
                    },
                }
                for tc in self.tool_calls
            ]
        return message

    def to_gemini_content(self) -> dict[str, Any]:
        role = "model" if self.normalized_role == "assistant" else self.normalized_role

        if role == "user" and self.tool_calls:
            # Tool results are carried as function_response parts in Gemini.
            raise ValueError(
                "tool results must be sent as separate ToolResultMessage items"
            )

        parts: list[dict[str, Any]] = []
        if self.content:
            parts.append({"text": self.content})
        if self.tool_calls:
            for tc in self.tool_calls:
                try:
                    args = json.loads(tc.raw_arguments or "{}")
                except json.JSONDecodeError:
                    args = {"raw_arguments": tc.raw_arguments}
                parts.append(
                    {
                        "function_call": {
                            "name": tc.name or "",
                            "args": args,
                            "id": tc.call_id,
                        }
                    }
                )
        return {"role": role, "parts": parts}


@dataclass
class ToolResultMessage:
    tool_call_id: str
    content: str
    is_error: bool = False
    name: str | None = None

    def to_openai_message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }

    def to_gemini_content(self) -> dict[str, Any]:
        return {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": self.name or "",
                        "response": {"result": self.content},
                        "id": self.tool_call_id,
                    }
                }
            ],
        }


def retry_on_transient(exc: Exception) -> bool:
    """Decide whether an API error is transient and worth a retry.

    Covers HTTP-level status errors (OpenAI raises APIStatusError with a
    ``status_code``; google-genai raises APIError with a ``code``) and
    connection-level failures (no code). Auth/config errors (4xx other than
    408/409/429) fail fast.
    """
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)

    if isinstance(code, int):
        return code in _RETRYABLE_STATUSES

    # Connection-level failures (dns, dropped connection, timeouts) that
    # expose no HTTP status are transient by nature.
    return True


@dataclass
class ToolSpec:
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
