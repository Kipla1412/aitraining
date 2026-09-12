"""A tiny provider-neutral agent loop.

Consumes only :mod:`llm.response` StreamEvents; it never inspects provider
wire formats. Tool execution is delegated to the caller-provided ``executor``
callable, and any provider can be swapped in without touching this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from llm.base import ChatMessage, ToolResultMessage, ToolSpec
from llm.response import StreamEvent, StreamEventType, ToolCall

ToolExecutor = Callable[[ToolCall], Awaitable[str]]


@dataclass
class AgentResult:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Any = None
    events: list[StreamEvent] = field(default_factory=list)
    error: str | None = None


class AgentRuntime:
    """Runs a provider through a request/execute loop until it stops."""

    def __init__(
        self,
        provider: Any,
        system_prompt: str | None = None,
        max_turns: int = 10,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt
        self._max_turns = max_turns

    async def run(
        self,
        user_input: str,
        tools: Sequence[ToolSpec] | None = None,
        executor: ToolExecutor | None = None,
    ) -> AgentResult:
        history: list[ChatMessage] = []
        if self._system_prompt:
            history.append(ChatMessage(role="system", content=self._system_prompt))
        history.append(ChatMessage(role="user", content=user_input))

        result = AgentResult()

        for _ in range(self._max_turns):
            turn_tool_calls: list[ToolCall] = []
            events: list[StreamEvent] = []
            usage: Any = None
            finish_reason: str | None = None
            turn_content: list[str] = []

            async for event in self._provider.generate(
                messages=history, tools=tools, stream=True
            ):
                events.append(event)
                if event.type is StreamEventType.TEXT_DELTA and event.text_delta:
                    turn_content.append(event.text_delta.content)
                    result.content += event.text_delta.content
                elif event.type is StreamEventType.THINKING_DELTA and event.thinking_delta:
                    result.thinking += event.thinking_delta.content
                elif event.type is StreamEventType.TOOL_CALL_COMPLETE and event.tool_call:
                    turn_tool_calls.append(event.tool_call)
                elif event.type is StreamEventType.MESSAGE_COMPLETE:
                    if event.finish_reason:
                        finish_reason = event.finish_reason
                    if event.usage:
                        usage = event.usage
                elif event.type is StreamEventType.ERROR:
                    result.error = event.error
                    break

            if result.error:
                break

            result.events.extend(events)
            result.finish_reason = finish_reason
            result.usage = usage or result.usage
            result.tool_calls.extend(turn_tool_calls)

            # Preserve the assistant turn in history for any follow-up turns.
            history.append(
                ChatMessage(
                    role="assistant",
                    content="".join(turn_content),
                    tool_calls=turn_tool_calls,
                )
            )

            if not turn_tool_calls or executor is None:
                break

            for tool_call in turn_tool_calls:
                try:
                    output = await executor(tool_call)
                    is_error = False
                except Exception as exc:  # noqa: BLE001 - surface tool failure
                    output = f"tool error: {exc}"
                    is_error = True
                history.append(
                    ToolResultMessage(
                        tool_call_id=tool_call.call_id,
                        content=output,
                        is_error=is_error,
                        name=tool_call.name,
                    )
                )

        return result
