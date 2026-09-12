"""A simple streaming chat agent (no tool calling).

It talks to one thing only: the OpenAI-compatible client in ``client.llmclient``.
Messages are plain dicts (``{"role": ..., "content": ...}``) -- no wrapper
classes, no provider switch, no tools. Easy to follow end to end.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List

from client.llmclient import LLMClient
from llm.response import StreamEventType, TokenUsage

from agent.event import AgentEvent


class Agent:
    """Streams a chat reply back as ``AgentEvent``s, keeping history."""

    def __init__(
        self,
        client: LLMClient,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._messages: List[Dict[str, Any]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    # state
 
    @property
    def messages(self) -> List[Dict[str, Any]]:
        return list(self._messages)

    def reset(self) -> None:
        self._messages = []
        if self._system_prompt:
            self._messages.append(
                {"role": "system", "content": self._system_prompt}
            )

    
    # run

    async def run(self, message: str) -> AsyncGenerator[AgentEvent, None]:
        """Send ``message`` and stream the reply as ``AgentEvent``s.

        Event order: AGENT_START -> TEXT_DELTA* -> TEXT_COMPLETE -> AGENT_END,
        or AGENT_START -> (TEXT_DELTA*) -> AGENT_ERROR on failure.
        """
        self._messages.append({"role": "user", "content": message})
        yield AgentEvent.agent_start(message)

        response_txt = ""
        usage: TokenUsage | None = None
        finish_reason: str | None = None

        async for event in self._client.chat_completion(
            self._messages, stream=True
        ):
            if event.type is StreamEventType.TEXT_DELTA and event.text_delta:
                response_txt += event.text_delta.content
                yield AgentEvent.text_delta(event.text_delta.content)
            elif event.type is StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or "Unknown error")
                return
            elif event.type is StreamEventType.MESSAGE_COMPLETE:
                if event.usage:
                    usage = event.usage
                if event.finish_reason:
                    finish_reason = event.finish_reason

        if response_txt:
            self._messages.append(
                {"role": "assistant", "content": response_txt}
            )
            yield AgentEvent.text_complete(response_txt)

        yield AgentEvent.agent_end(response_txt, usage, finish_reason)
