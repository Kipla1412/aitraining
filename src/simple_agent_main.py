"""Standalone entry point for the no-tool streaming agent.

Usage:
    python simple_agent_main.py                       # interactive chat
    python simple_agent_main.py "tell me about moon"  # single prompt

Environment:
    OPENAI_KEY / OPENAI_API_KEY   required (OPENAI_KEY preferred)
    OPENAI_BASE_URL               optional (e.g. OpenRouter/Groq endpoint)
    OPENAI_MODEL                  optional (default: gpt-4o-mini)
Example:
    .venv\\Scripts\\python src\\simple_agent_main.py "hi"
"""
import asyncio
import os
import sys

from dotenv import load_dotenv

from agent.agent import Agent
from agent.event import AgentEventType
from client.llmclient import LLMClient

load_dotenv()

SYSTEM_PROMPT = "You are a concise, helpful assistant."


def build_client() -> LLMClient:
    return LLMClient(
        api_key=os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL")
        or os.getenv("OPENAI_DEFAULT_MODEL"),
    )


async def stream_reply(agent: Agent, message: str) -> None:
    print(f"\n[user] {message}")
    print("[assistant] ", end="", flush=True)

    async for event in agent.run(message):
        if event.type is AgentEventType.TEXT_DELTA:
            print(event.data["content"], end="", flush=True)
        elif event.type is AgentEventType.AGENT_ERROR:
            print(f"\n[error] {event.data['error']}")
        elif event.type is AgentEventType.AGENT_END:
            usage = event.data.get("usage")
            if usage:
                print(
                    f"\n[tokens] prompt={usage['prompt_tokens']} "
                    f"completion={usage['completion_tokens']} "
                    f"total={usage['total_tokens']}"
                )

    print()


async def main(prompt: str | None) -> None:
    client = build_client()
    agent = Agent(client, system_prompt=SYSTEM_PROMPT)

    async with client:
        if prompt:
            await stream_reply(agent, prompt)
            return

        print("Interactive chat. Type /exit to quit.")
        while True:
            try:
                user_input = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input:
                continue
            if user_input.lower() in ("/exit", "/quit"):
                break
            await stream_reply(agent, user_input)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
