"""Standalone entry point for the no-tool streaming agent.

One agent, either provider (OpenAI or Gemini) — chosen with -p / LLM_PROVIDER.

Usage:
    .venv\\Scripts\\python src\\simple_agent_main.py
    .venv\\Scripts\\python src\\simple_agent_main.py "tell me about the moon"
    .venv\\Scripts\\python src\\simple_agent_main.py -p gemini "hi"

Environment:
    LLM_PROVIDER                  default provider: openai | gemini (default: openai)
    OPENAI_KEY / OPENAI_API_KEY   for the OpenAI client
    OPENAI_BASE_URL / OPENAI_MODEL
    GEMINI_API_KEY                for the Gemini client
    GEMINI_MODEL
"""
import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from agent.agent import Agent
from agent.event import AgentEventType
from client import ClientFactory

load_dotenv()

SYSTEM_PROMPT = "You are a concise, helpful assistant."


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


async def main(prompt: str | None, provider: str) -> None:
    client = ClientFactory.create_client(provider)
    agent = Agent(client, system_prompt=SYSTEM_PROMPT)

    print(f"Provider: {provider}")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-tool streaming agent.")
    parser.add_argument(
        "-p",
        "--provider",
        choices=["openai", "gemini"],
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="provider to use (default: $LLM_PROVIDER or openai)",
    )
    parser.add_argument("prompt", nargs="?", help="optional single prompt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.prompt, args.provider))
