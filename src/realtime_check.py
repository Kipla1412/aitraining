"""Realtime multi-turn chat with the OpenAI and Gemini clients.

A simple loop: you type a message, the reply streams in realtime, and the
conversation history is kept between turns. The conversation ends when you
type ``exit`` (or ``quit`` / ``/exit``), or press Ctrl+C / Ctrl+D.

Usage:
    .venv\\Scripts\\python src\\realtime_check.py              # default provider
    .venv\\Scripts\\python src\\realtime_check.py -p gemini    # choose provider

Environment:
    OPENAI_KEY / OPENAI_API_KEY   for the OpenAI client
    GEMINI_API_KEY                for the Gemini client
    OPENAI_MODEL / GEMINI_MODEL   optional model overrides
    LLM_PROVIDER                  default provider ("openai" if unset)
"""
import argparse
import asyncio
import os
import sys
import time

from dotenv import load_dotenv

from agent.agent import Agent
from agent.event import AgentEventType
from client import ClientFactory

load_dotenv()

SYSTEM_PROMPT = "You are a concise, helpful assistant."
EXIT_WORDS = {"exit", "quit", "/exit", "/quit"}

_HAS_KEY = {
    "openai": lambda: bool(os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")),
    "gemini": lambda: bool(os.getenv("GEMINI_API_KEY")),
}


async def stream_turn(agent: Agent, message: str) -> None:
    """Send one message and print the reply as it streams in."""
    print("[assistant] ", end="", flush=True)

    start = time.perf_counter()
    first_token = None

    async for event in agent.run(message):
        if event.type is AgentEventType.TEXT_DELTA:
            if first_token is None:
                first_token = time.perf_counter() - start
            print(event.data["content"], end="", flush=True)

        elif event.type is AgentEventType.AGENT_ERROR:
            print(f"\n[error] {event.data['error']}")
            return

        elif event.type is AgentEventType.AGENT_END:
            total = time.perf_counter() - start
            first = f"{first_token:.2f}s" if first_token is not None else "n/a"
            print(
                f"\n[done] finish={event.data.get('finish_reason')} "
                f"first_token={first} total={total:.2f}s"
            )


async def chat(provider: str) -> int:
    """Run the interactive multi-turn conversation loop."""
    client = ClientFactory.create_client(provider)
    agent = Agent(client, system_prompt=SYSTEM_PROMPT)

    print(f"Realtime chat with '{provider}'. Type 'exit' to end the conversation.\n")

    async with client:
        while True:
            try:
                message = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not message:
                continue

            if message.lower() in EXIT_WORDS:
                break

            await stream_turn(agent, message)
            print()

    print("Conversation ended.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime multi-turn LLM chat.")
    parser.add_argument(
        "-p",
        "--provider",
        choices=sorted(_HAS_KEY),
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="provider to chat with (default: $LLM_PROVIDER or openai)",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    provider = args.provider

    if not _HAS_KEY[provider]():
        print(f"No API key set for provider '{provider}'.")
        return 1

    return await chat(provider)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
