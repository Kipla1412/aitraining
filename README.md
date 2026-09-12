# Simple OpenAI Chat Client & Agent

A small, easy-to-read chat stack built on the OpenAI API:

- **`LLMClient`** — a thin async client for chat completions (streaming and non-streaming, with retries).
- **`Agent`** — a streaming chat agent that keeps conversation history and emits simple events.

It is intentionally minimal: **no tool calling, no multi-provider framework**. Plain `dict` messages
(`{"role": ..., "content": ...}`) in, streamed text out.

---

## Features

- Streaming and non-streaming responses
- Automatic retries with exponential backoff (rate limits / connection errors)
- Conversation history across turns
- Small, explicit event model (no hidden magic)
- Async-first (`asyncio`)

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- An OpenAI API key

> New here? See the step-by-step **[Installation guide](INSTALL.md)**.

## Installation

```bash
# 1. create the environment and install dependencies
uv sync

# 2. create your .env file
cp .env.example .env   # then edit it and add your key
```

Minimal `.env`:

```dotenv
OPENAI_KEY=sk-your-key-here
# optional:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini
```

Full details (Windows/macOS/Linux, troubleshooting): **[INSTALL.md](INSTALL.md)**.

## Configuration

The client reads these environment variables (constructor arguments override them):

| Variable                      | Required | Description                                            |
| ----------------------------- | -------- | ------------------------------------------------------ |
| `OPENAI_KEY` / `OPENAI_API_KEY` | yes    | API key (`OPENAI_KEY` is preferred)                    |
| `OPENAI_BASE_URL`             | no       | Custom base URL (for OpenAI-compatible endpoints)      |
| `OPENAI_MODEL`                | no       | Model name (default: `gpt-4o-mini`)                    |

## Quick start

Run the demo agent (single prompt):

```bash
uv run python src/simple_agent_main.py "Tell me about the moon in one sentence."
```

Interactive chat:

```bash
uv run python src/simple_agent_main.py
# type /exit to quit
```

## Usage

### Use the client directly

```python
import asyncio
from client.llmclient import LLMClient
from llm.response import StreamEventType


async def main():
    async with LLMClient() as client:
        messages = [{"role": "user", "content": "Say hello in three words."}]

        async for event in client.chat_completion(messages, stream=True):
            if event.type is StreamEventType.TEXT_DELTA:
                print(event.text_delta.content, end="", flush=True)
            elif event.type is StreamEventType.MESSAGE_COMPLETE:
                print("\nfinish:", event.finish_reason, "usage:", event.usage)
            elif event.type is StreamEventType.ERROR:
                print("\nerror:", event.error)


asyncio.run(main())
```

Set `stream=False` to get the whole reply in one event instead.

### Use the agent

```python
import asyncio
from agent.agent import Agent
from agent.event import AgentEventType
from client.llmclient import LLMClient


async def main():
    async with LLMClient() as client:
        agent = Agent(client, system_prompt="You are a concise assistant.")

        async for event in agent.run("What is the capital of France?"):
            if event.type is AgentEventType.TEXT_DELTA:
                print(event.data["content"], end="", flush=True)
            elif event.type is AgentEventType.AGENT_END:
                print("\n", event.data["response"])


asyncio.run(main())
```

`Agent` remembers the conversation, so calling `agent.run(...)` again continues the same chat.
Use `agent.reset()` to clear history (the system prompt is kept).

## Events

The client yields `StreamEvent` objects (`src/llm/response.py`):

| Type               | Meaning                                            |
| ------------------ | -------------------------------------------------- |
| `TEXT_DELTA`       | A chunk of streamed text (`text_delta.content`)    |
| `MESSAGE_COMPLETE` | End of message (`finish_reason`, `usage`)          |
| `ERROR`            | Something failed (`error`)                         |

The agent yields `AgentEvent` objects (`src/agent/event.py`):

| Type            | Data                             |
| --------------- | -------------------------------- |
| `AGENT_START`   | `message`                        |
| `TEXT_DELTA`    | `content`                        |
| `TEXT_COMPLETE` | `content` (full reply)           |
| `AGENT_END`     | `response`, `usage`, `finish_reason` |
| `AGENT_ERROR`   | `error`, `details`               |

## Project layout

```text
src/
├── client/
│   └── llmclient.py       # LLMClient — OpenAI chat client
├── agent/
│   ├── agent.py           # Agent — streaming chat loop
│   └── event.py           # AgentEvent / AgentEventType
├── llm/
│   └── response.py        # shared StreamEvent models
├── simple_agent_main.py   # runnable demo (single prompt / interactive)
└── tests/                 # unit tests + live (integration) tests
```

## Running tests

Unit tests (no network):

```bash
uv run pytest -m "not integration"
```

Live tests (call the real API and consume tokens; skipped without `OPENAI_KEY`):

```bash
uv run pytest -m integration
```

## Notes

- `openai==3.6.0` depends on `httpx2`. You may occasionally see a
  `httpcore2 ... generator didn't stop after athrow()` message when a script exits.
  It is a harmless shutdown race in `httpx2` and does not affect the output.
