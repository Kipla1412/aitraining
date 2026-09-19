# Simple OpenAI & Gemini Chat Client & Agent

A small, easy-to-read chat stack for **OpenAI** and **Gemini**:

- **`LLMClient` / `GeminiLLMClient`** — thin async clients for chat completions (streaming and non-streaming, with retries).
- **`ClientFactory`** — pick a provider by name, or from `LLM_PROVIDER`.
- **`Agent`** — a streaming chat agent that keeps conversation history and emits simple events.

It is intentionally minimal: **no tool calling** — no tool registry, no function schemas,
no execution loop. Plain `dict` messages (`{"role": ..., "content": ...}`) in, streamed text out.

---

## Features

- Streaming and non-streaming responses
- Automatic retries with exponential backoff (rate limits / connection errors)
- Conversation history across turns
- Small, explicit event model (no hidden magic)
- Async-first (`asyncio`)

## How it works (the approach)

A thin client talks to the LLM, and a small agent runs the chat loop. Data flows
one way — no framework, no hidden layers:

```text
your code
   │  agent.run("What is the capital of France?")
   ▼
Agent              keeps conversation history, streams AgentEvents
   │  client.chat_completion(messages, stream=True)
   ▼
Client             OpenAI or Gemini (same interface), emits StreamEvents
   │  openai / google-genai SDK
   ▼
LLM API
```

`Agent` never touches the network and never asks which provider it is — it only calls
`chat_completion(messages, stream=True)`. The client does one job: talk to the API and
normalize the reply into events.

The design is deliberately small:

- **Plain `dict` messages.** History is a `list[dict]` (`{"role": ..., "content": ...}`),
  exactly what the OpenAI API expects. No wrapper/`ChatMessage` classes.
- **One client interface, two providers.** `LLMClient` (OpenAI-compatible) and
  `GeminiLLMClient` expose the same `chat_completion(messages, stream=True)` and both
  yield `StreamEvent`s. Pick one with `ClientFactory.create_client("openai" | "gemini")`
  or the `LLM_PROVIDER` env var — the agent never branches on provider.
- **No tool calling.** Text in, text out — no tool registry, no function schemas,
  no execution loop.
- **Events, not callbacks.** Both layers are async generators yielding small event
  objects, so you `async for` and branch on `event.type`.

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- An API key: OpenAI and/or Gemini

> New here? See the step-by-step **[Installation guide](INSTALL.md)**.

## Installation

```bash
# 1. create the environment and install dependencies
uv sync

# 2. create your .env file
cp .env.example .env   # then edit it and add your key
```

Minimal `.env` (OpenAI):

```dotenv
OPENAI_KEY=sk-your-key-here
# optional:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini
```

Or Gemini:

```dotenv
GEMINI_API_KEY=your-gemini-key
# optional:
# GEMINI_MODEL=gemini-3.6-flash
# LLM_PROVIDER=gemini
```

Full details (Windows/macOS/Linux, troubleshooting): **[INSTALL.md](INSTALL.md)**.

## Configuration

The clients read these environment variables (constructor arguments override them):

| Variable                        | Required    | Description                                              |
| ------------------------------- | ----------- | -------------------------------------------------------- |
| `LLM_PROVIDER`                  | no          | Default provider: `openai` or `gemini` (default: `openai`) |
| `OPENAI_KEY` / `OPENAI_API_KEY` | for OpenAI  | API key (`OPENAI_KEY` is preferred)                      |
| `OPENAI_BASE_URL`               | no          | Custom base URL (for OpenAI-compatible endpoints)        |
| `OPENAI_MODEL`                  | no          | OpenAI model name (default: `gpt-4o-mini`)               |
| `GEMINI_API_KEY`                | for Gemini  | Gemini API key                                           |
| `GEMINI_MODEL`                  | no          | Gemini model name (default: `gemini-3.6-flash`)          |

## Quick start

Run the demo agent (OpenAI by default):

```bash
uv run python src/simple_agent_main.py "Tell me about the moon in one sentence."
```

Pick the provider with `-p` (or `LLM_PROVIDER`):

```bash
uv run python src/simple_agent_main.py -p gemini "Tell me about the moon in one sentence."
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

### Pick a provider

```python
from client import ClientFactory

client = ClientFactory.create_client("gemini")   # or "openai"
# or read LLM_PROVIDER from the environment:
client = ClientFactory.create_default()
```

Both clients share the same `chat_completion(messages, stream=True)` interface, so the
agent code below does not change.

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
│   ├── llmclient.py       # LLMClient — OpenAI chat client
│   ├── gemini.py          # GeminiLLMClient — Gemini chat client
│   └── factory.py         # ClientFactory — pick a provider by name
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

Live tests (call the real APIs and consume tokens; each provider is skipped when its
key is not set — `OPENAI_KEY` / `GEMINI_API_KEY`):

```bash
uv run pytest -m integration
```

## Notes

- `openai==3.6.0` depends on `httpx2`. You may occasionally see a
  `httpcore2 ... generator didn't stop after athrow()` message when a script exits.
  It is a harmless shutdown race in `httpx2` and does not affect the output.
