# Installation Guide

This guide sets up the **Simple OpenAI Chat Client & Agent** from a clean machine.

It uses [`uv`](https://docs.astral.sh/uv/), which creates the virtual environment and installs
dependencies in one step. A plain `pip`/`venv` alternative is included at the end.

---

## 1. Prerequisites

- **Python 3.10 or newer**

  ```bash
  python --version
  ```

  If it prints 3.9 or lower, install a newer Python first (https://www.python.org/downloads/).

- **uv** (recommended package manager)

  **Windows (PowerShell):**

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

  **macOS / Linux:**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  Verify:

  ```bash
  uv --version
  ```

- **An OpenAI API key** from https://platform.openai.com/api-keys

## 2. Get the project

```bash
cd path/to/aitraining
```

## 3. Create the environment and install dependencies

```bash
uv sync
```

This creates a `.venv/` folder and installs everything from `pyproject.toml`
(`openai`, `python-dotenv`, plus the dev tools used for testing).

## 4. Configure your API key

Copy the example file and edit it:

**Windows:**

```cmd
copy .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

Open `.env` and set your key:

```dotenv
OPENAI_KEY=sk-your-real-key-here
```

Optional settings:

```dotenv
# Use an OpenAI-compatible endpoint
# OPENAI_BASE_URL=https://api.openai.com/v1

# Pick a different model
# OPENAI_MODEL=gpt-4o-mini
```

> The key is read from the environment; `OPENAI_KEY` is preferred, `OPENAI_API_KEY` also works.
> Never commit `.env` — keep it out of version control.

## 5. Verify the installation

Run the demo (single prompt):

```bash
uv run python src/simple_agent_main.py "Say hello in three words."
```

You should see the streamed reply:

```text
[user] Say hello in three words.
[assistant] Hello, how are you?
```

Or start interactive chat:

```bash
uv run python src/simple_agent_main.py
```

Type a message and press Enter; type `/exit` to quit.

## 6. Run the tests (optional)

Unit tests (no network, fast):

```bash
uv run pytest -m "not integration"
```

Live tests (call the real API and use tokens; skipped if no key is set):

```bash
uv run pytest -m integration
```

---

## Alternative: plain pip / venv

If you prefer not to use `uv`:

```bash
# 1. create and activate a virtual environment
python -m venv .venv

# Windows (cmd)
.venv\Scripts\activate.bat
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. install dependencies
pip install -e .

# 3. configure your key
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
# then edit .env and add OPENAI_KEY

# 4. run
python src/simple_agent_main.py "Say hello in three words."
```

---

## Troubleshooting

**`No API key found` / authentication error**
Check that `.env` exists next to `pyproject.toml`, contains a valid `OPENAI_KEY`, and that the
file is loaded (the app calls `load_dotenv()` on startup).

**`ModuleNotFoundError: No module named 'client'` (or `agent`, `llm`)**
Run scripts from the project root and let the script's own folder be on the path:

```bash
uv run python src/simple_agent_main.py "hi"
```

For your own scripts, run them the same way (`uv run python ...`), or add `src` to `PYTHONPATH`.
For tests, `pyproject.toml` already sets `pythonpath = ["src"]`.

**`429 ... insufficient_quota`**
The OpenAI account behind the key has no credits. Add billing/credits, or point
`OPENAI_BASE_URL` at an OpenAI-compatible endpoint you have access to.

**`httpcore2 ... generator didn't stop after athrow()`**
A harmless shutdown message from `httpx2` (used by `openai`). It appears after the program has
finished and does not affect results.

**`uv: command not found`**
Close and reopen your terminal after installing `uv` (the installer updates your `PATH`).
