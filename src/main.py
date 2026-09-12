"""Demo: provider-neutral usage of the LLM framework.

Swap the provider name + config and the rest of the loop is unchanged.
"""
import asyncio
import os

from dotenv import load_dotenv

from llm.agent import AgentRuntime
from llm.base import ToolSpec
from llm.factory import LLMClientFactory

load_dotenv()


async def openai_demo() -> None:
    provider = LLMClientFactory.create_provider(
        "openai",
        LLMClientFactory.create_config(
            "openai",
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
        ),
    )

    async with provider:
        runtime = AgentRuntime(provider, system_prompt="You are terse.")
        result = await runtime.run("Tell me about the moon in one sentence.")
        print("content:", result.content)
        print("finish_reason:", result.finish_reason)


async def gemini_demo() -> None:
    provider = LLMClientFactory.create_provider(
        "gemini",
        LLMClientFactory.create_config(
            "gemini",
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash"),
        ),
    )

    async with provider:
        runtime = AgentRuntime(provider)
        result = await runtime.run("Explain what a function call is.")
        print("content:", result.content)
        if result.thinking:
            print("thinking:", result.thinking)


async def tool_loop_demo() -> None:
    """Same AgentRuntime drives a provider-neutral tool loop."""
    provider = LLMClientFactory.create_provider(
        "openai",
        LLMClientFactory.create_config(
            "openai",
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
        ),
    )

    async def weather_tool(call) -> str:
        city = call.arguments.get("city", "?")
        return f"sunny, 24C in {city}"

    async with provider:
        runtime = AgentRuntime(provider, system_prompt="Use tools when asked.")
        result = await runtime.run(
            "What is the weather in Paris?",
            tools=[
                ToolSpec(
                    name="get_weather",
                    description="Get the current weather for a city.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                    },
                )
            ],
            executor=weather_tool,
        )
        print("content:", result.content)
        print("tool_calls:", result.tool_calls)


if __name__ == "__main__":
    # asyncio.run(openai_demo())
    # asyncio.run(gemini_demo())
    asyncio.run(tool_loop_demo())
