"""OpenAI- and Gemini-compatible chat clients (text only, no tool calling).

Both clients expose the same interface::

    chat_completion(messages, stream=True) -> AsyncGenerator[StreamEvent, None]

so the agent can use either one without knowing which provider it is.
"""
from client.factory import ClientFactory
from client.gemini import GeminiLLMClient
from client.llmclient import LLMClient

__all__ = ["LLMClient", "GeminiLLMClient", "ClientFactory"]
