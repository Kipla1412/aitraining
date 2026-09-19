"""Unit tests for the client factory (no network)."""
import pytest

from client import ClientFactory, GeminiLLMClient, LLMClient


def test_create_openai_client():
    client = ClientFactory.create_client("openai")
    assert isinstance(client, LLMClient)


def test_create_gemini_client():
    client = ClientFactory.create_client("gemini")
    assert isinstance(client, GeminiLLMClient)


def test_default_provider_is_openai():
    assert isinstance(ClientFactory.create_client(), LLMClient)


def test_kwargs_are_forwarded():
    client = ClientFactory.create_client("openai", model="gpt-4o", max_retries=1)
    assert isinstance(client, LLMClient)
    assert client._model == "gpt-4o"
    assert client._max_retries == 1


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        ClientFactory.create_client("nope")


def test_create_default_uses_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert isinstance(ClientFactory.create_default(), GeminiLLMClient)


def test_create_default_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert isinstance(ClientFactory.create_default(), LLMClient)
