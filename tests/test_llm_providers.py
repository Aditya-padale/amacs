"""Tests for LLM provider abstraction."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from amacs.exceptions import LLMProviderError
from amacs.integrations.llm_providers import (
    GroqProvider,
    LLMResponse,
    Message,
    StubProvider,
    get_provider,
)


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="Groq response", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )


class _FakeClient:
    def __init__(self, **kwargs) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class _FakeAsyncCompletions(_FakeCompletions):
    async def create(self, **kwargs):
        return super().create(**kwargs)


class _FakeAsyncClient:
    def __init__(self, **kwargs) -> None:
        self.chat = SimpleNamespace(completions=_FakeAsyncCompletions())


class _FakeGroqModule:
    def __init__(self) -> None:
        self.clients = []

    def Groq(self, **kwargs):
        client = _FakeClient(**kwargs)
        self.clients.append(client)
        return client

    def AsyncGroq(self, **kwargs):
        return _FakeAsyncClient(**kwargs)


def test_groq_provider_uses_groq_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_groq = _FakeGroqModule()
    monkeypatch.setitem(__import__("sys").modules, "groq", fake_groq)
    provider = GroqProvider(api_key="test-key")

    response = provider.chat([Message(role="user", content="Hello")], timeout=12)

    assert response.content == "Groq response"
    assert response.usage["total_tokens"] == 5
    assert fake_groq.clients[0].chat.completions.calls[0]["model"] == provider.DEFAULT_MODEL
    assert fake_groq.clients[0].chat.completions.calls[0]["timeout"] == 12


@pytest.mark.asyncio
async def test_groq_provider_supports_async_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_groq = _FakeGroqModule()
    monkeypatch.setitem(__import__("sys").modules, "groq", fake_groq)
    provider = GroqProvider(api_key="test-key")

    response = await provider.achat([Message(role="user", content="Hello")])

    assert response.content == "Groq response"


class TestStubProvider:
    def test_sync_chat(self) -> None:
        provider = StubProvider()
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
        ]
        resp = provider.chat(messages)
        assert isinstance(resp, LLMResponse)
        assert "Hello" in resp.content
        assert resp.model == "stub"

    @pytest.mark.asyncio
    async def test_async_chat(self) -> None:
        provider = StubProvider()
        messages = [Message(role="user", content="Test")]
        resp = await provider.achat(messages)
        assert "Test" in resp.content


class TestProviderFactory:
    def test_get_stub_provider(self) -> None:
        provider = get_provider("stub")
        assert provider.name() == "stub"

    def test_get_unknown_provider_raises(self) -> None:
        with pytest.raises(LLMProviderError, match="Unknown"):
            get_provider("nonexistent")

    def test_env_fallback_defaults_to_stub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AMACS_LLM_PROVIDER", raising=False)
        provider = get_provider()
        assert provider.name() == "stub"
