"""Tests for LLM provider abstraction."""

from __future__ import annotations

import pytest

from amacs.exceptions import LLMProviderError
from amacs.integrations.llm_providers import (
    LLMResponse,
    Message,
    StubProvider,
    get_provider,
)


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
