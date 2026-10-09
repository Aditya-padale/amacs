"""Behavioral tests for FakeProvider and FlakyProvider."""

from __future__ import annotations

import pytest

from amacs.exceptions import LLMProviderError
from amacs.integrations.llm_providers import (
    FakeProvider,
    FlakyProvider,
    Message,
)


def test_fake_provider_call_recording_and_sequence() -> None:
    provider = FakeProvider(
        responses=["First response", "Second response"],
        default_response="Default response for {user}",
    )
    messages = [Message(role="user", content="Hello world")]

    res1 = provider.chat(messages, model="gpt-4o")
    assert res1.content == "First response"
    assert len(provider.calls) == 1
    assert provider.calls[0]["model"] == "gpt-4o"

    res2 = provider.chat(messages, model="gpt-4o")
    assert res2.content == "Second response"
    assert len(provider.calls) == 2

    res3 = provider.chat(messages, model="gpt-4o")
    assert res3.content == "Default response for Hello world"
    assert len(provider.calls) == 3


def test_fake_provider_raise_on_call() -> None:
    provider = FakeProvider(
        raise_on_call={2: LLMProviderError("Simulated API error on call 2")}
    )
    messages = [Message(role="user", content="Test")]

    res1 = provider.chat(messages)
    assert "[fake]" in res1.content

    with pytest.raises(LLMProviderError, match="Simulated API error on call 2"):
        provider.chat(messages)

    res3 = provider.chat(messages)
    assert "[fake]" in res3.content


@pytest.mark.asyncio
async def test_flaky_provider_behavior() -> None:
    fake = FakeProvider(default_response="Healthy output")
    flaky = FlakyProvider(
        base_provider=fake,
        failure_rate=0.5,
        latency_seconds=0.01,
        empty_output_rate=0.2,
        seed=42,
    )
    messages = [Message(role="user", content="Ping")]

    responses = []
    failures = 0
    for _ in range(10):
        try:
            res = await flaky.achat(messages)
            responses.append(res.content)
        except LLMProviderError:
            failures += 1

    assert failures > 0  # Seeded RNG produced failures
    assert len(responses) + failures == 10
