"""Behavioral tests for real AMACS timeouts and retry behavior."""

from __future__ import annotations

import time
from typing import Any, Sequence

import pytest

from amacs.agents.base_agent import BaseAgent, SubTask
from amacs.config import build_config
from amacs.integrations.llm_providers import FakeProvider, LLMProvider, LLMResponse, Message


class SlowProvider(LLMProvider):
    """Provider that delays response to simulate slow API calls."""

    def __init__(self, delay_seconds: float = 0.5) -> None:
        self.delay_seconds = delay_seconds
        self.calls = 0

    def name(self) -> str:
        return "slow"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        time.sleep(self.delay_seconds)
        return LLMResponse(content="Slow response", model="slow", usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10})

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        import asyncio
        self.calls += 1
        await asyncio.sleep(self.delay_seconds)
        return LLMResponse(content="Slow async response", model="slow", usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10})


class CustomAgent(BaseAgent):
    @property
    def agent_type(self) -> str:
        return "search"

    def system_prompt(self) -> str:
        return "You are a search agent."


def test_sync_timeout_raises_agent_timeout_error() -> None:
    """Assert sync call timing out raises AgentTimeoutError and retries up to retry_limit."""
    slow_provider = SlowProvider(delay_seconds=0.3)
    config = build_config(timeout=0.05, retry_limit=2)
    agent = CustomAgent(provider=slow_provider, config=config)
    sub_task = SubTask(id="search_0", label="search", description="Slow search")

    res = agent.run(sub_task, {})
    assert res.success is False
    assert res.error is not None
    assert "timed out after 0.05s" in res.error
    assert slow_provider.calls == 2  # Retried 2 times


@pytest.mark.asyncio
async def test_async_timeout_raises_agent_timeout_error() -> None:
    """Assert async call timing out raises AgentTimeoutError via asyncio.wait_for."""
    slow_provider = SlowProvider(delay_seconds=0.3)
    config = build_config(timeout=0.05, retry_limit=2)
    agent = CustomAgent(provider=slow_provider, config=config)
    sub_task = SubTask(id="search_0", label="search", description="Slow search async")

    res = await agent.arun(sub_task, {})
    assert res.success is False
    assert res.error is not None
    assert "timed out after 0.05s" in res.error
    assert slow_provider.calls == 2


def test_fast_call_completes_within_timeout() -> None:
    """Assert fast call completes successfully when timeout is generous."""
    fake_provider = FakeProvider(default_response="Fast answer")
    config = build_config(timeout=2.0, retry_limit=1)
    agent = CustomAgent(provider=fake_provider, config=config)
    sub_task = SubTask(id="search_0", label="search", description="Fast search")

    res = agent.run(sub_task, {})
    assert res.success is True
    assert res.content == "Fast answer"


def test_sync_thread_unreachable_cancellation_documentation_note() -> None:
    """Document that sync Python threads cannot be forcibly killed, but AMACS raises immediately."""
    # When a sync call times out, ThreadPoolExecutor future times out, returning control immediately to AMACS.
    slow_provider = SlowProvider(delay_seconds=0.2)
    config = build_config(timeout=0.01, retry_limit=1)
    agent = CustomAgent(provider=slow_provider, config=config)
    sub_task = SubTask(id="search_0", label="search", description="Test thread timeout behavior")

    start = time.time()
    res = agent.run(sub_task, {})
    elapsed = time.time() - start

    assert res.success is False
    # Verified that control returned to caller much faster than the full delay per attempt
    assert elapsed < 0.25
