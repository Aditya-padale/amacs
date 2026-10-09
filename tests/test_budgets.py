"""Behavioral tests for AMACS budget limits, pricing calculation, and clean degradation."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from amacs.agents.base_agent import AgentResult, SubTask
from amacs.config import build_config
from amacs.decorator import amacs
from amacs.exceptions import BudgetExceededError
from amacs.executor import WaveExecutor
from amacs.integrations.llm_providers import (
    LLMProvider,
    LLMResponse,
    Message,
    register_provider,
)
from amacs.orchestrator.scheduler import ExecutionPlan
from amacs.pricing import calculate_cost


class HighTokenProvider(LLMProvider):
    def name(self) -> str:
        return "high_token"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(
            content="Token heavy response",
            model="gpt-4o",
            usage={"prompt_tokens": 500, "completion_tokens": 500, "total_tokens": 1000},
        )

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return self.chat(messages, model=model, **kwargs)


class HighCostProvider(LLMProvider):
    def name(self) -> str:
        return "high_cost"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # 1M prompt + 1M completion on gpt-4o = $12.50
        return LLMResponse(
            content="Expensive response",
            model="gpt-4o",
            usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "total_tokens": 2_000_000},
        )

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return self.chat(messages, model=model, **kwargs)


register_provider("high_token", HighTokenProvider)
register_provider("high_cost", HighCostProvider)


def test_calculate_cost_with_model_rates_and_user_overrides() -> None:
    cost_gpt4o = calculate_cost("gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert pytest.approx(cost_gpt4o, 0.001) == 12.50

    cost_mini = calculate_cost("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert pytest.approx(cost_mini, 0.001) == 0.75

    overrides = {"custom-model": {"prompt": 5.00, "completion": 20.00}}
    cost_custom = calculate_cost("custom-model", 500_000, 500_000, price_overrides=overrides)
    assert pytest.approx(cost_custom, 0.001) == 12.50


def test_max_total_tokens_budget_exceeded() -> None:
    @amacs(max_total_tokens=250, return_details=True, llm_provider="high_token")
    def task(topic: str) -> str:
        return f"Task: {topic}"

    with pytest.raises(BudgetExceededError, match="Total token limit exceeded"):
        task("Budget test")


def test_max_cost_usd_budget_exceeded() -> None:
    @amacs(max_cost_usd=1.00, return_details=True, llm_provider="high_cost")
    def task(topic: str) -> str:
        return f"Task: {topic}"

    with pytest.raises(BudgetExceededError, match="Cost limit exceeded"):
        task("Cost test")


def test_budget_threshold_clean_degradation_event() -> None:
    config = build_config(max_total_tokens=200)
    executor = WaveExecutor(config=config)
    res = AgentResult(
        sub_task_id="search_0",
        agent_name="search",
        content="Search output",
        token_usage={"prompt_tokens": 80, "completion_tokens": 80, "total_tokens": 160},
        metadata={"model": "gpt-4o"},
    )

    executor._process_result(res, on_result=None)
    plan = ExecutionPlan(
        waves=[
            [SubTask(id="search_0", label="search", description="S")],
            [SubTask(id="write_1", label="write", description="W", critical=False)],
        ]
    )

    executor._check_and_degrade_budget(current_wave_idx=0, plan=plan, agents={})

    assert len(executor.adaptation_events) > 0
    assert executor.adaptation_events[0].action_type == "budget_degradation"
    assert len(plan.waves[1]) == 0  # Non-critical task in wave 2 was trimmed for budget preservation
