"""Test strategy resolution, timeouts, and budget enforcement (Items 2 & 3)."""

from __future__ import annotations

import pytest

from amacs.agents.base_agent import AgentResult, SubTask
from amacs.agents.writer_agent import WriterAgent
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.exceptions import BudgetExceededError
from amacs.executor import WaveExecutor
from amacs.orchestrator.scheduler import ExecutionPlan
from amacs.strategy import StrategyPolicy


def test_strategy_model_resolution() -> None:
    assert StrategyPolicy.resolve_model("performance", "openai") == "gpt-4o"
    assert StrategyPolicy.resolve_model("cost", "openai") == "gpt-4o-mini"
    assert StrategyPolicy.resolve_model("speed", "anthropic") == "claude-3-5-haiku-20241022"
    assert StrategyPolicy.resolve_model("performance", "openai", requested_model="custom-model") == "custom-model"


def test_token_budget_exceeded_raises_budget_error() -> None:
    config = AMACSConfig(max_total_tokens=100)
    executor = WaveExecutor(config=config)

    task1 = SubTask(id="t1", label="write", description="Task 1")
    plan = ExecutionPlan(waves=[[task1]])

    class HeavyTokenAgent(WriterAgent):
        def run(self, sub_task, context, bus=None):
            return AgentResult(
                sub_task_id=sub_task.id,
                agent_name="heavy",
                content="Huge output",
                success=True,
                token_usage={"total_tokens": 500, "prompt_tokens": 300, "completion_tokens": 200},
            )

    agents = {"t1": HeavyTokenAgent()}
    bus = CommunicationBus()

    with pytest.raises(BudgetExceededError, match="Total token limit exceeded"):
        executor.execute_sync(plan, agents, bus)


def test_cost_budget_exceeded_raises_budget_error() -> None:
    config = AMACSConfig(max_cost_usd=0.001)
    executor = WaveExecutor(config=config)

    task1 = SubTask(id="t1", label="write", description="Task 1")
    plan = ExecutionPlan(waves=[[task1]])

    class ExpensiveAgent(WriterAgent):
        def run(self, sub_task, context, bus=None):
            return AgentResult(
                sub_task_id=sub_task.id,
                agent_name="expensive",
                content="Expensive output",
                success=True,
                token_usage={"prompt_tokens": 1000000, "completion_tokens": 1000000, "total_tokens": 2000000},
            )

    agents = {"t1": ExpensiveAgent()}
    bus = CommunicationBus()

    with pytest.raises(BudgetExceededError, match="Cost limit exceeded"):
        executor.execute_sync(plan, agents, bus)
