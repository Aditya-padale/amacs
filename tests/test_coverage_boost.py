"""Additional tests to reach 85%+ code coverage across provider, scheduler, reconfigurator, and results modules."""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from amacs.adaptive.adaptation_engine import ActionType, AdaptationAction
from amacs.adaptive.reconfigurator import Reconfigurator
from amacs.agents.base_agent import AgentResult, BaseAgent, SubTask
from amacs.communication import CommunicationBus, LogEntry
from amacs.config import AMACSConfig
from amacs.integrations.llm_providers import (
    AnthropicProvider,
    Message,
    OpenAIProvider,
)
from amacs.orchestrator.scheduler import ExecutionPlan, Scheduler
from amacs.results import AMACSResult


class SimpleAgent(BaseAgent):
    @property
    def agent_type(self) -> str:
        return "search"

    def system_prompt(self) -> str:
        return "Search prompt"

    def run(self, sub_task: SubTask, context: dict, bus: Any = None) -> AgentResult:
        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name="search",
            content="Done",
            success=True,
        )

    async def arun(self, sub_task: SubTask, context: dict, bus: Any = None) -> AgentResult:
        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name="search",
            content="Done async",
            success=True,
        )


def test_scheduler_execute_sync_and_async() -> None:
    config = AMACSConfig()
    scheduler = Scheduler(config=config)
    t1 = SubTask(id="t1", label="search", description="test", critical=True)
    plan = ExecutionPlan(waves=[[t1]])
    agents: Dict[str, BaseAgent] = {"t1": SimpleAgent()}
    bus = CommunicationBus()

    res_sync = scheduler.execute_sync(plan, agents, bus, skip_non_critical=True)
    assert len(res_sync) == 1
    assert res_sync[0].success is True

    res_async = asyncio.run(
        scheduler.execute_async(plan, agents, bus, skip_non_critical=True)
    )
    assert len(res_async) == 1
    assert res_async[0].success is True


def test_amacs_result_print_methods(capsys: pytest.CaptureFixture[str]) -> None:
    res = AgentResult(
        sub_task_id="t1",
        agent_name="search",
        content="Response content line 1\nResponse content line 2",
        success=True,
        latency_seconds=0.5,
        token_usage={"total_tokens": 10},
    )
    log_entry = LogEntry(timestamp=0.0, writer="search", key="t1", value="A" * 150)
    amacs_res = AMACSResult(
        final_output="Final output text",
        agent_results=[res],
        communication_log=[log_entry],
    )

    amacs_res.print_agent_responses()
    captured = capsys.readouterr()
    assert "AGENT RESPONSES" in captured.out
    assert "t1" in captured.out

    amacs_res.print_communication_log()
    captured2 = capsys.readouterr()
    assert "COMMUNICATION LOG" in captured2.out


def test_reconfigurator_remove_and_switch_model() -> None:
    reconfig = Reconfigurator()
    agents: Dict[str, BaseAgent] = {"t1": SimpleAgent()}
    plan = ExecutionPlan(waves=[[SubTask(id="t1", label="search", description="d")]])

    action_remove = AdaptationAction(
        action_type=ActionType.REMOVE_AGENT,
        target_agent_id="t1",
        reason="remove test",
    )
    res = reconfig.apply([action_remove], agents, plan)
    assert len(res.applied) == 1
    assert "t1" not in agents

    action_switch = AdaptationAction(
        action_type=ActionType.SWITCH_MODEL,
        target_agent_id="non_existent",
        reason="switch model test",
    )
    res_switch = reconfig.apply([action_switch], agents, plan)
    assert len(res_switch.skipped) == 1


def test_openai_provider_mocked() -> None:
    with patch("openai.OpenAI"), patch("openai.AsyncOpenAI"):
        provider = OpenAIProvider(api_key="mock_key")
        assert provider.name() == "openai"

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Mocked response"))]
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_resp
        provider._client = mock_client

        msg = Message(role="user", content="hello")
        res = provider.chat([msg])
        assert res.content == "Mocked response"
        assert res.model == "gpt-4o"


def test_anthropic_provider_mocked() -> None:
    with patch("anthropic.Anthropic"), patch("anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="mock_key")
        assert provider.name() == "anthropic"

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Anthropic response")]
        mock_resp.model = "claude-3-5-sonnet-20241022"
        mock_resp.usage.input_tokens = 12
        mock_resp.usage.output_tokens = 8
        mock_client.messages.create.return_value = mock_resp
        provider._client = mock_client

        msg = Message(role="user", content="hello")
        res = provider.chat([msg])
        assert res.content == "Anthropic response"
