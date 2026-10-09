"""Regression tests for Phase 1 bug fixes (Items 7 - 13)."""

from __future__ import annotations

import pytest

from amacs.adaptive.adaptation_engine import ActionType, AdaptationAction
from amacs.adaptive.reconfigurator import Reconfigurator
from amacs.agents.base_agent import AgentResult, SubTask
from amacs.agents.writer_agent import WriterAgent
from amacs.aggregation import Aggregator
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.context_builder import ContextBuilder
from amacs.exceptions import (
    LLMProviderError,
    OrchestrationError,
)
from amacs.integrations.llm_providers import LLMResponse, StubProvider
from amacs.orchestrator.agent_selector import (
    get_alternative_agent_classes,
    get_registered_agents,
)
from amacs.orchestrator.scheduler import ExecutionPlan, Scheduler
from amacs.orchestrator.task_analyzer import TaskProfile
from amacs.orchestrator.task_decomposer import _TEMPLATES, TaskDecomposer


class TestTaskDecomposerFixes:
    """Test fixes for Item 7 (Task decomposition template mutation & symbolic deps)."""

    def test_templates_not_mutated(self) -> None:
        decomposer = TaskDecomposer()
        profile = TaskProfile(
            domain="research",
            estimated_sub_tasks=6,
            secondary_domains=["analysis", "coding"],
        )
        orig_len = len(_TEMPLATES["research"])
        decomposer.decompose(profile)
        # Verify original template in _TEMPLATES was not mutated
        assert len(_TEMPLATES["research"]) == orig_len

    def test_symbolic_dependencies_valid(self) -> None:
        decomposer = TaskDecomposer()
        profile = TaskProfile(domain="research", estimated_sub_tasks=3)
        sub_tasks = decomposer.decompose(profile)
        all_ids = {st.id for st in sub_tasks}
        for st in sub_tasks:
            for dep in st.dependencies:
                assert dep in all_ids
                assert dep != st.id


class TestSchedulerFixes:
    """Test fixes for Item 7 (Scheduler unknown dependency check)."""

    def test_unknown_dependency_raises_orchestration_error(self) -> None:
        scheduler = Scheduler()
        sub_tasks = [
            SubTask(id="task_1", label="search", description="Step 1", dependencies=[]),
            SubTask(id="task_2", label="analyze", description="Step 2", dependencies=["ghost_task"]),
        ]
        with pytest.raises(OrchestrationError, match="ghost_task"):
            scheduler.plan(sub_tasks)


class TestContextBuilderFixes:
    """Test fixes for Item 8 (Context builder truncation & token awareness)."""

    def test_context_truncation(self) -> None:
        builder = ContextBuilder(max_tokens=50)  # ~200 chars
        large_context = {
            "task_1": "A" * 500,
            "task_2": "B" * 500,
        }
        res = builder.build_context_string(large_context)
        assert "truncated" in res
        assert "task_1" in res
        assert "task_2" in res
        assert len(res) < 600


class TestRetryClassificationFixes:
    """Test fixes for Item 12 (Retry classification for fatal vs retryable errors)."""

    def test_fatal_auth_error_not_retried(self) -> None:
        class FatalProvider(StubProvider):
            def chat(self, messages, *, model=None, temperature=0.7, max_tokens=2048, timeout=None, **kwargs):
                raise LLMProviderError("401 Authentication Error: Invalid API key")

        agent = WriterAgent(provider=FatalProvider())
        task = SubTask(id="t1", label="write", description="Write something")
        res = agent.run(task, {})
        assert not res.success
        assert "401 Authentication Error" in (res.error or "")

    def test_retryable_rate_limit_retried_and_exhausted(self) -> None:
        class RateLimitProvider(StubProvider):
            def chat(self, messages, *, model=None, temperature=0.7, max_tokens=2048, timeout=None, **kwargs):
                raise LLMProviderError("429 Rate limit exceeded. Try again in 2s")

        agent = WriterAgent(provider=RateLimitProvider(), config=AMACSConfig(retry_limit=2))
        task = SubTask(id="t1", label="write", description="Write something")
        res = agent.run(task, {})
        assert not res.success
        assert "exhausted 2 retries" in (res.error or "")


class TestReconfiguratorFixes:
    """Test fixes for Item 9 (Public registry & max swaps limit)."""

    def test_public_agent_selector_api(self) -> None:
        reg = get_registered_agents()
        assert "search" in reg
        alts = get_alternative_agent_classes("search")
        assert all(cls.agent_type != "search" for cls in alts if hasattr(cls, "agent_type"))

    def test_max_swaps_limit_enforced(self) -> None:
        reconfig = Reconfigurator()
        reconfig.MAX_SWAPS_PER_TASK = 2
        agent = WriterAgent()
        agents = {"task_1": agent}
        action = AdaptationAction(
            action_type=ActionType.SWAP_AGENT,
            target_agent_id="task_1",
            reason="High error rate",
        )

        # 1st swap succeeds
        res1 = reconfig.apply([action], agents, ExecutionPlan([]))
        assert len(res1.applied) == 1

        # 2nd swap succeeds
        res2 = reconfig.apply([action], agents, ExecutionPlan([]))
        assert len(res2.applied) == 1

        # 3rd swap reaches max swaps limit and is skipped
        res3 = reconfig.apply([action], agents, ExecutionPlan([]))
        assert len(res3.skipped) == 1


class TestAggregatorFixes:
    """Test fixes for Item 10 (Non-destructive validation pass)."""

    def test_empty_validator_output_falls_back_to_unvalidated(self) -> None:
        class EmptyValidatorProvider(StubProvider):
            def chat(self, messages, *, model=None, temperature=0.7, max_tokens=2048, timeout=None, **kwargs):
                return LLMResponse(content="", model="stub", usage={})

        aggregator = Aggregator(provider=EmptyValidatorProvider())
        results = [AgentResult(sub_task_id="t1", agent_name="write", content="Original Content", success=True)]
        sub_tasks = [SubTask(id="t1", label="write", description="Task 1")]
        bus = CommunicationBus()

        output = aggregator.aggregate(results, sub_tasks, bus)
        assert output == "Original Content"
