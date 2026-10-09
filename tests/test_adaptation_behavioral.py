"""Behavioral test suite for real adaptation, quality signals, and wave re-execution."""

from __future__ import annotations

from typing import Any, Dict

from amacs.adaptive.adaptation_engine import ActionType, AdaptationAction, AdaptationEngine
from amacs.adaptive.evaluator import Evaluator, ThresholdConfig
from amacs.adaptive.monitor import Monitor
from amacs.adaptive.reconfigurator import Reconfigurator
from amacs.agents.base_agent import AgentResult, BaseAgent, SubTask
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.executor import WaveExecutor
from amacs.orchestrator.scheduler import ExecutionPlan


class FlakyTestAgent(BaseAgent):
    """Agent that fails on first call, then succeeds on retry."""

    def __init__(self, provider: Any = None, config: Any = None) -> None:
        super().__init__(provider=provider, config=config)
        self.call_count = 0

    @property
    def agent_type(self) -> str:
        return "search"

    def system_prompt(self) -> str:
        return "Search agent"

    def run(self, sub_task: SubTask, context: dict, bus: Any = None) -> AgentResult:
        self.call_count += 1
        if self.call_count == 1:
            return AgentResult(
                sub_task_id=sub_task.id,
                agent_name=self.agent_type,
                content="",
                success=False,
                error="Simulated network failure",
            )
        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name=self.agent_type,
            content="Repaired search output content",
            success=True,
        )


class RefusalTestAgent(BaseAgent):
    @property
    def agent_type(self) -> str:
        return "search"

    def system_prompt(self) -> str:
        return "Search agent"

    def run(self, sub_task: SubTask, context: dict, bus: Any = None) -> AgentResult:
        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name=self.agent_type,
            content="I cannot fulfill this request as an AI language model.",
            success=True,
        )


class RepeatedTestAgent(BaseAgent):
    @property
    def agent_type(self) -> str:
        return "search"

    def system_prompt(self) -> str:
        return "Search agent"

    def run(self, sub_task: SubTask, context: dict, bus: Any = None) -> AgentResult:
        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name=self.agent_type,
            content="Repeated line here\nRepeated line here\nRepeated line here\n",
            success=True,
        )


class LowQualityScoreAgent(BaseAgent):
    @property
    def agent_type(self) -> str:
        return "search"

    def system_prompt(self) -> str:
        return "Search agent"

    def run(self, sub_task: SubTask, context: dict, bus: Any = None) -> AgentResult:
        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name=self.agent_type,
            content="Low quality output content",
            success=True,
            metadata={"quality_score": 0.2},
        )


# ── Behavioral Tests ──────────────────────────────────────────────────────


def test_wave_1_failure_triggers_retry_and_repaired_wave_2() -> None:
    config = AMACSConfig(adaptive=True, skip_non_critical=True)
    monitor = Monitor()
    evaluator = Evaluator()
    engine = AdaptationEngine()
    reconfig = Reconfigurator(config=config)

    executor = WaveExecutor(
        config=config,
        monitor=monitor,
        evaluator=evaluator,
        engine=engine,
        reconfigurator=reconfig,
    )

    t1 = SubTask(id="search_0", label="search", description="Search task", critical=True)
    t2 = SubTask(id="write_1", label="write", description="Write task", critical=True)
    plan = ExecutionPlan(waves=[[t1], [t2]])

    agents: Dict[str, BaseAgent] = {
        "search_0": FlakyTestAgent(config=config),
        "write_1": FlakyTestAgent(config=config),
    }
    bus = CommunicationBus()

    results = executor.execute_sync(plan, agents, bus)

    assert len(results) == 2
    # Verify wave 1 result was repaired
    search_res = next(r for r in results if r.sub_task_id == "search_0")
    assert search_res.success is True
    assert search_res.agent_name != "search"

    # Verify adaptation event logged with trigger, action, target, outcome
    assert len(executor.adaptation_events) >= 1
    evt = executor.adaptation_events[0]
    assert evt.target == "search_0"
    assert evt.outcome == "retry_succeeded"
    assert evt.action in (ActionType.SWAP_AGENT.value, ActionType.RETRY_WITH_DIFFERENT_AGENT.value)


def test_swap_limit_enforced() -> None:
    config = AMACSConfig(adaptive=True, skip_non_critical=True)
    monitor = Monitor()
    evaluator = Evaluator()
    engine = AdaptationEngine()
    reconfig = Reconfigurator(config=config)
    reconfig.MAX_SWAPS_PER_TASK = 1  # Low limit for fast test
    reconfig._swap_counts["search_0"] = 1  # Pre-fill swap limit

    class AlwaysFailingAgent(BaseAgent):
        @property
        def agent_type(self) -> str:
            return "search"

        def system_prompt(self) -> str:
            return "Failing"

        def run(self, sub_task: SubTask, context: dict, bus: Any = None) -> AgentResult:
            return AgentResult(
                sub_task_id=sub_task.id,
                agent_name=self.agent_type,
                content="",
                success=False,
                error="Permanent failure",
            )

    executor = WaveExecutor(
        config=config,
        monitor=monitor,
        evaluator=evaluator,
        engine=engine,
        reconfigurator=reconfig,
    )

    t1 = SubTask(id="search_0", label="search", description="Task", critical=False)
    plan = ExecutionPlan(waves=[[t1]])
    agents: Dict[str, BaseAgent] = {"search_0": AlwaysFailingAgent(config=config)}
    bus = CommunicationBus()

    executor.execute_sync(plan, agents, bus)
    assert len(executor.adaptation_events) >= 1
    outcomes = [e.outcome for e in executor.adaptation_events]
    assert "skipped" in outcomes


def test_quality_signal_refusal_pattern_triggers_adaptation() -> None:
    config = AMACSConfig(adaptive=True, skip_non_critical=True)
    monitor = Monitor()
    evaluator = Evaluator(ThresholdConfig(detect_refusal=True))
    engine = AdaptationEngine()
    reconfig = Reconfigurator(config=config)

    executor = WaveExecutor(
        config=config,
        monitor=monitor,
        evaluator=evaluator,
        engine=engine,
        reconfigurator=reconfig,
    )

    t1 = SubTask(id="search_0", label="search", description="Task", critical=False)
    t2 = SubTask(id="write_1", label="write", description="Task 2", critical=False)
    plan = ExecutionPlan(waves=[[t1], [t2]])
    agents: Dict[str, BaseAgent] = {
        "search_0": RefusalTestAgent(config=config),
        "write_1": RefusalTestAgent(config=config),
    }
    bus = CommunicationBus()

    executor.execute_sync(plan, agents, bus)
    assert len(executor.adaptation_events) >= 1
    evt = executor.adaptation_events[0]
    assert "refusal_pattern" in evt.trigger


def test_quality_signal_repeated_output_triggers_adaptation() -> None:
    config = AMACSConfig(adaptive=True, skip_non_critical=True)
    monitor = Monitor()
    evaluator = Evaluator(ThresholdConfig(detect_repeated=True))
    engine = AdaptationEngine()
    reconfig = Reconfigurator(config=config)

    executor = WaveExecutor(
        config=config,
        monitor=monitor,
        evaluator=evaluator,
        engine=engine,
        reconfigurator=reconfig,
    )

    t1 = SubTask(id="search_0", label="search", description="Task", critical=False)
    t2 = SubTask(id="write_1", label="write", description="Task 2", critical=False)
    plan = ExecutionPlan(waves=[[t1], [t2]])
    agents: Dict[str, BaseAgent] = {
        "search_0": RepeatedTestAgent(config=config),
        "write_1": RepeatedTestAgent(config=config),
    }
    bus = CommunicationBus()

    executor.execute_sync(plan, agents, bus)
    assert len(executor.adaptation_events) >= 1
    evt = executor.adaptation_events[0]
    assert "repeated_output" in evt.trigger or "short_output" in evt.trigger


def test_quality_signal_low_judge_score_triggers_adaptation() -> None:
    config = AMACSConfig(adaptive=True, skip_non_critical=True)
    monitor = Monitor()
    evaluator = Evaluator(ThresholdConfig(min_llm_judge_score=0.6))
    engine = AdaptationEngine()
    reconfig = Reconfigurator(config=config)

    executor = WaveExecutor(
        config=config,
        monitor=monitor,
        evaluator=evaluator,
        engine=engine,
        reconfigurator=reconfig,
    )

    t1 = SubTask(id="search_0", label="search", description="Task", critical=False)
    t2 = SubTask(id="write_1", label="write", description="Task 2", critical=False)
    plan = ExecutionPlan(waves=[[t1], [t2]])
    agents: Dict[str, BaseAgent] = {
        "search_0": LowQualityScoreAgent(config=config),
        "write_1": LowQualityScoreAgent(config=config),
    }
    bus = CommunicationBus()

    executor.execute_sync(plan, agents, bus)
    assert len(executor.adaptation_events) >= 1
    evt = executor.adaptation_events[0]
    assert "low_judge_score" in evt.trigger


def test_budget_degradation_path_skips_optional_tasks() -> None:
    config = AMACSConfig(max_total_tokens=100, adaptive=True, skip_non_critical=True)
    monitor = Monitor()
    evaluator = Evaluator()
    engine = AdaptationEngine()
    reconfig = Reconfigurator(config=config)

    executor = WaveExecutor(
        config=config,
        monitor=monitor,
        evaluator=evaluator,
        engine=engine,
        reconfigurator=reconfig,
    )
    # Inject initial accumulated tokens to exceed 75% threshold
    executor.accumulated_tokens = 80

    t1 = SubTask(id="task_1", label="search", description="Task 1", critical=True)
    t2 = SubTask(id="task_2", label="search", description="Task 2", critical=False)
    plan = ExecutionPlan(waves=[[t1], [t2]])

    agents: Dict[str, BaseAgent] = {
        "task_1": FlakyTestAgent(config=config),
        "task_2": FlakyTestAgent(config=config),
    }
    bus = CommunicationBus()

    executor.execute_sync(plan, agents, bus)
    budget_evts = [e for e in executor.adaptation_events if e.action == "budget_degradation"]
    assert len(budget_evts) == 1
    assert budget_evts[0].trigger == "budget:near_limit"


def test_non_critical_task_failure_skipped_without_crashing() -> None:
    config = AMACSConfig(adaptive=False, skip_non_critical=True)
    executor = WaveExecutor(config=config)

    class FailingNonCriticalAgent(BaseAgent):
        @property
        def agent_type(self) -> str:
            return "search"

        def system_prompt(self) -> str:
            return "Failing"

        def run(self, sub_task: SubTask, context: dict, bus: Any = None) -> AgentResult:
            return AgentResult(
                sub_task_id=sub_task.id,
                agent_name=self.agent_type,
                content="",
                success=False,
                error="Non-critical failure",
            )

    t1 = SubTask(id="task_1", label="search", description="Task 1", critical=False)
    plan = ExecutionPlan(waves=[[t1]])
    agents: Dict[str, BaseAgent] = {"task_1": FailingNonCriticalAgent(config=config)}
    bus = CommunicationBus()

    results = executor.execute_sync(plan, agents, bus)
    assert len(results) == 1
    assert results[0].success is False


def test_add_agent_reconfigurator_action() -> None:
    config = AMACSConfig(adaptive=True)
    reconfig = Reconfigurator(config=config)

    t1 = SubTask(id="task_1", label="search", description="Task 1", critical=True)
    plan = ExecutionPlan(waves=[[t1]])
    agents: Dict[str, BaseAgent] = {"task_1": FlakyTestAgent(config=config)}

    action = AdaptationAction(
        action_type=ActionType.ADD_AGENT,
        target_agent_id="task_1",
        reason="Low quality output on critical task",
    )

    res = reconfig.apply([action], agents, plan, remaining_wave_idx=0)
    assert len(res.applied) == 1
    assert "task_1_aux" in agents
