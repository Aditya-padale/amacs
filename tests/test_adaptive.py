"""Tests for the adaptive control loop — monitor, evaluator, engine, reconfigurator."""

from __future__ import annotations

from amacs.adaptive.adaptation_engine import ActionType, AdaptationEngine
from amacs.adaptive.evaluator import Evaluator, HealthStatus, ThresholdConfig
from amacs.adaptive.monitor import Monitor
from amacs.adaptive.reconfigurator import Reconfigurator
from amacs.agents.base_agent import SubTask
from amacs.agents.search_agent import SearchAgent
from amacs.agents.writer_agent import WriterAgent
from amacs.orchestrator.scheduler import ExecutionPlan


class TestMonitor:
    def test_record_success(self) -> None:
        m = Monitor()
        m.record_success("agent_1", "search", latency=1.5, tokens=100)
        metrics = m.get_agent_metrics("agent_1")
        assert metrics is not None
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 0
        assert metrics.total_latency == 1.5
        assert metrics.total_tokens == 100

    def test_record_failure(self) -> None:
        m = Monitor()
        m.record_failure("agent_1", "search", latency=2.0, error="timeout")
        metrics = m.get_agent_metrics("agent_1")
        assert metrics is not None
        assert metrics.failed_calls == 1
        assert metrics.last_error == "timeout"

    def test_snapshot(self) -> None:
        m = Monitor()
        m.record_success("a1", "search", 1.0, 50)
        m.record_success("a2", "write", 2.0, 100)
        snap = m.snapshot()
        assert snap.active_agents == 2
        assert snap.total_calls == 2
        assert snap.total_tokens == 150


class TestEvaluator:
    def test_healthy_agent(self) -> None:
        m = Monitor()
        m.record_success("a1", "search", 0.5, 50)
        ev = Evaluator()
        report = ev.evaluate(m)
        assert report.system_healthy
        assert len(report.underperforming) == 0

    def test_failing_agent_by_failures(self) -> None:
        m = Monitor()
        m.record_failure("a1", "search", 1.0, "err1")
        m.record_failure("a1", "search", 1.0, "err2")
        ev = Evaluator(ThresholdConfig(max_failures=2))
        report = ev.evaluate(m)
        assert not report.system_healthy
        assert "a1" in report.underperforming
        assert report.evaluations["a1"].status == HealthStatus.FAILING

    def test_degraded_agent_by_latency(self) -> None:
        m = Monitor()
        m.record_success("a1", "search", 50.0, 50)
        ev = Evaluator(ThresholdConfig(max_latency_seconds=30.0))
        report = ev.evaluate(m)
        assert "a1" in report.underperforming
        assert report.evaluations["a1"].status == HealthStatus.DEGRADED


class TestAdaptationEngine:
    def test_healthy_system_no_action(self) -> None:
        m = Monitor()
        m.record_success("a1", "search", 0.5, 50)
        ev = Evaluator()
        report = ev.evaluate(m)
        engine = AdaptationEngine()
        actions = engine.decide(report)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.NO_ACTION

    def test_failing_agent_swap_action(self) -> None:
        m = Monitor()
        m.record_failure("a1", "search", 1.0, "err1")
        m.record_failure("a1", "search", 1.0, "err2")
        ev = Evaluator(ThresholdConfig(max_failures=2))
        report = ev.evaluate(m)
        engine = AdaptationEngine()
        actions = engine.decide(report, total_agents=2)
        swap_actions = [a for a in actions if a.action_type == ActionType.SWAP_AGENT]
        assert len(swap_actions) >= 1
        assert swap_actions[0].target_agent_id == "a1"


class TestReconfigurator:
    def test_swap_agent(self) -> None:
        agents = {"a1": SearchAgent()}
        plan = ExecutionPlan(
            waves=[[SubTask(id="a1", label="search", description="test")]]
        )
        reconf = Reconfigurator()

        from amacs.adaptive.adaptation_engine import AdaptationAction

        actions = [
            AdaptationAction(
                action_type=ActionType.SWAP_AGENT,
                target_agent_id="a1",
                reason="test",
                params={"original_type": "search"},
            )
        ]
        result = reconf.apply(actions, agents, plan)
        assert len(result.applied) == 1
        # agent should have been swapped to a different type
        assert agents["a1"].agent_type != "search"

    def test_skip_task(self) -> None:
        agents = {"a1": SearchAgent(), "a2": WriterAgent()}
        plan = ExecutionPlan(
            waves=[
                [SubTask(id="a1", label="search", description="test")],
                [SubTask(id="a2", label="write", description="test")],
            ]
        )
        from amacs.adaptive.adaptation_engine import AdaptationAction

        actions = [
            AdaptationAction(
                action_type=ActionType.SKIP_TASK,
                target_agent_id="a2",
                reason="degraded",
            )
        ]
        reconf = Reconfigurator()
        result = reconf.apply(actions, agents, plan, remaining_wave_idx=1)
        assert len(result.applied) == 1
        assert "a2" not in agents
