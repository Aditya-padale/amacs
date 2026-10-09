"""Test real inter-wave adaptation in WaveExecutor (Items 1 & 9)."""

from __future__ import annotations

from amacs.adaptive.adaptation_engine import AdaptationEngine
from amacs.adaptive.evaluator import Evaluator
from amacs.adaptive.monitor import Monitor
from amacs.adaptive.reconfigurator import Reconfigurator
from amacs.agents.base_agent import AgentResult, BaseAgent, SubTask
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.executor import WaveExecutor
from amacs.orchestrator.scheduler import ExecutionPlan


class FailingAgent(BaseAgent):
    @property
    def agent_type(self) -> str:
        return "failing"

    def system_prompt(self) -> str:
        return "Failing agent"

    def run(self, sub_task: SubTask, context: dict, bus=None) -> AgentResult:
        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name=self.agent_type,
            content="",
            success=False,
            error="Simulated failure",
        )


def test_interwave_adaptation_swaps_future_wave_agents() -> None:
    config = AMACSConfig(adaptive=True, skip_non_critical=True)
    monitor = Monitor()
    evaluator = Evaluator()
    engine = AdaptationEngine()
    reconfigurator = Reconfigurator(config=config)

    executor = WaveExecutor(
        config=config,
        monitor=monitor,
        evaluator=evaluator,
        engine=engine,
        reconfigurator=reconfigurator,
    )

    # Wave 0: task_1 (failing agent)
    # Wave 1: task_2 (initially failing agent)
    task1 = SubTask(id="task_1", label="search", description="Task 1", critical=False)
    task2 = SubTask(id="task_2", label="search", description="Task 2", critical=False)
    plan = ExecutionPlan(waves=[[task1], [task2]])

    agents: dict[str, BaseAgent] = {
        "task_1": FailingAgent(),
        "task_2": FailingAgent(),
    }
    bus = CommunicationBus()

    results = executor.execute_sync(plan, agents, bus)

    assert len(results) == 2
    # Verify adaptation event recorded
    assert len(executor.adaptation_events) >= 1
    # Verify task_2 agent was swapped before Wave 1 ran
    assert not isinstance(agents["task_2"], FailingAgent)
