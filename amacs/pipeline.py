"""Pipeline — unified orchestration pipeline runner for sync and async tasks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from amacs.adaptive.adaptation_engine import AdaptationEngine
from amacs.adaptive.evaluator import Evaluator
from amacs.adaptive.monitor import Monitor
from amacs.adaptive.reconfigurator import Reconfigurator
from amacs.agents.base_agent import AgentResult, BaseAgent
from amacs.aggregation import Aggregator
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.executor import WaveExecutor
from amacs.integrations.llm_providers import LLMProvider, get_provider
from amacs.orchestrator.agent_selector import AgentSelector
from amacs.orchestrator.scheduler import ExecutionPlan, Scheduler
from amacs.orchestrator.task_analyzer import TaskAnalyzer
from amacs.orchestrator.task_decomposer import TaskDecomposer
from amacs.results import AMACSResult

logger = logging.getLogger("amacs.pipeline")


class Pipeline:
    """Encapsulates the end-to-end AMACS orchestration pipeline."""

    def __init__(
        self,
        config: AMACSConfig,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        self.config = config
        self.provider = provider or get_provider(config.llm_provider)

    def _prepare(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[
        CommunicationBus,
        ExecutionPlan,
        Dict[str, BaseAgent],
        Any,  # sub_tasks
        WaveExecutor,
        Monitor | None,
        Callable[[AgentResult], None],
    ]:
        # 1. Analyse
        analyzer = TaskAnalyzer()
        profile = analyzer.analyze(func, args, kwargs)
        logger.info("Task profile: domain=%s, complexity=%.2f", profile.domain, profile.complexity)

        # 2. Decompose
        decomposer = TaskDecomposer()
        sub_tasks = decomposer.decompose(profile)
        logger.info("Decomposed into %d sub-tasks", len(sub_tasks))

        # 3. Select agents
        selector = AgentSelector(provider=self.provider, config=self.config)
        agents = selector.select(sub_tasks)
        logger.info("Agents assigned: %s", {k: v.agent_type for k, v in agents.items()})

        # 4. Schedule
        scheduler = Scheduler(config=self.config)
        plan = scheduler.plan(sub_tasks)
        logger.info("Execution plan: %s", plan)

        # 5. Communication bus
        bus = CommunicationBus()

        # 6. Adaptive control setup
        monitor = Monitor() if self.config.adaptive else None
        evaluator = Evaluator() if self.config.adaptive else None
        engine = AdaptationEngine() if self.config.adaptive else None
        reconfigurator = Reconfigurator(provider=self.provider, config=self.config) if self.config.adaptive else None

        wave_executor = WaveExecutor(
            config=self.config,
            monitor=monitor,
            evaluator=evaluator,
            engine=engine,
            reconfigurator=reconfigurator,
        )

        def on_result(result: AgentResult) -> None:
            status_str = "SUCCESS" if result.success else f"FAILED ({result.error})"
            logger.info(
                "[Agent Response] agent='%s' sub_task='%s' status=%s: %.200s",
                result.agent_name,
                result.sub_task_id,
                status_str,
                result.content,
            )
            if self.config.verbose:
                print(f"\n🤖 [Agent Response] Agent '{result.agent_name}' (task='{result.sub_task_id}') [{status_str}]:")
                for line in result.content.splitlines():
                    print(f"   │ {line}")

        return bus, plan, agents, sub_tasks, wave_executor, monitor, on_result

    def _finalize(
        self,
        results: list[AgentResult],
        sub_tasks: list[Any],
        bus: CommunicationBus,
        plan: ExecutionPlan,
        wave_executor: WaveExecutor,
        monitor: Monitor | None,
    ) -> AMACSResult:
        aggregator = Aggregator(provider=self.provider, config=self.config)
        final_output = aggregator.aggregate(results, sub_tasks, bus)

        amacs_res = AMACSResult(
            final_output=final_output,
            agent_results=results,
            communication_log=bus.get_log(),
            sub_tasks=sub_tasks,
            execution_plan=plan,
            system_snapshot=monitor.snapshot() if monitor else None,
            adaptation_events=wave_executor.adaptation_events,
            raw_merge=aggregator.raw_merge,
            validation_report=aggregator.validation_report,
        )

        if self.config.verbose:
            amacs_res.print_communication_log()

        return amacs_res

    def run(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> AMACSResult:
        """Run the orchestration pipeline synchronously."""
        bus, plan, agents, sub_tasks, wave_executor, monitor, on_result = self._prepare(
            func, args, kwargs
        )

        # Store original function result as context
        original_result = func(*args, **kwargs)
        bus.publish("original_input", str(original_result), writer="decorator")

        results = wave_executor.execute_sync(
            plan,
            agents,
            bus,
            on_result=on_result,
        )

        return self._finalize(results, sub_tasks, bus, plan, wave_executor, monitor)

    async def arun(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> AMACSResult:
        """Run the orchestration pipeline asynchronously."""
        bus, plan, agents, sub_tasks, wave_executor, monitor, on_result = self._prepare(
            func, args, kwargs
        )

        # Store original function result as context
        if asyncio.iscoroutinefunction(func):
            original_result = await func(*args, **kwargs)
        else:
            original_result = func(*args, **kwargs)

        bus.publish("original_input", str(original_result), writer="decorator")

        results = await wave_executor.execute_async(
            plan,
            agents,
            bus,
            on_result=on_result,
        )

        return self._finalize(results, sub_tasks, bus, plan, wave_executor, monitor)
