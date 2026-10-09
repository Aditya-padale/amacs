"""Pipeline — unified orchestration pipeline runner for sync and async tasks."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from amacs.adaptive.adaptation_engine import AdaptationEngine
from amacs.adaptive.evaluator import Evaluator
from amacs.adaptive.monitor import Monitor
from amacs.adaptive.reconfigurator import Reconfigurator
from amacs.agents.base_agent import AgentResult, BaseAgent
from amacs.aggregation import Aggregator
from amacs.cache import ResponseCache
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.executor import WaveExecutor
from amacs.integrations.llm_providers import LLMProvider, Message, get_provider
from amacs.orchestrator.agent_selector import AgentSelector
from amacs.orchestrator.scheduler import ExecutionPlan, Scheduler
from amacs.orchestrator.task_analyzer import TaskAnalyzer
from amacs.orchestrator.task_decomposer import TaskDecomposer
from amacs.results import AMACSResult
from amacs.tracing import Tracer

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
        self._tracer: Optional[Tracer] = None
        self._cache: Optional[ResponseCache] = None

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
        tracer = Tracer(on_event=self.config.on_event)
        self._tracer = tracer
        analysis_span = tracer.start_span("analyse", "pipeline_stage")
        # 1. Analyse
        analyzer = TaskAnalyzer()
        profile = analyzer.analyze(func, args, kwargs)
        tracer.end_span(analysis_span, {"domain": profile.domain})
        logger.info("Task profile: domain=%s, complexity=%.2f", profile.domain, profile.complexity)

        # 2. Decompose (planner="template" | "llm")
        decompose_span = tracer.start_span("decompose", "pipeline_stage")
        if self.config.planner == "llm":
            from amacs.orchestrator.llm_decomposer import LLMTaskDecomposer
            decomposer: Any = LLMTaskDecomposer(provider=self.provider, config=self.config)
        else:
            decomposer = TaskDecomposer()
        sub_tasks = decomposer.decompose(profile)
        tracer.end_span(decompose_span, {"task_count": len(sub_tasks)})
        logger.info("Decomposed into %d sub-tasks", len(sub_tasks))

        # 3. Select agents
        selector = AgentSelector(provider=self.provider, config=self.config)
        agents = selector.select(sub_tasks)
        if self._cache is None:
            cache_path = self.config.cache if isinstance(self.config.cache, str) else None
            self._cache = ResponseCache(enabled=bool(self.config.cache), filepath=cache_path)
        for agent in agents.values():
            agent.attach_tracer(tracer)
            agent.attach_cache(self._cache)
        logger.info("Agents assigned: %s", {k: v.agent_type for k, v in agents.items()})

        # 4. Schedule
        schedule_span = tracer.start_span("schedule", "pipeline_stage")
        scheduler = Scheduler(config=self.config)
        plan = scheduler.plan(sub_tasks)
        tracer.end_span(schedule_span, {"wave_count": len(plan.waves)})
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
            tracer=tracer,
            on_event=self.config.on_event,
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

        # Output schema validation and retry
        if self.config.output_schema is not None:
            schema_cls = self.config.output_schema
            try:
                cleaned = final_output.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                elif cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1].split("```")[0].strip()

                data = json.loads(cleaned)
                if hasattr(schema_cls, "model_validate"):
                    parsed = schema_cls.model_validate(data)
                    final_output = json.dumps(parsed.model_dump())
            except Exception as first_exc:
                logger.warning("Output schema validation failed (%s). Retrying once with validation error feedback...", first_exc)
                retry_messages = [
                    Message(role="system", content="You are a schema formatting assistant. Output valid JSON matching the requested schema."),
                    Message(role="user", content=f"Your output failed validation schema error: {first_exc}\n\nOriginal Output:\n{final_output}\n\nPlease output valid JSON only:"),
                ]
                try:
                    resp = self.provider.chat(retry_messages)
                    cleaned = resp.content.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                    elif cleaned.startswith("```"):
                        cleaned = cleaned.split("```")[1].split("```")[0].strip()
                    data = json.loads(cleaned)
                    if hasattr(schema_cls, "model_validate"):
                        parsed = schema_cls.model_validate(data)
                        final_output = json.dumps(parsed.model_dump())
                    else:
                        final_output = resp.content
                except Exception as second_exc:
                    logger.warning("Output schema retry failed: %s", second_exc)

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
            trace=self._tracer or Tracer(),
            cost_report={
                "total_tokens": wave_executor.accumulated_tokens,
                "total_usd": wave_executor.accumulated_cost_usd,
                "by_agent": wave_executor.cost_by_agent,
                "by_stage": wave_executor.cost_by_stage,
            },
        )

        if self.config.verbose:
            amacs_res.print_communication_log()

        if self.config.otel_endpoint:
            amacs_res.trace.export_opentelemetry(self.config.otel_endpoint)

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

        # Decorator contract input mode handling
        if self.config.task:
            task_val = self.config.task
        elif self.config.input_mode == "return_value_as_prompt":
            task_val = str(func(*args, **kwargs))
        else:
            task_val = str(func(*args, **kwargs))

        bus.publish("original_input", task_val, writer="decorator")

        # Coordination modes: "pipeline" | "debate" | "manager_worker"
        if self.config.mode == "debate":
            from amacs.coordination.debate import DebateCoordinator
            debate_coord = DebateCoordinator(provider=self.provider, config=self.config)
            debate_coord.attach_runtime(self._tracer, self._cache)
            results: list[AgentResult] = []
            for st in sub_tasks:
                span = self._tracer.start_span(st.id, "coordination_task", {"mode": "debate"}) if self._tracer else None
                res = debate_coord.coordinate(st, bus.snapshot(), bus)
                results.append(res)
                wave_executor._process_result(res, on_result)
                if span and self._tracer:
                    self._tracer.end_span(span, {"success": res.success, "status": "done"})
        elif self.config.mode == "manager_worker":
            from amacs.coordination.manager_worker import ManagerWorkerCoordinator
            mw_coord = ManagerWorkerCoordinator(provider=self.provider, config=self.config)
            mw_coord.attach_runtime(self._tracer, self._cache)
            results = []
            for st in sub_tasks:
                span = self._tracer.start_span(st.id, "coordination_task", {"mode": "manager_worker"}) if self._tracer else None
                res = mw_coord.coordinate(st, bus.snapshot(), bus)
                results.append(res)
                wave_executor._process_result(res, on_result)
                if span and self._tracer:
                    self._tracer.end_span(span, {"success": res.success, "status": "done"})
        else:
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

        if self.config.task:
            task_val = self.config.task
        elif self.config.input_mode == "return_value_as_prompt":
            if asyncio.iscoroutinefunction(func):
                task_val = str(await func(*args, **kwargs))
            else:
                task_val = str(func(*args, **kwargs))
        else:
            if asyncio.iscoroutinefunction(func):
                task_val = str(await func(*args, **kwargs))
            else:
                task_val = str(func(*args, **kwargs))

        bus.publish("original_input", task_val, writer="decorator")

        if self.config.mode == "debate":
            from amacs.coordination.debate import DebateCoordinator
            async_debate_coord = DebateCoordinator(provider=self.provider, config=self.config)
            async_debate_coord.attach_runtime(self._tracer, self._cache)
            results = []
            for st in sub_tasks:
                span = self._tracer.start_span(st.id, "coordination_task", {"mode": "debate"}) if self._tracer else None
                res = await async_debate_coord.acoordinate(st, bus.snapshot(), bus)
                results.append(res)
                wave_executor._process_result(res, on_result)
                if span and self._tracer:
                    self._tracer.end_span(span, {"success": res.success, "status": "done"})
        elif self.config.mode == "manager_worker":
            from amacs.coordination.manager_worker import ManagerWorkerCoordinator
            async_mw_coord = ManagerWorkerCoordinator(provider=self.provider, config=self.config)
            async_mw_coord.attach_runtime(self._tracer, self._cache)
            results = []
            for st in sub_tasks:
                span = self._tracer.start_span(st.id, "coordination_task", {"mode": "manager_worker"}) if self._tracer else None
                res = await async_mw_coord.acoordinate(st, bus.snapshot(), bus)
                results.append(res)
                wave_executor._process_result(res, on_result)
                if span and self._tracer:
                    self._tracer.end_span(span, {"success": res.success, "status": "done"})
        else:
            results = await wave_executor.execute_async(
                plan,
                agents,
                bus,
                on_result=on_result,
            )

        return self._finalize(results, sub_tasks, bus, plan, wave_executor, monitor)
