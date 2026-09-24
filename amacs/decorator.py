"""The ``@amacs`` decorator — the single public entry point for the framework.

Wraps any sync or async function with the full AMACS orchestration pipeline:

    @amacs(max_agents=8, strategy="performance")
    def research_task(topic: str) -> str:
        return f"Research on {topic}"

    result = research_task("Renewable Energy")
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable, TypeVar, cast

from amacs.adaptive.adaptation_engine import AdaptationEngine
from amacs.adaptive.evaluator import Evaluator, ThresholdConfig
from amacs.adaptive.monitor import Monitor
from amacs.adaptive.reconfigurator import Reconfigurator
from amacs.aggregation import Aggregator
from amacs.agents.base_agent import AgentResult
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig, build_config
from amacs.exceptions import OrchestrationError
from amacs.integrations.llm_providers import LLMProvider, get_provider
from amacs.orchestrator.agent_selector import AgentSelector
from amacs.orchestrator.scheduler import Scheduler
from amacs.orchestrator.task_analyzer import TaskAnalyzer
from amacs.orchestrator.task_decomposer import TaskDecomposer

logger = logging.getLogger("amacs.decorator")

F = TypeVar("F", bound=Callable[..., Any])


def amacs(**kwargs: Any) -> Callable[[F], F]:
    """Decorator factory that wraps a function with multi-agent orchestration.

    Parameters
    ----------
    max_agents : int
        Maximum number of concurrent agents (default 4).
    strategy : str
        One of ``"performance"``, ``"cost"``, ``"speed"`` (default ``"performance"``).
    adaptive : bool
        Enable the adaptive control loop (default ``False``).
    retry_limit : int
        Per-agent retry limit (default 3).
    timeout : float
        Per-agent timeout in seconds (default 120).
    llm_provider : str | None
        LLM provider name. Falls back to env var ``AMACS_LLM_PROVIDER``.
    llm_model : str | None
        Model name (e.g. ``"gpt-4o"``).
    skip_non_critical : bool
        Skip failed non-critical sub-tasks instead of crashing (default ``True``).

    Returns
    -------
    Callable
        The decorated function that transparently returns the aggregated result.
    """
    config = build_config(**kwargs)

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kw: Any) -> Any:
                return await _run_pipeline_async(func, args, kw, config)

            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kw: Any) -> Any:
                return _run_pipeline_sync(func, args, kw, config)

            return cast(F, sync_wrapper)

    return decorator


# ── Pipeline — sync path ──────────────────────────────────────────────────

def _run_pipeline_sync(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    config: AMACSConfig,
) -> Any:
    """Full orchestration pipeline (synchronous)."""
    provider = get_provider(config.llm_provider)

    # 1. Analyse
    analyzer = TaskAnalyzer()
    profile = analyzer.analyze(func, args, kwargs)
    logger.info("Task profile: domain=%s, complexity=%.2f", profile.domain, profile.complexity)

    # 2. Decompose
    decomposer = TaskDecomposer()
    sub_tasks = decomposer.decompose(profile)
    logger.info("Decomposed into %d sub-tasks", len(sub_tasks))

    # 3. Select agents
    selector = AgentSelector(provider=provider, config=config)
    agents = selector.select(sub_tasks)
    logger.info("Agents assigned: %s", {k: v.agent_type for k, v in agents.items()})

    # 4. Schedule
    scheduler = Scheduler(config=config)
    plan = scheduler.plan(sub_tasks)
    logger.info("Execution plan: %s", plan)

    # 5. Communication bus
    bus = CommunicationBus()

    # Store original function result as context
    original_result = func(*args, **kwargs)
    bus.publish("original_input", str(original_result), writer="decorator")

    # 6. Adaptive control (optional)
    monitor = Monitor() if config.adaptive else None
    evaluator = Evaluator() if config.adaptive else None
    engine = AdaptationEngine() if config.adaptive else None
    reconfigurator = Reconfigurator(provider=provider, config=config) if config.adaptive else None

    def on_result(result: AgentResult) -> None:
        if monitor is None:
            return
        if result.success:
            monitor.record_success(
                result.sub_task_id, result.agent_name,
                result.latency_seconds,
                result.token_usage.get("total_tokens", 0),
            )
        else:
            monitor.record_failure(
                result.sub_task_id, result.agent_name,
                result.latency_seconds,
                result.error or "unknown",
            )

    # 7. Execute
    results = scheduler.execute_sync(
        plan, agents, bus,
        skip_non_critical=config.skip_non_critical,
        on_result=on_result,
    )

    # 8. Adaptive reconfiguration (check after execution)
    if config.adaptive and monitor and evaluator and engine and reconfigurator:
        report = evaluator.evaluate(monitor)
        if not report.system_healthy:
            actions = engine.decide(report, total_agents=len(agents))
            reconfigurator.apply(actions, agents, plan)
            logger.info("Adaptive reconfiguration applied")

    # 9. Aggregate
    aggregator = Aggregator(provider=provider, config=config)
    return aggregator.aggregate(results, sub_tasks, bus)


# ── Pipeline — async path ─────────────────────────────────────────────────

async def _run_pipeline_async(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    config: AMACSConfig,
) -> Any:
    """Full orchestration pipeline (asynchronous)."""
    provider = get_provider(config.llm_provider)

    # 1. Analyse
    analyzer = TaskAnalyzer()
    profile = analyzer.analyze(func, args, kwargs)
    logger.info("Task profile: domain=%s, complexity=%.2f", profile.domain, profile.complexity)

    # 2. Decompose
    decomposer = TaskDecomposer()
    sub_tasks = decomposer.decompose(profile)
    logger.info("Decomposed into %d sub-tasks", len(sub_tasks))

    # 3. Select agents
    selector = AgentSelector(provider=provider, config=config)
    agents = selector.select(sub_tasks)
    logger.info("Agents assigned: %s", {k: v.agent_type for k, v in agents.items()})

    # 4. Schedule
    scheduler = Scheduler(config=config)
    plan = scheduler.plan(sub_tasks)
    logger.info("Execution plan: %s", plan)

    # 5. Communication bus
    bus = CommunicationBus()

    # Store original function result as context
    original_result = await func(*args, **kwargs)
    bus.publish("original_input", str(original_result), writer="decorator")

    # 6. Adaptive control (optional)
    monitor = Monitor() if config.adaptive else None

    def on_result(result: AgentResult) -> None:
        if monitor is None:
            return
        if result.success:
            monitor.record_success(
                result.sub_task_id, result.agent_name,
                result.latency_seconds,
                result.token_usage.get("total_tokens", 0),
            )
        else:
            monitor.record_failure(
                result.sub_task_id, result.agent_name,
                result.latency_seconds,
                result.error or "unknown",
            )

    # 7. Execute
    results = await scheduler.execute_async(
        plan, agents, bus,
        skip_non_critical=config.skip_non_critical,
        on_result=on_result,
    )

    # 8. Adaptive reconfiguration
    if config.adaptive and monitor:
        evaluator = Evaluator()
        report = evaluator.evaluate(monitor)
        if not report.system_healthy:
            engine = AdaptationEngine()
            actions = engine.decide(report, total_agents=len(agents))
            reconfigurator = Reconfigurator(provider=provider, config=config)
            reconfigurator.apply(actions, agents, plan)
            logger.info("Adaptive reconfiguration applied")

    # 9. Aggregate
    aggregator = Aggregator(provider=provider, config=config)
    return aggregator.aggregate(results, sub_tasks, bus)
