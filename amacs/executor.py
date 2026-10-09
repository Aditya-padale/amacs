"""WaveExecutor — wave-by-wave execution engine with inter-wave adaptation.

Executes sub-task waves sequentially. Between waves, evaluates system health,
triggers the adaptive control loop, and mutates remaining waves in real time.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from amacs.adaptive.adaptation_engine import ActionType, AdaptationEngine
from amacs.adaptive.evaluator import Evaluator
from amacs.adaptive.monitor import Monitor
from amacs.adaptive.reconfigurator import Reconfigurator
from amacs.agents.base_agent import AgentResult, BaseAgent
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.exceptions import OrchestrationError
from amacs.orchestrator.scheduler import ExecutionPlan
from amacs.pricing import calculate_cost
from amacs.results import AdaptationEvent

logger = logging.getLogger("amacs.executor")


class WaveExecutor:
    """Executes an :class:`ExecutionPlan` wave by wave, adapting between waves."""

    def __init__(
        self,
        config: AMACSConfig,
        monitor: Optional[Monitor] = None,
        evaluator: Optional[Evaluator] = None,
        engine: Optional[AdaptationEngine] = None,
        reconfigurator: Optional[Reconfigurator] = None,
    ) -> None:
        self._config = config
        self._monitor = monitor
        self._evaluator = evaluator
        self._engine = engine
        self._reconfigurator = reconfigurator
        self.adaptation_events: List[AdaptationEvent] = []
        self.accumulated_tokens: int = 0
        self.accumulated_cost_usd: float = 0.0
        self._budget_degraded: bool = False

    def execute_sync(
        self,
        plan: ExecutionPlan,
        agents: Dict[str, BaseAgent],
        bus: CommunicationBus,
        on_result: Optional[Callable[[AgentResult], None]] = None,
    ) -> List[AgentResult]:
        """Execute the plan synchronously wave by wave with inter-wave adaptation."""
        all_results: List[AgentResult] = []
        context = bus.snapshot()
        max_workers = self._config.max_agents

        wave_idx = 0
        while wave_idx < len(plan.waves):
            wave = plan.waves[wave_idx]
            wave_results: List[AgentResult] = []

            if wave:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {}
                    for st in wave:
                        agent = agents.get(st.id)
                        if agent is None:
                            continue
                        future = pool.submit(agent.run, st, context, bus)
                        futures[future] = st

                    for future in as_completed(futures):
                        st = futures[future]
                        try:
                            res = future.result()
                        except Exception as exc:
                            res = AgentResult(
                                sub_task_id=st.id,
                                agent_name="unknown",
                                content="",
                                success=False,
                                error=str(exc),
                            )
                        wave_results.append(res)
                        self._process_result(res, on_result)

                # Adaptation & re-execution step for current wave
                if (
                    self._config.adaptive
                    and self._evaluator
                    and self._engine
                    and self._reconfigurator
                    and self._monitor
                ):
                    report = self._evaluator.evaluate(self._monitor)
                    if not report.system_healthy:
                        actions = self._engine.decide(report, total_agents=len(agents))
                        for act in actions:
                            if act.action_type in (
                                ActionType.RETRY_WITH_DIFFERENT_AGENT,
                                ActionType.SWAP_AGENT,
                                ActionType.SWITCH_MODEL,
                            ):
                                st_id = act.target_agent_id
                                st_obj = next((s for s in wave if s.id == st_id), None)
                                reconfig_res = self._reconfigurator.apply(
                                    [act], agents, plan, remaining_wave_idx=wave_idx + 1
                                )
                                if reconfig_res.applied and st_obj and st_id in agents:
                                    new_agent = agents[st_id]
                                    try:
                                        res_retry = new_agent.run(st_obj, context, bus)
                                    except Exception as exc:
                                        res_retry = AgentResult(
                                            sub_task_id=st_id,
                                            agent_name=new_agent.agent_type,
                                            content="",
                                            success=False,
                                            error=str(exc),
                                        )
                                    self._process_result(res_retry, on_result)
                                    outcome = "retry_succeeded" if res_retry.success else "retry_failed"

                                    for idx, existing_res in enumerate(wave_results):
                                        if existing_res.sub_task_id == st_id:
                                            wave_results[idx] = res_retry
                                            break
                                    else:
                                        wave_results.append(res_retry)

                                    trigger_val = act.params.get("trigger", "evaluation_trigger")
                                    signal_vals = act.params.get("signal_values", {})
                                    event = AdaptationEvent(
                                        wave_index=wave_idx,
                                        action_type=act.action_type.value,
                                        target_agent_id=st_id,
                                        description=f"Adapted task '{st_id}' via {act.action_type.value} ({outcome})",
                                        details={"reason": act.reason},
                                        trigger=trigger_val,
                                        signal_values=signal_vals,
                                        action=act.action_type.value,
                                        target=st_id,
                                        outcome=outcome,
                                    )
                                    self.adaptation_events.append(event)
                                else:
                                    trigger_val = act.params.get("trigger", "evaluation_trigger")
                                    signal_vals = act.params.get("signal_values", {})
                                    event = AdaptationEvent(
                                        wave_index=wave_idx,
                                        action_type=act.action_type.value,
                                        target_agent_id=st_id,
                                        description=f"Skipped adaptation for '{st_id}': max swaps or no alternative",
                                        details={"reason": act.reason},
                                        trigger=trigger_val,
                                        signal_values=signal_vals,
                                        action=act.action_type.value,
                                        target=st_id,
                                        outcome="skipped",
                                    )
                                    self.adaptation_events.append(event)
                            else:
                                reconfig_res = self._reconfigurator.apply(
                                    [act], agents, plan, remaining_wave_idx=wave_idx + 1
                                )
                                for item in reconfig_res.applied:
                                    trigger_val = act.params.get("trigger", "evaluation_trigger")
                                    signal_vals = act.params.get("signal_values", {})
                                    event = AdaptationEvent(
                                        wave_index=wave_idx,
                                        action_type=act.action_type.value,
                                        target_agent_id=act.target_agent_id or "system",
                                        description=item,
                                        details={"reason": act.reason},
                                        trigger=trigger_val,
                                        signal_values=signal_vals,
                                        action=act.action_type.value,
                                        target=act.target_agent_id or "system",
                                        outcome="applied",
                                    )
                                    self.adaptation_events.append(event)

                # Check critical failure
                for r in wave_results:
                    if not r.success:
                        st_obj = next((s for s in wave if s.id == r.sub_task_id), None)
                        if st_obj and st_obj.critical and not self._config.skip_non_critical:
                            raise OrchestrationError(
                                f"Critical sub-task '{r.sub_task_id}' failed: {r.error}"
                            )

                all_results.extend(wave_results)
                context = bus.snapshot()

            self._adapt_between_waves(wave_idx, len(plan.waves), agents, plan)
            wave_idx += 1

        return all_results

    async def execute_async(
        self,
        plan: ExecutionPlan,
        agents: Dict[str, BaseAgent],
        bus: CommunicationBus,
        on_result: Optional[Callable[[AgentResult], None]] = None,
    ) -> List[AgentResult]:
        """Execute the plan asynchronously wave by wave with inter-wave adaptation."""
        all_results: List[AgentResult] = []
        context = bus.snapshot()

        wave_idx = 0
        while wave_idx < len(plan.waves):
            wave = plan.waves[wave_idx]
            if wave:
                tasks = []
                for st in wave:
                    agent = agents.get(st.id)
                    if agent is None:
                        continue
                    tasks.append(agent.arun(st, context, bus))

                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
                wave_results: List[AgentResult] = []
                for i, res in enumerate(raw_results):
                    if isinstance(res, Exception):
                        st = wave[i]
                        res = AgentResult(
                            sub_task_id=st.id,
                            agent_name="unknown",
                            content="",
                            success=False,
                            error=str(res),
                        )
                    wave_results.append(res)  # type: ignore[arg-type]
                    self._process_result(res, on_result)  # type: ignore[arg-type]

                # Adaptation & re-execution step for current wave
                if (
                    self._config.adaptive
                    and self._evaluator
                    and self._engine
                    and self._reconfigurator
                    and self._monitor
                ):
                    report = self._evaluator.evaluate(self._monitor)
                    if not report.system_healthy:
                        actions = self._engine.decide(report, total_agents=len(agents))
                        for act in actions:
                            if act.action_type in (
                                ActionType.RETRY_WITH_DIFFERENT_AGENT,
                                ActionType.SWAP_AGENT,
                                ActionType.SWITCH_MODEL,
                            ):
                                st_id = act.target_agent_id
                                st_obj = next((s for s in wave if s.id == st_id), None)
                                reconfig_res = self._reconfigurator.apply(
                                    [act], agents, plan, remaining_wave_idx=wave_idx + 1
                                )
                                if reconfig_res.applied and st_obj and st_id in agents:
                                    new_agent = agents[st_id]
                                    try:
                                        res_retry = await new_agent.arun(st_obj, context, bus)
                                    except Exception as exc:
                                        res_retry = AgentResult(
                                            sub_task_id=st_id,
                                            agent_name=new_agent.agent_type,
                                            content="",
                                            success=False,
                                            error=str(exc),
                                        )
                                    self._process_result(res_retry, on_result)
                                    outcome = "retry_succeeded" if res_retry.success else "retry_failed"

                                    for idx, existing_res in enumerate(wave_results):
                                        if existing_res.sub_task_id == st_id:
                                            wave_results[idx] = res_retry
                                            break
                                    else:
                                        wave_results.append(res_retry)

                                    trigger_val = act.params.get("trigger", "evaluation_trigger")
                                    signal_vals = act.params.get("signal_values", {})
                                    event = AdaptationEvent(
                                        wave_index=wave_idx,
                                        action_type=act.action_type.value,
                                        target_agent_id=st_id,
                                        description=f"Adapted task '{st_id}' via {act.action_type.value} ({outcome})",
                                        details={"reason": act.reason},
                                        trigger=trigger_val,
                                        signal_values=signal_vals,
                                        action=act.action_type.value,
                                        target=st_id,
                                        outcome=outcome,
                                    )
                                    self.adaptation_events.append(event)
                                else:
                                    trigger_val = act.params.get("trigger", "evaluation_trigger")
                                    signal_vals = act.params.get("signal_values", {})
                                    event = AdaptationEvent(
                                        wave_index=wave_idx,
                                        action_type=act.action_type.value,
                                        target_agent_id=st_id,
                                        description=f"Skipped adaptation for '{st_id}': max swaps or no alternative",
                                        details={"reason": act.reason},
                                        trigger=trigger_val,
                                        signal_values=signal_vals,
                                        action=act.action_type.value,
                                        target=st_id,
                                        outcome="skipped",
                                    )
                                    self.adaptation_events.append(event)
                            else:
                                reconfig_res = self._reconfigurator.apply(
                                    [act], agents, plan, remaining_wave_idx=wave_idx + 1
                                )
                                for item in reconfig_res.applied:
                                    trigger_val = act.params.get("trigger", "evaluation_trigger")
                                    signal_vals = act.params.get("signal_values", {})
                                    event = AdaptationEvent(
                                        wave_index=wave_idx,
                                        action_type=act.action_type.value,
                                        target_agent_id=act.target_agent_id or "system",
                                        description=item,
                                        details={"reason": act.reason},
                                        trigger=trigger_val,
                                        signal_values=signal_vals,
                                        action=act.action_type.value,
                                        target=act.target_agent_id or "system",
                                        outcome="applied",
                                    )
                                    self.adaptation_events.append(event)

                for r in wave_results:
                    if not r.success:
                        st_obj = next((s for s in wave if s.id == r.sub_task_id), None)
                        if st_obj and st_obj.critical and not self._config.skip_non_critical:
                            raise OrchestrationError(
                                f"Critical sub-task '{r.sub_task_id}' failed: {r.error}"
                            )

                all_results.extend(wave_results)
                context = bus.snapshot()

            self._adapt_between_waves(wave_idx, len(plan.waves), agents, plan)
            wave_idx += 1

        return all_results

    def _process_result(
        self,
        res: AgentResult,
        on_result: Optional[Callable[[AgentResult], None]],
    ) -> None:
        if on_result:
            on_result(res)

        tokens = res.token_usage.get("total_tokens", 0)
        prompt_tok = res.token_usage.get("prompt_tokens", 0)
        comp_tok = res.token_usage.get("completion_tokens", 0)
        model_name = res.metadata.get("model", "unknown") if res.metadata else "unknown"
        price_overrides = (
            self._config.extra.get("price_overrides")
            if self._config and self._config.extra
            else None
        )

        cost = calculate_cost(model_name, prompt_tok, comp_tok, price_overrides)
        self.accumulated_tokens += tokens
        self.accumulated_cost_usd += cost

        if self._config.max_total_tokens and self.accumulated_tokens > self._config.max_total_tokens:
            from amacs.exceptions import BudgetExceededError
            raise BudgetExceededError(
                f"Total token limit exceeded: {self.accumulated_tokens} > {self._config.max_total_tokens}"
            )

        if self._config.max_cost_usd and self.accumulated_cost_usd > self._config.max_cost_usd:
            from amacs.exceptions import BudgetExceededError
            raise BudgetExceededError(
                f"Cost limit exceeded: ${self.accumulated_cost_usd:.4f} > ${self._config.max_cost_usd:.4f}"
            )

        token_rate = self.accumulated_tokens / self._config.max_total_tokens if self._config.max_total_tokens else 0.0
        cost_rate = self.accumulated_cost_usd / self._config.max_cost_usd if self._config.max_cost_usd else 0.0
        burn_rate = max(token_rate, cost_rate)
        quality_score = res.metadata.get("quality_score") if res.metadata else None

        if self._monitor is not None:
            if res.success:
                self._monitor.record_success(
                    res.sub_task_id,
                    res.agent_name,
                    res.latency_seconds,
                    tokens,
                    content=res.content,
                    quality_score=quality_score,
                    burn_rate=burn_rate,
                )
            else:
                self._monitor.record_failure(
                    res.sub_task_id,
                    res.agent_name,
                    res.latency_seconds,
                    res.error or "unknown",
                    content=res.content,
                )

    def _check_and_degrade_budget(
        self,
        current_wave_idx: int,
        plan: ExecutionPlan,
        agents: Dict[str, BaseAgent],
    ) -> None:
        """Check if budget threshold is reached and degrade remaining execution plan cleanly."""
        token_limit = self._config.max_total_tokens
        cost_limit = self._config.max_cost_usd

        near_tokens = token_limit and (self.accumulated_tokens >= token_limit * 0.75)
        near_cost = cost_limit and (self.accumulated_cost_usd >= cost_limit * 0.75)

        if (near_tokens or near_cost) and not self._budget_degraded:
            self._budget_degraded = True
            removed_count = 0
            if plan and plan.waves:
                for wave in plan.waves[current_wave_idx + 1:]:
                    to_keep = []
                    for st in wave:
                        if st.critical:
                            to_keep.append(st)
                        else:
                            agents.pop(st.id, None)
                            removed_count += 1
                    wave[:] = to_keep

            desc = (
                f"Budget threshold reached (tokens={self.accumulated_tokens}, "
                f"cost=${self.accumulated_cost_usd:.4f}). "
            )
            if removed_count > 0:
                desc += f"Skipped {removed_count} optional sub-tasks for budget preservation."
            else:
                desc += "Switching remaining tasks to cost-effective execution."

            self.adaptation_events.append(
                AdaptationEvent(
                    wave_index=current_wave_idx,
                    action_type="budget_degradation",
                    target_agent_id="system",
                    description=desc,
                    details={
                        "accumulated_tokens": self.accumulated_tokens,
                        "accumulated_cost_usd": self.accumulated_cost_usd,
                    },
                    trigger="budget:near_limit",
                    signal_values={
                        "accumulated_tokens": self.accumulated_tokens,
                        "accumulated_cost_usd": self.accumulated_cost_usd,
                    },
                    action="budget_degradation",
                    target="system",
                    outcome="applied",
                )
            )
            logger.info(desc)

    def _adapt_between_waves(
        self,
        current_wave_idx: int,
        total_waves: int,
        agents: Dict[str, BaseAgent],
        plan: ExecutionPlan,
    ) -> None:
        self._check_and_degrade_budget(current_wave_idx, plan, agents)
