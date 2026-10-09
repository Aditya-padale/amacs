"""WaveExecutor — wave-by-wave execution engine with inter-wave adaptation.

Executes sub-task waves sequentially. Between waves, evaluates system health,
triggers the adaptive control loop, and mutates remaining waves in real time.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from amacs.adaptive.adaptation_engine import AdaptationEngine
from amacs.adaptive.evaluator import Evaluator
from amacs.adaptive.monitor import Monitor
from amacs.adaptive.reconfigurator import Reconfigurator
from amacs.agents.base_agent import AgentResult, BaseAgent
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.exceptions import OrchestrationError
from amacs.orchestrator.scheduler import ExecutionPlan
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

            # Inter-wave adaptation step
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
        self.accumulated_tokens += tokens
        self.accumulated_cost_usd += (prompt_tok * 0.000002) + (comp_tok * 0.000006)

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

        if self._monitor is not None:
            if res.success:
                self._monitor.record_success(
                    res.sub_task_id,
                    res.agent_name,
                    res.latency_seconds,
                    tokens,
                )
            else:
                self._monitor.record_failure(
                    res.sub_task_id,
                    res.agent_name,
                    res.latency_seconds,
                    res.error or "unknown",
                )

    def _adapt_between_waves(
        self,
        current_wave_idx: int,
        total_waves: int,
        agents: Dict[str, BaseAgent],
        plan: ExecutionPlan,
    ) -> None:
        if not (self._config.adaptive and self._monitor and self._evaluator and self._engine and self._reconfigurator):
            return

        if current_wave_idx >= total_waves - 1:
            return  # No remaining waves to adapt

        report = self._evaluator.evaluate(self._monitor)
        if not report.system_healthy:
            actions = self._engine.decide(report, total_agents=len(agents))
            reconfig_res = self._reconfigurator.apply(
                actions, agents, plan, remaining_wave_idx=current_wave_idx + 1
            )
            for item in reconfig_res.applied:
                self.adaptation_events.append(
                    AdaptationEvent(
                        wave_index=current_wave_idx,
                        action_type="applied",
                        target_agent_id="system",
                        description=item,
                    )
                )
            logger.info("Inter-wave adaptation applied after wave %d: %s", current_wave_idx, reconfig_res)
