"""Scheduler — plans execution order and parallelism for the sub-task DAG.

Respects the ``max_agents`` limit from :class:`AMACSConfig`.  Provides both
sync (``ThreadPoolExecutor``) and async (``asyncio.gather``) execution paths.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Set

from amacs.agents.base_agent import AgentResult, BaseAgent, SubTask
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.exceptions import OrchestrationError


class ExecutionPlan:
    """An ordered list of *waves* (groups of sub-tasks that can run in parallel)."""

    def __init__(self, waves: List[List[SubTask]]) -> None:
        self.waves = waves

    def __repr__(self) -> str:
        return f"ExecutionPlan(waves={len(self.waves)}, total={sum(len(w) for w in self.waves)})"


class Scheduler:
    """Builds an :class:`ExecutionPlan` from a DAG of sub-tasks and executes it."""

    def __init__(self, config: Optional[AMACSConfig] = None) -> None:
        self._config = config
        self._max_agents = config.max_agents if config else 4

    # ── Planning ──────────────────────────────────────────────────────

    def plan(self, sub_tasks: List[SubTask]) -> ExecutionPlan:
        """Topological-sort the sub-task DAG into parallel waves."""
        task_map: Dict[str, SubTask] = {st.id: st for st in sub_tasks}

        # Build in-degree map
        in_degree: Dict[str, int] = {st.id: 0 for st in sub_tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)
        for st in sub_tasks:
            for dep in st.dependencies:
                if dep not in task_map:
                    raise OrchestrationError(
                        f"SubTask '{st.id}' references unknown dependency '{dep}'"
                    )
                in_degree[st.id] += 1
                dependents[dep].append(st.id)

        waves: List[List[SubTask]] = []
        remaining: Set[str] = set(task_map.keys())

        while remaining:
            # collect tasks with no outstanding dependencies
            ready = [tid for tid in remaining if in_degree[tid] == 0]
            if not ready:
                raise OrchestrationError(
                    "Circular dependency detected in sub-task DAG. "
                    f"Remaining tasks: {remaining}"
                )
            # respect max_agents: chunk the ready list
            for i in range(0, len(ready), self._max_agents):
                chunk = ready[i : i + self._max_agents]
                waves.append([task_map[tid] for tid in chunk])

            for tid in ready:
                remaining.discard(tid)
                for dependent in dependents[tid]:
                    in_degree[dependent] -= 1

        return ExecutionPlan(waves)

    # ── Sync execution ────────────────────────────────────────────────

    def execute_sync(
        self,
        plan: ExecutionPlan,
        agents: Dict[str, BaseAgent],
        bus: CommunicationBus,
        *,
        skip_non_critical: bool = True,
        on_result: Optional[Callable[[AgentResult], None]] = None,
    ) -> List[AgentResult]:
        """Execute the plan synchronously using a thread pool."""
        all_results: List[AgentResult] = []
        context = bus.snapshot()

        for wave in plan.waves:
            wave_results: List[AgentResult] = []
            with ThreadPoolExecutor(max_workers=self._max_agents) as pool:
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
                        result = future.result()
                    except Exception as exc:
                        result = AgentResult(
                            sub_task_id=st.id,
                            agent_name="unknown",
                            content="",
                            success=False,
                            error=str(exc),
                        )
                    wave_results.append(result)
                    if on_result:
                        on_result(result)

            # Check for critical failures
            for r in wave_results:
                if not r.success:
                    st_obj = next((s for s in wave if s.id == r.sub_task_id), None)
                    if st_obj and st_obj.critical and not skip_non_critical:
                        raise OrchestrationError(
                            f"Critical sub-task '{r.sub_task_id}' failed: {r.error}"
                        )

            all_results.extend(wave_results)
            # refresh context after each wave
            context = bus.snapshot()

        return all_results

    # ── Async execution ───────────────────────────────────────────────

    async def execute_async(
        self,
        plan: ExecutionPlan,
        agents: Dict[str, BaseAgent],
        bus: CommunicationBus,
        *,
        skip_non_critical: bool = True,
        on_result: Optional[Callable[[AgentResult], None]] = None,
    ) -> List[AgentResult]:
        """Execute the plan asynchronously using ``asyncio.gather``."""
        all_results: List[AgentResult] = []
        context = bus.snapshot()

        for wave in plan.waves:
            tasks = []
            for st in wave:
                agent = agents.get(st.id)
                if agent is None:
                    continue
                tasks.append(self._run_agent_async(agent, st, context, bus))

            wave_results = await asyncio.gather(*tasks, return_exceptions=True)

            processed: List[AgentResult] = []
            for i, result in enumerate(wave_results):
                if isinstance(result, Exception):
                    st = wave[i]
                    result = AgentResult(
                        sub_task_id=st.id,
                        agent_name="unknown",
                        content="",
                        success=False,
                        error=str(result),
                    )
                processed.append(result)  # type: ignore[arg-type]
                if on_result:
                    on_result(result)  # type: ignore[arg-type]

            for r in processed:
                if not r.success:
                    st_obj = next((s for s in wave if s.id == r.sub_task_id), None)
                    if st_obj and st_obj.critical and not skip_non_critical:
                        raise OrchestrationError(
                            f"Critical sub-task '{r.sub_task_id}' failed: {r.error}"
                        )

            all_results.extend(processed)
            context = bus.snapshot()

        return all_results

    @staticmethod
    async def _run_agent_async(
        agent: BaseAgent,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: CommunicationBus,
    ) -> AgentResult:
        return await agent.arun(sub_task, context, bus)
