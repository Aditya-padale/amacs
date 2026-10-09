"""ManagerWorkerCoordinator — hierarchical delegation and synthesis protocol.

A manager agent delegates parts of a complex sub-task to specialized worker agents
and aggregates their findings into a cohesive response.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from amacs.agents.analysis_agent import AnalysisAgent
from amacs.agents.base_agent import AgentResult, SubTask
from amacs.agents.search_agent import SearchAgent
from amacs.agents.writer_agent import WriterAgent
from amacs.communication import CommunicationBus
from amacs.coordination.base import Coordinator

logger = logging.getLogger("amacs.coordination.manager_worker")


class ManagerWorkerCoordinator(Coordinator):
    """Hierarchical coordinator: Manager delegates to Search and Analysis workers."""

    def coordinate(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: CommunicationBus,
    ) -> AgentResult:
        # Worker 1: Search worker
        search_worker = SearchAgent(provider=self._provider, config=self._config)
        search_task = SubTask(
            id=f"{sub_task.id}_worker_search",
            label="search",
            description=f"Gather foundational information for: {sub_task.description}",
        )
        search_res = search_worker.run(search_task, context, bus=None)

        # Worker 2: Analysis worker
        worker_context = dict(context)
        if search_res.success and search_res.content:
            worker_context["worker_search"] = search_res.content

        analysis_worker = AnalysisAgent(provider=self._provider, config=self._config)
        analysis_task = SubTask(
            id=f"{sub_task.id}_worker_analysis",
            label="analyze",
            description=f"Analyze data for: {sub_task.description}",
        )
        analysis_res = analysis_worker.run(analysis_task, worker_context, bus=None)

        # Manager synthesis
        manager = WriterAgent(provider=self._provider, config=self._config)
        mgr_context = dict(context)
        mgr_context["search_worker_output"] = search_res.content if search_res.success else ""
        mgr_context["analysis_worker_output"] = analysis_res.content if analysis_res.success else ""

        manager_task = SubTask(
            id=sub_task.id,
            label="write",
            description=f"Synthesize worker inputs into final result for: {sub_task.description}",
        )
        return manager.run(manager_task, mgr_context, bus)
