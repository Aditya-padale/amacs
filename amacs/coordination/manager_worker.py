"""ManagerWorkerCoordinator — hierarchical delegation and synthesis protocol.

A manager agent delegates parts of a complex sub-task to specialized worker agents
and aggregates their findings into a cohesive response.
"""

from __future__ import annotations

import asyncio
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
        search_worker = self.prepare_agent(SearchAgent(provider=self._provider, config=self._config))
        search_task = SubTask(
            id=f"{sub_task.id}_worker_search",
            label="search",
            description=f"Gather foundational information for: {sub_task.description}",
        )
        search_res = search_worker.run(search_task, context, bus=bus)

        # Worker 2: Analysis worker
        worker_context = dict(context)
        if search_res.success and search_res.content:
            worker_context["worker_search"] = search_res.content

        analysis_worker = self.prepare_agent(AnalysisAgent(provider=self._provider, config=self._config))
        analysis_task = SubTask(
            id=f"{sub_task.id}_worker_analysis",
            label="analyze",
            description=f"Analyze data for: {sub_task.description}",
        )
        analysis_res = analysis_worker.run(analysis_task, worker_context, bus=bus)

        # Manager synthesis
        manager = self.prepare_agent(WriterAgent(provider=self._provider, config=self._config))
        mgr_context = dict(context)
        mgr_context["search_worker_output"] = search_res.content if search_res.success else ""
        mgr_context["analysis_worker_output"] = analysis_res.content if analysis_res.success else ""

        manager_task = SubTask(
            id=sub_task.id,
            label="write",
            description=f"Synthesize worker inputs into final result for: {sub_task.description}",
        )
        mgr_res = manager.run(manager_task, mgr_context, bus)

        # Combine token usage
        p_tokens = (
            search_res.token_usage.get("prompt_tokens", 0)
            + analysis_res.token_usage.get("prompt_tokens", 0)
            + mgr_res.token_usage.get("prompt_tokens", 0)
        )
        c_tokens = (
            search_res.token_usage.get("completion_tokens", 0)
            + analysis_res.token_usage.get("completion_tokens", 0)
            + mgr_res.token_usage.get("completion_tokens", 0)
        )
        mgr_res.token_usage = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
        }
        return mgr_res

    async def acoordinate(self, sub_task: SubTask, context: Dict[str, Any], bus: CommunicationBus) -> AgentResult:
        search = self.prepare_agent(SearchAgent(provider=self._provider, config=self._config))
        analysis = self.prepare_agent(AnalysisAgent(provider=self._provider, config=self._config))
        search_task = SubTask(id=f"{sub_task.id}_worker_search", label="search", description=f"Gather information for: {sub_task.description}")
        analysis_task = SubTask(id=f"{sub_task.id}_worker_analysis", label="analyze", description=f"Analyze: {sub_task.description}")
        search_res, analysis_res = await asyncio.gather(search.arun(search_task, context, bus), analysis.arun(analysis_task, context, bus))
        manager = self.prepare_agent(WriterAgent(provider=self._provider, config=self._config))
        mgr_context = {**context, "search_worker_output": search_res.content, "analysis_worker_output": analysis_res.content}
        result = await manager.arun(SubTask(id=sub_task.id, label="write", description=f"Synthesize findings for: {sub_task.description}"), mgr_context, bus)
        prompt = sum(r.token_usage.get("prompt_tokens", 0) for r in (search_res, analysis_res, result))
        completion = sum(r.token_usage.get("completion_tokens", 0) for r in (search_res, analysis_res, result))
        result.token_usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
        return result
