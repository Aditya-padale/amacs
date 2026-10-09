"""Test suite for Phase 4 genuine multi-agent features (Tools, LLMDecomposer, Coordination)."""

from __future__ import annotations

from amacs.agents.base_agent import SubTask
from amacs.communication import CommunicationBus
from amacs.coordination import DebateCoordinator, ManagerWorkerCoordinator
from amacs.orchestrator.llm_decomposer import LLMTaskDecomposer
from amacs.orchestrator.task_analyzer import TaskProfile
from amacs.tools import PythonCalcTool, ToolRegistry, VectorSearchTool, WebSearchTool


def test_tools_execution() -> None:
    calc = PythonCalcTool()
    assert calc.execute(expression="2 + 3 * 4") == "14"
    assert "Error" in calc.execute(expression="import os")

    web = WebSearchTool()
    assert "Search Results" in web.execute(query="quantum computing")

    vec = VectorSearchTool()
    assert "AMACS" in vec.execute(query="AMACS adaptive")


def test_tool_registry() -> None:
    registry = ToolRegistry()
    assert registry.get("python_calc") is not None
    assert registry.get("web_search") is not None
    assert registry.get("vector_search") is not None
    assert len(registry.list_tools()) >= 3


def test_llm_task_decomposer_fallback() -> None:
    decomposer = LLMTaskDecomposer()
    profile = TaskProfile(domain="research", estimated_sub_tasks=4)
    tasks = decomposer.decompose(profile)
    assert len(tasks) >= 1
    assert isinstance(tasks[0], SubTask)


def test_debate_coordinator() -> None:
    coordinator = DebateCoordinator()
    task = SubTask(id="debate_1", label="write", description="Should AI code decorators be used?")
    bus = CommunicationBus()

    res = coordinator.coordinate(task, {}, bus)
    assert res.success
    assert len(res.content) > 0


def test_manager_worker_coordinator() -> None:
    coordinator = ManagerWorkerCoordinator()
    task = SubTask(id="mgr_1", label="write", description="Synthesize multi-agent research")
    bus = CommunicationBus()

    res = coordinator.coordinate(task, {}, bus)
    assert res.success
    assert len(res.content) > 0
