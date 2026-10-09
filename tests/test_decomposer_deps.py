"""Regression and Hypothesis property tests for task decomposer dependencies and scheduling."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from amacs.exceptions import OrchestrationError
from amacs.orchestrator.scheduler import Scheduler
from amacs.orchestrator.task_analyzer import TaskProfile
from amacs.orchestrator.task_decomposer import TaskDecomposer


def test_extra_search_tasks_and_analyze_deps() -> None:
    """Regression test: secondary domain search tasks are included in analyze deps."""
    decomposer = TaskDecomposer()
    profile = TaskProfile(
        domain="research",
        complexity=0.8,
        estimated_sub_tasks=5,
        secondary_domains=["coding"],
    )
    sub_tasks = decomposer.decompose(profile)
    
    search_tasks = [st for st in sub_tasks if st.label == "search"]
    analyze_tasks = [st for st in sub_tasks if st.label == "analyze"]
    
    assert len(search_tasks) == 2
    for s_task in search_tasks:
        assert s_task.dependencies == []
        
    for a_task in analyze_tasks:
        for s_task in search_tasks:
            assert s_task.id in a_task.dependencies, f"{s_task.id} missing from {a_task.id} deps"


def test_unknown_dep_raises_in_scheduler() -> None:
    from amacs.agents.base_agent import SubTask

    scheduler = Scheduler()
    tasks = [
        SubTask(id="a_0", label="search", description="search"),
        SubTask(id="a_1", label="analyze", description="analyze", dependencies=["unknown_999"]),
    ]
    with pytest.raises(OrchestrationError, match="references unknown dependency"):
        scheduler.plan(tasks)


@given(
    domain=st.sampled_from(["research", "analysis", "content_creation", "coding", "planning", "general"]),
    complexity=st.floats(min_value=0.0, max_value=1.0),
    estimated_sub_tasks=st.integers(min_value=1, max_value=8),
    secondary_domains=st.lists(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))), max_size=3),
)
def test_decomposer_hypothesis_properties(
    domain: str,
    complexity: float,
    estimated_sub_tasks: int,
    secondary_domains: list[str],
) -> None:
    decomposer = TaskDecomposer()
    scheduler = Scheduler()
    profile = TaskProfile(
        domain=domain,
        complexity=complexity,
        estimated_sub_tasks=estimated_sub_tasks,
        secondary_domains=secondary_domains,
    )
    
    try:
        sub_tasks = decomposer.decompose(profile)
    except OrchestrationError:
        # Invalid trimming/configuration raising OrchestrationError is acceptable
        return

    task_map = {st.id: st for st in sub_tasks}
    search_ids = [st.id for st in sub_tasks if st.label == "search"]

    for st_obj in sub_tasks:
        # 1. Every dep resolves
        for dep in st_obj.dependencies:
            assert dep in task_map, f"Unresolved dep '{dep}' in task '{st_obj.id}'"
        
        # 2. Analyze tasks depend on all search tasks
        if st_obj.label == "analyze":
            for s_id in search_ids:
                if s_id != st_obj.id:
                    assert s_id in st_obj.dependencies

    # 3. DAG is acyclic and every task is reachable in plan
    plan = scheduler.plan(sub_tasks)
    planned_ids = {st_obj.id for wave in plan.waves for st_obj in wave}
    assert planned_ids == set(task_map.keys())
