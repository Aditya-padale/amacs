"""Tests for the orchestrator — task analysis, decomposition, agent selection, scheduling."""

from __future__ import annotations

import pytest

from amacs.agents.analysis_agent import AnalysisAgent
from amacs.agents.base_agent import SubTask
from amacs.agents.search_agent import SearchAgent
from amacs.agents.validator_agent import ValidatorAgent
from amacs.agents.writer_agent import WriterAgent
from amacs.exceptions import OrchestrationError
from amacs.orchestrator.agent_selector import _AGENT_REGISTRY, AgentSelector, register_agent
from amacs.orchestrator.scheduler import Scheduler
from amacs.orchestrator.task_analyzer import TaskAnalyzer, TaskProfile
from amacs.orchestrator.task_decomposer import TaskDecomposer


class TestTaskAnalyzer:
    def setup_method(self) -> None:
        self.analyzer = TaskAnalyzer()

    def test_research_domain(self) -> None:
        def research_task(topic: str) -> str:
            """Research and investigate the given topic."""
            return topic

        profile = self.analyzer.analyze(research_task, ("AI",), {})
        assert profile.domain == "research"
        assert profile.function_name == "research_task"

    def test_analysis_domain(self) -> None:
        def analyze_data(data: str) -> str:
            """Analyze and evaluate the dataset."""
            return data

        profile = self.analyzer.analyze(analyze_data, ("dataset",), {})
        assert profile.domain == "analysis"

    def test_general_domain_fallback(self) -> None:
        def foo(x: int) -> int:
            return x

        profile = self.analyzer.analyze(foo, (42,), {})
        assert profile.domain == "general"

    def test_complexity_scaling(self) -> None:
        def simple_task(x: str) -> str:
            return x

        def complex_task(x: str) -> str:
            """Comprehensive, detailed, in-depth multi-step analysis."""
            return x

        p1 = self.analyzer.analyze(simple_task, ("a",), {})
        p2 = self.analyzer.analyze(complex_task, ("a",), {})
        assert p2.complexity > p1.complexity

    def test_arg_summary(self) -> None:
        def task(topic: str, depth: int) -> str:
            return topic

        profile = self.analyzer.analyze(task, ("AI", 5), {})
        assert "topic='AI'" in profile.arg_summary
        assert "depth=5" in profile.arg_summary

    def test_substring_matching_prevention(self) -> None:
        def contest_results(database: str, smallest_item: str) -> str:
            """Process contest results from database for smallest entry."""
            return database

        profile = self.analyzer.analyze(contest_results, ("db", "item"), {})
        # "contest" should not trigger coding ("test"), "database" should not trigger analysis ("data"), "smallest" should not trigger complexity ("all")
        assert profile.domain == "general"
        assert profile.complexity < 0.5


class TestTaskDecomposer:
    def setup_method(self) -> None:
        self.decomposer = TaskDecomposer()

    def test_research_decomposition(self) -> None:
        profile = TaskProfile(domain="research", complexity=0.5, estimated_sub_tasks=4)
        sub_tasks = self.decomposer.decompose(profile)
        assert len(sub_tasks) == 4
        labels = [st.label for st in sub_tasks]
        assert "search" in labels
        assert "validate" in labels

    def test_sub_tasks_have_ids(self) -> None:
        profile = TaskProfile(domain="general", complexity=0.5, estimated_sub_tasks=4)
        sub_tasks = self.decomposer.decompose(profile)
        ids = [st.id for st in sub_tasks]
        assert len(ids) == len(set(ids))  # unique

    def test_dependencies_reference_valid_ids(self) -> None:
        profile = TaskProfile(domain="research", complexity=0.5, estimated_sub_tasks=4)
        sub_tasks = self.decomposer.decompose(profile)
        valid_ids = {st.id for st in sub_tasks}
        for st in sub_tasks:
            for dep in st.dependencies:
                assert dep in valid_ids, f"Dependency '{dep}' not in {valid_ids}"

    def test_trimming_to_fewer_tasks(self) -> None:
        profile = TaskProfile(domain="research", complexity=0.1, estimated_sub_tasks=2)
        sub_tasks = self.decomposer.decompose(profile)
        assert len(sub_tasks) == 2


class TestAgentSelector:
    def setup_method(self) -> None:
        self.selector = AgentSelector()

    def test_label_to_agent_mapping(self) -> None:
        sub_tasks = [
            SubTask(id="s0", label="search", description="search task"),
            SubTask(id="a1", label="analyze", description="analysis task"),
            SubTask(id="w2", label="write", description="write task"),
            SubTask(id="v3", label="validate", description="validate task"),
        ]
        agents = self.selector.select(sub_tasks)
        assert isinstance(agents["s0"], SearchAgent)
        assert isinstance(agents["a1"], AnalysisAgent)
        assert isinstance(agents["w2"], WriterAgent)
        assert isinstance(agents["v3"], ValidatorAgent)

    def test_unknown_label_fallback(self) -> None:
        sub_tasks = [SubTask(id="x0", label="unknown_type", description="something")]
        agents = self.selector.select(sub_tasks)
        assert "x0" in agents  # should get a fallback

    def test_register_custom_agent(self) -> None:
        class MyAgent(SearchAgent):
            @property
            def agent_type(self) -> str:
                return "custom"

        register_agent("custom", MyAgent)
        sub_tasks = [SubTask(id="c0", label="custom", description="custom task")]
        agents = self.selector.select(sub_tasks)
        assert isinstance(agents["c0"], MyAgent)
        # cleanup
        del _AGENT_REGISTRY["custom"]


class TestScheduler:
    def setup_method(self) -> None:
        self.scheduler = Scheduler()

    def test_linear_plan(self) -> None:
        tasks = [
            SubTask(id="a", label="search", description="a"),
            SubTask(id="b", label="analyze", description="b", dependencies=["a"]),
            SubTask(id="c", label="write", description="c", dependencies=["b"]),
        ]
        plan = self.scheduler.plan(tasks)
        assert len(plan.waves) == 3
        assert plan.waves[0][0].id == "a"
        assert plan.waves[1][0].id == "b"
        assert plan.waves[2][0].id == "c"

    def test_parallel_plan(self) -> None:
        tasks = [
            SubTask(id="a", label="search", description="a"),
            SubTask(id="b", label="search", description="b"),
            SubTask(id="c", label="write", description="c", dependencies=["a", "b"]),
        ]
        plan = self.scheduler.plan(tasks)
        assert len(plan.waves) == 2
        assert len(plan.waves[0]) == 2  # a and b in parallel

    def test_max_agents_chunking(self) -> None:
        from amacs.config import build_config

        cfg = build_config(max_agents=1)
        scheduler = Scheduler(config=cfg)
        tasks = [
            SubTask(id="a", label="search", description="a"),
            SubTask(id="b", label="search", description="b"),
        ]
        plan = scheduler.plan(tasks)
        # With max_agents=1, each parallel task becomes its own wave
        assert len(plan.waves) == 2

    def test_circular_dependency_raises(self) -> None:
        tasks = [
            SubTask(id="a", label="s", description="a", dependencies=["b"]),
            SubTask(id="b", label="s", description="b", dependencies=["a"]),
        ]
        with pytest.raises(OrchestrationError, match="Circular dependency"):
            self.scheduler.plan(tasks)
