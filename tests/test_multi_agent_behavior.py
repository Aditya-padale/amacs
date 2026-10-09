"""Comprehensive behavioral tests for AMACS multi-agent features:
- Coordination modes (pipeline, debate, manager_worker)
- Planner (template vs llm with Pydantic & DAG validation & fallback)
- Hardened PythonCalcTool AST evaluator & escape attempt tests
- Pluggable WebSearchTool backends & VectorSearchTool integration
- Bounded tool loop in SearchAgent
- Candidate selection (k candidates + LLM judge)
- Critic/reviser loop
- Decorator contract (task, return_value_as_prompt, context)
- Typed output validation and retry
- ResponseCache in-memory and disk persistence
"""

from __future__ import annotations

import os
import tempfile

from pydantic import BaseModel

from amacs.agents.base_agent import SubTask
from amacs.agents.search_agent import SearchAgent
from amacs.cache import ResponseCache
from amacs.communication import CommunicationBus
from amacs.config import build_config
from amacs.coordination.debate import DebateCoordinator
from amacs.coordination.manager_worker import ManagerWorkerCoordinator
from amacs.decorator import amacs
from amacs.integrations.llm_providers import FakeProvider, LLMResponse, Message
from amacs.integrations.vector_db import SimpleVectorStore
from amacs.orchestrator.llm_decomposer import LLMTaskDecomposer
from amacs.orchestrator.task_analyzer import TaskProfile
from amacs.pipeline import Pipeline
from amacs.tools.builtins import (
    MockSearchBackend,
    PythonCalcTool,
    VectorSearchTool,
    WebSearchTool,
)

# ── 1. Coordination Modes Tests ──────────────────────────────────────────────

def test_coordination_mode_debate() -> None:
    provider = FakeProvider(default_response="Debated consensus result")
    config = build_config(mode="debate")
    coordinator = DebateCoordinator(provider=provider, config=config)
    bus = CommunicationBus()
    sub_task = SubTask(id="task_1", label="write", description="Draft section")

    res = coordinator.coordinate(sub_task, {}, bus)
    assert res.success
    assert res.agent_name == "debate_consensus"
    assert "Debated consensus" in res.content
    assert res.token_usage.get("total_tokens", 0) > 0
    assert "task_1" in bus.snapshot()


def test_coordination_mode_manager_worker() -> None:
    provider = FakeProvider(default_response="Worker synthesis output")
    config = build_config(mode="manager_worker")
    coordinator = ManagerWorkerCoordinator(provider=provider, config=config)
    bus = CommunicationBus()
    sub_task = SubTask(id="task_1", label="write", description="Research topic")

    res = coordinator.coordinate(sub_task, {}, bus)
    assert res.success
    assert res.token_usage.get("total_tokens", 0) > 0
    assert "task_1" in bus.snapshot()


@amacs(mode="debate", llm_provider="stub", return_details=True)
def debate_pipeline_func(topic: str) -> str:
    return f"Topic: {topic}"


def test_decorator_mode_debate() -> None:
    res = debate_pipeline_func("AI Ethics")
    assert res.final_output != ""
    assert len(res.agent_results) > 0


# ── 2. Planner & DAG Validation Tests ─────────────────────────────────────────

def test_llm_decomposer_valid_json() -> None:
    json_plan = """[
        {"id": "search_0", "label": "search", "description": "Search info", "dependencies": [], "critical": true},
        {"id": "write_1", "label": "write", "description": "Write report", "dependencies": ["search_0"], "critical": true}
    ]"""
    provider = FakeProvider(default_response=json_plan)
    decomposer = LLMTaskDecomposer(provider=provider)
    profile = TaskProfile(domain="research", description="Test", function_name="test", estimated_sub_tasks=2)

    tasks = decomposer.decompose(profile)
    assert len(tasks) == 2
    assert tasks[0].id == "search_0"
    assert tasks[1].dependencies == ["search_0"]


def test_llm_decomposer_fallback_on_invalid_dag() -> None:
    # Invalid DAG with cycle
    cycle_json = """[
        {"id": "task_a", "label": "search", "description": "A", "dependencies": ["task_b"], "critical": true},
        {"id": "task_b", "label": "write", "description": "B", "dependencies": ["task_a"], "critical": true}
    ]"""
    provider = FakeProvider(default_response=cycle_json)
    decomposer = LLMTaskDecomposer(provider=provider)
    profile = TaskProfile(domain="research", description="Test", function_name="test", estimated_sub_tasks=2)

    # Should safely fallback to rule-based decomposer without crashing
    tasks = decomposer.decompose(profile)
    assert len(tasks) > 0
    assert tasks[0].id.startswith("search_") or tasks[0].id.startswith("write_")


# ── 3. Hardened Tools & AST Evaluator Tests ──────────────────────────────────

def test_python_calc_ast_valid() -> None:
    calc = PythonCalcTool()
    assert calc.execute(expression="2 + 2") == "4"
    assert calc.execute(expression="10 * 3.5 - 5") == "30.0"
    assert calc.execute(expression="sqrt(16)") == "4.0"
    assert calc.execute(expression="min(10, 20) + max(1, 5)") == "15"


def test_python_calc_ast_escape_attempts() -> None:
    calc = PythonCalcTool()
    # Code execution escape attempts must be rejected safely
    res1 = calc.execute(expression="__import__('os').system('ls')")
    assert "Error" in res1 or "not allowed" in res1 or "forbidden" in res1

    res2 = calc.execute(expression="open('/etc/passwd').read()")
    assert "Error" in res2 or "not allowed" in res2 or "forbidden" in res2

    res3 = calc.execute(expression="(lambda: 1)()")
    assert "Error" in res3 or "forbidden" in res3

    res4 = calc.execute(expression="eval('2+2')")
    assert "Error" in res4 or "not allowed" in res4


def test_web_search_mock_backend() -> None:
    tool = WebSearchTool(backend=MockSearchBackend())
    out = tool.execute(query="Quantum Computing")
    assert "[MOCK DATA]" in out
    assert "Quantum Computing" in out


def test_vector_search_tool_with_simple_store() -> None:
    store = SimpleVectorStore(documents=["Python is popular.", "AMACS supports multi-agent AI."])
    tool = VectorSearchTool(vector_store=store)
    out = tool.execute(query="multi-agent")
    assert "AMACS supports multi-agent AI." in out


def test_search_agent_bounded_tool_loop() -> None:
    agent = SearchAgent(provider=FakeProvider(), max_tool_steps=2)
    bus = CommunicationBus()
    sub_task = SubTask(id="search_0", label="search", description="Search test")
    prompt = agent.build_user_prompt(sub_task, {}, bus=bus)
    assert "Tool Context:" in prompt
    assert "search_0_tool_step_1" in bus.snapshot()


# ── 4. Candidate Selection Tests ──────────────────────────────────────────────

def test_candidate_selection_k_candidates() -> None:
    provider = FakeProvider(responses=[
        LLMResponse(content="Short draft", model="stub", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        LLMResponse(content="Comprehensive draft with detailed insights", model="stub", usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}),
        LLMResponse(content="1", model="stub", usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}),  # LLM Judge picks candidate 1 (index 0) or 2
    ])
    config = build_config(candidates_k=2, strategy="performance")
    agent = SearchAgent(provider=provider, config=config)
    sub_task = SubTask(id="task_critical", label="search", description="Critical search", critical=True)
    res = agent.run(sub_task, {})

    assert res.success
    assert "candidates" in res.metadata
    assert len(res.metadata["candidates"]) == 2


# ── 5. Critic / Reviser Loop Tests ─────────────────────────────────────────────

def test_critic_reviser_loop() -> None:
    provider = FakeProvider(responses=[
        LLMResponse(content="Initial rough draft", model="stub", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        LLMResponse(content="Score: 0.40 - Needs structure and detail", model="stub", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),  # Critic
        LLMResponse(content="Revised polished draft with structure and detail", model="stub", usage={"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}),  # Revision
        LLMResponse(content="Score: 0.90 - Excellent draft", model="stub", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),  # Critic pass
    ])
    config = build_config(max_revisions=2, critic_score_threshold=0.8)
    from amacs.agents.writer_agent import WriterAgent
    agent = WriterAgent(provider=provider, config=config)
    sub_task = SubTask(id="write_0", label="write", description="Draft essay")
    res = agent.run(sub_task, {})

    assert res.success
    assert "revisions" in res.metadata
    assert len(res.metadata["revisions"]) >= 1


# ── 6. Decorator Contract Tests ───────────────────────────────────────────────

@amacs(task="Explicit custom task prompt", return_details=True)
def explicit_task_func(x: int) -> str:
    return f"Value: {x}"


def test_decorator_explicit_task() -> None:
    res = explicit_task_func(42)
    assert res.communication_log[0].value == "Explicit custom task prompt"


@amacs(input_mode="return_value_as_prompt", return_details=True)
def return_prompt_func() -> str:
    return "Dynamic prompt generated by return"


def test_decorator_return_value_as_prompt() -> None:
    res = return_prompt_func()
    assert res.communication_log[0].value == "Dynamic prompt generated by return"


# ── 7. Typed Output Schema Tests ─────────────────────────────────────────────

class ReportSchema(BaseModel):
    summary: str
    score: int


def test_typed_output_schema_parsing() -> None:
    valid_json = '{"summary": "Clean summary", "score": 95}'
    provider = FakeProvider(default_response=valid_json)
    config = build_config(output_schema=ReportSchema)
    pipeline = Pipeline(config=config, provider=provider)

    def dummy() -> str:
        return "Task"

    res = pipeline.run(dummy, (), {})
    assert "summary" in res.final_output
    assert "95" in res.final_output


# ── 8. Response Cache Tests ──────────────────────────────────────────────────

def test_response_cache_in_memory() -> None:
    cache = ResponseCache(enabled=True)
    msgs = [Message(role="user", content="Hello")]
    key = ResponseCache.compute_key(msgs, provider="stub", model="test")

    resp = LLMResponse(content="Hi there", model="test", usage={"total_tokens": 5})
    cache.set(key, resp)

    cached = cache.get(key)
    assert cached is not None
    assert cached.content == "Hi there"


def test_response_cache_disk_persistence() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cache.json")
        cache1 = ResponseCache(enabled=True, filepath=path)
        msgs = [Message(role="user", content="Persistent query")]
        key = ResponseCache.compute_key(msgs, provider="stub", model="test")

        cache1.set(key, LLMResponse(content="Saved answer", model="test", usage={"total_tokens": 10}))

        # Load fresh cache from same path
        cache2 = ResponseCache(enabled=True, filepath=path)
        cached = cache2.get(key)
        assert cached is not None
        assert cached.content == "Saved answer"
