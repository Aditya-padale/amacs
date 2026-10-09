"""End-to-end checks for completed execution guarantees."""

from __future__ import annotations

from amacs.agents.base_agent import AgentResult, SubTask
from amacs.agents.search_agent import SearchAgent
from amacs.config import build_config
from amacs.context_builder import ContextBuilder
from amacs.exceptions import LLMProviderError
from amacs.integrations.llm_providers import FakeProvider, LLMResponse
from amacs.pipeline import Pipeline
from amacs.tracing import Tracer


def test_pipeline_owned_cache_reuses_agent_response() -> None:
    provider = FakeProvider(default_response="cached")
    config = build_config(cache=True, strategy="cost")
    agent = SearchAgent(provider=provider, config=config)
    # Simulate one pipeline-owned cache being attached to every selected agent.
    from amacs.cache import ResponseCache
    agent.attach_cache(ResponseCache())
    task = SubTask(id="search_0", label="search", description="same request")
    first = agent.run(task, {})
    second = agent.run(task, {})
    assert first.success and second.success
    assert len(provider.calls) == 1
    assert second.metadata["cached"] is True


def test_model_directed_tool_loop_executes_only_requested_tool() -> None:
    provider = FakeProvider(responses=[
        LLMResponse("", "stub", {}, tool_calls=[{"name": "python_calc", "arguments": '{"expression":"2+2"}'}]),
        LLMResponse("final", "stub", {}),
    ])
    agent = SearchAgent(provider=provider, config=build_config(max_tool_steps=1))
    result = agent.run(SubTask(id="search_0", label="search", description="calculate"), {})
    assert result.content == "final"
    assert len(provider.calls) == 2


def test_provider_success_retryable_and_fatal_failure_behavior() -> None:
    task = SubTask(id="search_0", label="search", description="request")
    retryable = FakeProvider(responses=[LLMProviderError("503 unavailable"), "recovered"])
    result = SearchAgent(provider=retryable, config=build_config(retry_limit=2)).run(task, {})
    assert result.success and len(retryable.calls) == 2

    fatal = FakeProvider(responses=[LLMProviderError("401 invalid api key")])
    failed = SearchAgent(provider=fatal, config=build_config(retry_limit=3)).run(task, {})
    assert not failed.success and len(fatal.calls) == 1


async def test_async_debate_and_manager_worker_flow_through_pipeline() -> None:
    async def work(topic: str) -> str:
        return topic

    for mode in ("debate", "manager_worker"):
        result = await Pipeline(build_config(mode=mode, strategy="speed"), provider=FakeProvider()).arun(work, ("topic",), {})
        assert result.agent_results
        assert result.cost_report["total_tokens"] > 0
        assert "success" in result.to_mermaid()


def test_context_summary_and_execution_dag_rendering() -> None:
    builder = ContextBuilder(max_tokens=5, token_counter=lambda value: len(value.split()))
    rendered = builder.summarize_then_build({"original_input": "one two three four five six"}, lambda _: "short summary")
    assert "summarized" in rendered
    assert builder.truncation_record and builder.truncation_record["summarized"]

    tracer = Tracer()
    from amacs.orchestrator.scheduler import ExecutionPlan
    plan = ExecutionPlan(waves=[[SubTask("a", "search", "a"), SubTask("b", "write", "b", dependencies=["a"])]])
    graph = tracer.render_execution_dag(plan, [AgentResult("a", "search", "", success=True), AgentResult("b", "write", "", success=False)])
    assert "a --> b" in graph
    assert "a: success" in graph
    assert "b: failed" in graph

