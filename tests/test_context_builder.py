"""Tests for token-budgeted and dependency-filtered ContextBuilder."""

from __future__ import annotations

from amacs.agents.base_agent import SubTask
from amacs.agents.writer_agent import WriterAgent
from amacs.config import build_config
from amacs.context_builder import ContextBuilder
from amacs.integrations.llm_providers import StubProvider


def test_context_builder_filters_undeclared_dependencies() -> None:
    context = {
        "original_input": "User query",
        "search_0": "Search result",
        "search_1_unrelated": "Unrelated search result",
        "analyze_2": "Analyze result",
    }
    sub_task = SubTask(
        id="write_3",
        label="write",
        description="Write draft",
        dependencies=["analyze_2"],
    )

    agent = WriterAgent(provider=StubProvider(), config=build_config())
    prompt = agent.build_user_prompt(sub_task, context)

    assert "original_input" in prompt
    assert "analyze_2" in prompt
    assert "search_0" not in prompt
    assert "search_1_unrelated" not in prompt


def test_context_builder_truncation_when_over_budget() -> None:
    builder = ContextBuilder(max_tokens=50)
    long_text = "word " * 300
    context = {
        "original_input": long_text,
    }

    ctx_str = builder.build_context_string(context, allowed_keys=["original_input"])

    assert builder.was_truncated is True
    assert "[TRUNCATED:" in ctx_str
    assert builder.truncation_record is not None
    assert builder.truncation_record["max_tokens"] == 50


def test_custom_token_counter() -> None:
    # Custom counter: 1 token per word
    builder = ContextBuilder(max_tokens=5, token_counter=lambda s: len(s.split()))
    context = {"k1": "one two three four five six seven eight"}

    ctx_str = builder.build_context_string(context, allowed_keys=["k1"])
    assert builder.was_truncated is True
    assert "[TRUNCATED:" in ctx_str
