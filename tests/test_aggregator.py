"""Tests for non-destructive aggregation and validation parsing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from amacs.agents.base_agent import AgentResult, SubTask
from amacs.aggregation import Aggregator
from amacs.communication import CommunicationBus
from amacs.config import build_config
from amacs.integrations.llm_providers import LLMResponse, StubProvider


def test_aggregator_preserves_raw_merge_on_pass() -> None:
    bus = CommunicationBus()
    sub_tasks = [
        SubTask(id="s_0", label="search", description="s"),
        SubTask(id="a_1", label="analyze", description="a"),
    ]
    results = [
        AgentResult(sub_task_id="s_0", agent_name="search", content="Search Output"),
        AgentResult(sub_task_id="a_1", agent_name="analyze", content="Analyze Output"),
    ]

    mock_provider = MagicMock(spec=StubProvider)
    mock_provider.name.return_value = "stub"
    # Return JSON pass report
    mock_provider.chat.return_value = LLMResponse(
        content=json.dumps({"verdict": "pass", "issues": [], "revised_text": None}),
        model="stub",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )

    aggregator = Aggregator(provider=mock_provider)
    final_output = aggregator.aggregate(results, sub_tasks, bus)

    assert final_output == "Search Output\n\nAnalyze Output"
    assert aggregator.raw_merge == "Search Output\n\nAnalyze Output"
    assert aggregator.validation_report["verdict"] == "pass"


def test_aggregator_replaces_on_valid_revision() -> None:
    bus = CommunicationBus()
    sub_tasks = [SubTask(id="s_0", label="search", description="s")]
    results = [AgentResult(sub_task_id="s_0", agent_name="search", content="Original content string.")]

    mock_provider = MagicMock(spec=StubProvider)
    mock_provider.name.return_value = "stub"
    revised = "This is a revised content string that is sufficiently long."
    mock_provider.chat.return_value = LLMResponse(
        content=json.dumps({"verdict": "revise", "issues": ["typo"], "revised_text": revised}),
        model="stub",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )

    cfg = build_config(min_revision_ratio=0.5)
    aggregator = Aggregator(provider=mock_provider, config=cfg)
    final_output = aggregator.aggregate(results, sub_tasks, bus)

    assert final_output == revised
    assert aggregator.raw_merge == "Original content string."
    assert aggregator.validation_report["verdict"] == "revise"
    assert aggregator.validation_report["issues"] == ["typo"]


def test_aggregator_rejects_short_revision() -> None:
    bus = CommunicationBus()
    sub_tasks = [SubTask(id="s_0", label="search", description="s")]
    results = [AgentResult(sub_task_id="s_0", agent_name="search", content="Original content string that is quite long.")]

    mock_provider = MagicMock(spec=StubProvider)
    mock_provider.name.return_value = "stub"
    revised = "Too short"
    mock_provider.chat.return_value = LLMResponse(
        content=json.dumps({"verdict": "revise", "issues": ["too long"], "revised_text": revised}),
        model="stub",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )

    cfg = build_config(min_revision_ratio=0.6)
    aggregator = Aggregator(provider=mock_provider, config=cfg)
    final_output = aggregator.aggregate(results, sub_tasks, bus)

    # Rejects short revision and keeps raw merge
    assert final_output == "Original content string that is quite long."
    assert aggregator.raw_merge == "Original content string that is quite long."
    assert aggregator.validation_report["verdict"] == "revise"


def test_aggregator_fallback_on_parse_error() -> None:
    bus = CommunicationBus()
    sub_tasks = [SubTask(id="s_0", label="search", description="s")]
    results = [AgentResult(sub_task_id="s_0", agent_name="search", content="Original content.")]

    mock_provider = MagicMock(spec=StubProvider)
    mock_provider.name.return_value = "stub"
    mock_provider.chat.return_value = LLMResponse(
        content="Invalid non-JSON response from LLM",
        model="stub",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )

    aggregator = Aggregator(provider=mock_provider)
    final_output = aggregator.aggregate(results, sub_tasks, bus)

    assert final_output == "Original content."
    assert aggregator.validation_report["verdict"] == "pass"
