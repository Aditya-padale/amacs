"""Behavioral tests for StrategyPolicy rules, per-agent routing, and DEFAULT_MODELS."""

from __future__ import annotations

import os

import pytest

from amacs.config import build_config
from amacs.strategy import DEFAULT_MODELS, StrategyPolicy


def test_strategy_performance_rules_and_behavior() -> None:
    rules = StrategyPolicy.get_rules("performance")
    assert rules.strategy_name == "performance"
    assert rules.default_model == "gpt-4o"
    assert rules.retry_limit == 3
    assert rules.enable_validation_pass is True
    assert rules.max_concurrency_multiplier == 1.0
    assert rules.enable_candidate_selection is True
    assert rules.enable_critique_revision is True


def test_strategy_cost_rules_and_behavior() -> None:
    rules = StrategyPolicy.get_rules("cost")
    assert rules.strategy_name == "cost"
    assert rules.default_model == "gpt-4o-mini"
    assert rules.retry_limit == 1
    assert rules.enable_validation_pass is False
    assert rules.max_tokens == 512
    assert rules.enable_critique_revision is False

    # Assert model resolution for cost strategy
    model_openai = StrategyPolicy.resolve_model("cost", "openai")
    assert model_openai == "gpt-4o-mini"

    model_anthropic = StrategyPolicy.resolve_model("cost", "anthropic")
    assert model_anthropic == "claude-3-5-haiku-20241022"


def test_strategy_speed_rules_and_behavior() -> None:
    rules = StrategyPolicy.get_rules("speed")
    assert rules.strategy_name == "speed"
    assert rules.default_model == "gpt-4o-mini"
    assert rules.retry_limit == 1
    assert rules.enable_validation_pass is False
    assert rules.max_concurrency_multiplier == 2.0
    assert rules.max_tokens == 256
    assert rules.timeout_seconds == 15.0
    assert rules.enable_critique_revision is False


def test_per_agent_model_routing() -> None:
    config = build_config(
        strategy="performance",
        models={"search": "gpt-4o-mini", "write": "gpt-4o"},
    )

    resolved_search = StrategyPolicy.resolve_model("performance", "openai", agent_type="search", config=config)
    assert resolved_search == "gpt-4o-mini"

    resolved_write = StrategyPolicy.resolve_model("performance", "openai", agent_type="write", config=config)
    assert resolved_write == "gpt-4o"

    resolved_other = StrategyPolicy.resolve_model("performance", "openai", agent_type="analysis", config=config)
    assert resolved_other == "gpt-4o"


def test_default_models_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMACS_DEFAULT_OPENAI_MODEL", "gpt-4o-2026-custom")

    resolved = DEFAULT_MODELS.get("openai")
    assert resolved == "gpt-4o"
    assert os.getenv("AMACS_DEFAULT_OPENAI_MODEL") == "gpt-4o-2026-custom"
