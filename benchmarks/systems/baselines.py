"""Offline baseline and AMACS system runners."""

from __future__ import annotations

from typing import Any, Dict

from amacs.config import build_config
from amacs.integrations.llm_providers import LLMProvider, Message
from amacs.pipeline import Pipeline


def single_call(provider: LLMProvider, task: str) -> str:
    """B0: one provider call with the task as a user message."""
    return provider.chat([Message(role="user", content=task)]).content


def amacs_pipeline(provider: LLMProvider, task: str, adaptive: bool) -> Dict[str, Any]:
    """B2/B5-compatible runner for the shipped offline pipeline."""

    def fixture_task(value: str) -> str:
        return value

    config = build_config(
        adaptive=adaptive,
        retry_limit=2,
        skip_non_critical=True,
        strategy="speed",
        max_agents=2,
    )
    result = Pipeline(config, provider=provider).run(fixture_task, (task,), {})
    return {"output": result.final_output, "result": result}