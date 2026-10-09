"""Offline baseline and AMACS system runners."""

from __future__ import annotations

from typing import Any, Dict

from amacs.config import build_config
from amacs.integrations.llm_providers import LLMProvider, Message
from amacs.pipeline import Pipeline


def single_call(provider: LLMProvider, task: str) -> str:
    """B0: one provider call with the task as a user message."""
    return provider.chat([Message(role="user", content=task)]).content


def b0_single_call(provider: LLMProvider, task: str) -> str:
    return single_call(provider, task)


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


def b2_amacs(provider: LLMProvider, task: str) -> Dict[str, Any]:
    return amacs_pipeline(provider, task, adaptive=False)


def b1_sequential(provider: LLMProvider, task: str) -> Dict[str, Any]:
    """B1: decomposed sequential pipeline (no adaptive control)."""
    return amacs_pipeline(provider, task, adaptive=False)


def b3_debate(provider: LLMProvider, task: str) -> Dict[str, Any]:
    """B3: debate coordinator baseline."""
    config = build_config(mode="debate", strategy="speed", max_agents=2)
    result = Pipeline(config, provider=provider).run(lambda value: value, (task,), {})
    return {"output": result.final_output, "result": result}


def b4_manager_worker(provider: LLMProvider, task: str) -> Dict[str, Any]:
    """B4: hierarchical manager/worker coordinator baseline."""
    config = build_config(mode="manager_worker", strategy="speed", max_agents=2)
    result = Pipeline(config, provider=provider).run(lambda value: value, (task,), {})
    return {"output": result.final_output, "result": result}


def b5_adaptive(provider: LLMProvider, task: str) -> Dict[str, Any]:
    """B5: full adaptive AMACS system."""
    return amacs_pipeline(provider, task, adaptive=True)
