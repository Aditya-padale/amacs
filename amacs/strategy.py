"""Strategy policy module — translates configuration strategy into execution rules.

Supported strategies:
- ``performance``: Highest quality, uses main models, validation pass enabled, full retries, critique enabled.
- ``cost``: Minimises API expenditure, uses lightweight models, skips optional validation, caps max_tokens.
- ``speed``: Minimises overall latency, enables high concurrency, tighter timeout, short max_tokens, skips validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Default model IDs looked up from official provider documentation as of 2026-10-09.
# OpenAI: https://platform.openai.com/docs/models
# Groq: https://console.groq.com/docs/models
# Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
# Gemini: https://ai.google.dev/gemini-api/docs/models/gemini
DEFAULT_MODELS: Dict[str, str] = {
    "openai": os.getenv("AMACS_DEFAULT_OPENAI_MODEL", "gpt-4o"),
    "groq": os.getenv("AMACS_DEFAULT_GROQ_MODEL", "llama-3.3-70b-versatile"),
    "anthropic": os.getenv("AMACS_DEFAULT_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "gemini": os.getenv("AMACS_DEFAULT_GEMINI_MODEL", "gemini-2.5-flash"),
    "ollama": os.getenv("AMACS_DEFAULT_OLLAMA_MODEL", "llama3"),
    "stub": "stub",
}


@dataclass(frozen=True)
class StrategyRules:
    """Rules derived from the chosen execution strategy."""

    strategy_name: str
    default_model: str
    retry_limit: int
    enable_validation_pass: bool
    max_concurrency_multiplier: float
    model_override_map: Dict[str, str]
    max_tokens: Optional[int] = None
    timeout_seconds: Optional[float] = None
    enable_candidate_selection: bool = True
    enable_critique_revision: bool = True


_MODEL_MAPS: Dict[str, Dict[str, str]] = {
    "performance": {
        "openai": "gpt-4o",
        "groq": "llama-3.3-70b-versatile",
        "anthropic": "claude-sonnet-4-20250514",
        "gemini": "gemini-2.5-flash",
        "ollama": "llama3",
        "stub": "stub",
    },
    "cost": {
        "openai": "gpt-4o-mini",
        "groq": "llama-3.1-8b-instant",
        "anthropic": "claude-3-5-haiku-20241022",
        "gemini": "gemini-2.5-flash-8b",
        "ollama": "llama3:8b",
        "stub": "stub",
    },
    "speed": {
        "openai": "gpt-4o-mini",
        "groq": "llama-3.1-8b-instant",
        "anthropic": "claude-3-5-haiku-20241022",
        "gemini": "gemini-2.5-flash-8b",
        "ollama": "llama3:8b",
        "stub": "stub",
    },
}


class StrategyPolicy:
    """Translates strategy string into actionable runtime rules."""

    @staticmethod
    def get_rules(strategy: str) -> StrategyRules:
        strat = strategy.lower().strip() if strategy else "performance"
        if strat == "cost":
            return StrategyRules(
                strategy_name="cost",
                default_model="gpt-4o-mini",
                retry_limit=1,
                enable_validation_pass=False,
                max_concurrency_multiplier=1.0,
                model_override_map=_MODEL_MAPS["cost"],
                max_tokens=512,
                timeout_seconds=60.0,
                enable_candidate_selection=False,
                enable_critique_revision=False,
            )
        elif strat == "speed":
            return StrategyRules(
                strategy_name="speed",
                default_model="gpt-4o-mini",
                retry_limit=1,
                enable_validation_pass=False,
                max_concurrency_multiplier=2.0,
                model_override_map=_MODEL_MAPS["speed"],
                max_tokens=256,
                timeout_seconds=15.0,
                enable_candidate_selection=False,
                enable_critique_revision=False,
            )
        else:  # performance
            return StrategyRules(
                strategy_name="performance",
                default_model="gpt-4o",
                retry_limit=3,
                enable_validation_pass=True,
                max_concurrency_multiplier=1.0,
                model_override_map=_MODEL_MAPS["performance"],
                max_tokens=2048,
                timeout_seconds=120.0,
                enable_candidate_selection=True,
                enable_critique_revision=True,
            )

    @staticmethod
    def resolve_model(
        strategy: str,
        provider_name: str,
        requested_model: Optional[str] = None,
        agent_type: Optional[str] = None,
        config: Optional[Any] = None,
    ) -> str:
        """Resolve model based on strategy, provider, agent type, per-agent routing, and explicit request."""
        if config and hasattr(config, "models") and config.models and agent_type:
            agent_key = agent_type.lower()
            if agent_key in config.models:
                return str(config.models[agent_key])

        if config and hasattr(config, "extra") and isinstance(config.extra, dict) and agent_type:
            models_dict = config.extra.get("models")
            if isinstance(models_dict, dict) and agent_type.lower() in models_dict:
                return str(models_dict[agent_type.lower()])

        if requested_model:
            return requested_model

        rules = StrategyPolicy.get_rules(strategy)
        provider_key = provider_name.lower()
        if provider_key in rules.model_override_map:
            return rules.model_override_map[provider_key]

        return DEFAULT_MODELS.get(provider_key, rules.default_model)
