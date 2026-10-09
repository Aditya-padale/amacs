"""Strategy policy module — translates configuration strategy into execution rules.

Supported strategies:
- ``performance``: Highest quality, uses main models, validation pass enabled, full retries.
- ``cost``: Minimises API expenditure, uses lightweight models, skips optional validation.
- ``speed``: Minimises overall latency, enables high concurrency, skips optional validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class StrategyRules:
    """Rules derived from the chosen execution strategy."""

    strategy_name: str
    default_model: str
    retry_limit: int
    enable_validation_pass: bool
    max_concurrency_multiplier: float
    model_override_map: Dict[str, str]


_MODEL_MAPS: Dict[str, Dict[str, str]] = {
    "performance": {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "gemini": "gemini-2.5-flash",
        "ollama": "llama3",
        "stub": "stub",
    },
    "cost": {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
        "gemini": "gemini-2.5-flash-8b",
        "ollama": "llama3:8b",
        "stub": "stub",
    },
    "speed": {
        "openai": "gpt-4o-mini",
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
        strat = strategy.lower().strip()
        if strat == "cost":
            return StrategyRules(
                strategy_name="cost",
                default_model="gpt-4o-mini",
                retry_limit=1,
                enable_validation_pass=False,
                max_concurrency_multiplier=1.0,
                model_override_map=_MODEL_MAPS["cost"],
            )
        elif strat == "speed":
            return StrategyRules(
                strategy_name="speed",
                default_model="gpt-4o-mini",
                retry_limit=1,
                enable_validation_pass=False,
                max_concurrency_multiplier=2.0,
                model_override_map=_MODEL_MAPS["speed"],
            )
        else:  # performance
            return StrategyRules(
                strategy_name="performance",
                default_model="gpt-4o",
                retry_limit=3,
                enable_validation_pass=True,
                max_concurrency_multiplier=1.0,
                model_override_map=_MODEL_MAPS["performance"],
            )

    @staticmethod
    def resolve_model(strategy: str, provider_name: str, requested_model: Optional[str] = None) -> str:
        """Resolve the model to use given strategy, provider, and explicit request."""
        if requested_model:
            return requested_model
        rules = StrategyPolicy.get_rules(strategy)
        return rules.model_override_map.get(provider_name.lower(), rules.default_model)
