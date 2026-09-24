"""AMACS configuration — parsed from ``@amacs(...)`` decorator parameters.

Uses Pydantic for validation so invalid configs fail fast with clear messages.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from amacs.exceptions import ConfigurationError


class Strategy(str, Enum):
    """Execution strategy that biases orchestration decisions."""

    PERFORMANCE = "performance"
    COST = "cost"
    SPEED = "speed"


class AMACSConfig(BaseModel):
    """Immutable configuration for a single ``@amacs``-decorated call."""

    max_agents: int = Field(default=4, ge=1, le=64, description="Max concurrent agents.")
    strategy: Strategy = Field(default=Strategy.PERFORMANCE, description="Optimization strategy.")
    adaptive: bool = Field(default=False, description="Enable the adaptive control loop.")
    retry_limit: int = Field(default=3, ge=0, le=10, description="Per-agent retry limit.")
    timeout: float = Field(default=120.0, gt=0, description="Per-agent timeout in seconds.")
    llm_provider: Optional[str] = Field(
        default=None,
        description=(
            "LLM provider to use. One of 'openai', 'anthropic', 'gemini', 'ollama'. "
            "Falls back to AMACS_LLM_PROVIDER env var, then 'openai'."
        ),
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Model name (e.g. 'gpt-4o'). Falls back to provider default.",
    )
    skip_non_critical: bool = Field(
        default=True,
        description="If True, a failed non-critical sub-task is skipped instead of crashing.",
    )
    verbose: bool = Field(
        default=False,
        description="Log and print detailed agent responses and inter-agent communication.",
    )
    return_details: bool = Field(
        default=False,
        description="Return an AMACSResult object with full agent outputs and communication log.",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra parameters forwarded to agents and hooks.",
    )

    model_config = {"frozen": True}

    @field_validator("strategy", mode="before")
    @classmethod
    def _coerce_strategy(cls, v: Any) -> Strategy:
        if isinstance(v, str):
            try:
                return Strategy(v.lower())
            except ValueError:
                raise ConfigurationError(
                    f"Invalid strategy '{v}'. Choose from: "
                    f"{', '.join(s.value for s in Strategy)}"
                ) from None
        return v


def build_config(**kwargs: Any) -> AMACSConfig:
    """Build and validate an :class:`AMACSConfig` from raw decorator kwargs.

    Raises :class:`ConfigurationError` on validation failure.
    """
    try:
        return AMACSConfig(**kwargs)
    except Exception as exc:
        raise ConfigurationError(f"Invalid AMACS configuration: {exc}") from exc
