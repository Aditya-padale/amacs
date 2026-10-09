"""Pricing table and cost calculation module for AMACS.

Rates per 1,000,000 tokens as of 2026-10-09.
Source docs:
- OpenAI: https://platform.openai.com/docs/models
- Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
- Gemini: https://ai.google.dev/gemini-api/docs/models/gemini
"""

from __future__ import annotations

from typing import Dict, Optional

# Price table per 1M tokens (USD) as of 2026-10-09
PRICE_TABLE: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    # Anthropic
    "claude-sonnet-4-20250514": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-haiku-20241022": {"prompt": 0.80, "completion": 4.00},
    # Gemini
    "gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30},
    "gemini-2.5-flash-8b": {"prompt": 0.0375, "completion": 0.15},
    # Ollama / Stub (local / zero-cost models)
    "llama3": {"prompt": 0.0, "completion": 0.0},
    "llama3:8b": {"prompt": 0.0, "completion": 0.0},
    "stub": {"prompt": 0.0, "completion": 0.0},
}

# Safe default for unknown commercial models (per 1M tokens in USD)
DEFAULT_PRICE: Dict[str, float] = {"prompt": 2.00, "completion": 6.00}


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    price_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Compute cost in USD given model name and token counts.

    Parameters
    ----------
    model : str
        Model identifier.
    prompt_tokens : int
        Number of prompt tokens used.
    completion_tokens : int
        Number of completion tokens generated.
    price_overrides : Optional[Dict[str, Dict[str, float]]]
        Documented user override table mapping model names to prompt/completion USD per 1M tokens.
    """
    table = dict(PRICE_TABLE)
    if price_overrides:
        table.update({k.lower(): v for k, v in price_overrides.items()})

    model_key = model.lower() if model else "unknown"
    rates = table.get(model_key, DEFAULT_PRICE)

    prompt_cost = (prompt_tokens / 1_000_000.0) * rates.get("prompt", DEFAULT_PRICE["prompt"])
    completion_cost = (completion_tokens / 1_000_000.0) * rates.get("completion", DEFAULT_PRICE["completion"])

    return prompt_cost + completion_cost
