"""Token-aware context builder for AMACS agents.

Formats and manages shared context passed to LLMs to prevent token limit
overflows while retaining key information from prior sub-tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("amacs.context_builder")


class ContextBuilder:
    """Formats and truncates prior context to fit within token budgets."""

    DEFAULT_MAX_TOKENS = 2000  # ~8000 characters limit for prior context

    def __init__(self, max_tokens: Optional[int] = None) -> None:
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        # Estimate ~4 characters per token
        self.max_chars = self.max_tokens * 4

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count of a string (~4 chars per token)."""
        return (len(text) + 3) // 4

    def build_context_string(
        self,
        context: Dict[str, Any],
        exclude_key: Optional[str] = None,
    ) -> str:
        """Format prior context into a single prompt section.

        If total context exceeds max_chars, entries are truncated proportionally.
        """
        if not context:
            return ""

        filtered = {
            k: str(v)
            for k, v in context.items()
            if k != exclude_key and v is not None
        }
        if not filtered:
            return ""

        raw_parts = [f"--- Output of [{key}] ---\n{val}" for key, val in filtered.items()]
        full_text = "\n\n".join(raw_parts)

        if len(full_text) <= self.max_chars:
            return f"Prior context:\n{full_text}"

        logger.warning(
            "Context size (%d chars / ~%d tokens) exceeds limit (%d chars / ~%d tokens). Truncating.",
            len(full_text),
            self.estimate_tokens(full_text),
            self.max_chars,
            self.max_tokens,
        )

        # Budget per entry
        per_entry_limit = max(200, self.max_chars // len(filtered))
        truncated_parts = []
        for key, val in filtered.items():
            if len(val) > per_entry_limit:
                head = val[: per_entry_limit // 2]
                tail = val[-per_entry_limit // 4 :]
                val = f"{head}\n... [truncated {len(val) - len(head) - len(tail)} chars] ...\n{tail}"
            truncated_parts.append(f"--- Output of [{key}] ---\n{val}")

        return "Prior context (truncated):\n" + "\n\n".join(truncated_parts)
