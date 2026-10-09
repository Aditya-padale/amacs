"""Token-aware context builder for AMACS agents.

Formats and manages shared context passed to LLMs to prevent token limit
overflows while retaining key information from prior sub-tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("amacs.context_builder")


import logging
from typing import Callable, Sequence

logger = logging.getLogger("amacs.context_builder")


def _get_default_token_counter() -> Callable[[str], int]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    except ImportError:
        return lambda text: (len(text) + 3) // 4


class ContextBuilder:
    """Formats and truncates prior context to fit within token budgets."""

    DEFAULT_MAX_TOKENS = 2000

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.token_counter = token_counter or _get_default_token_counter()
        self.was_truncated: bool = False
        self.truncation_record: Optional[Dict[str, Any]] = None

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count of a string (~4 chars per token)."""
        return (len(text) + 3) // 4

    def build_context_string(
        self,
        context: Dict[str, Any],
        exclude_key: Optional[str] = None,
        allowed_keys: Optional[Sequence[str]] = None,
    ) -> str:
        """Format prior context into a single prompt section.

        Only keys in allowed_keys (or declared dependencies + original_input) are included.
        If total context exceeds max_tokens, entries are truncated with an explicit marker.
        """
        if not context:
            return ""

        filtered: Dict[str, str] = {}
        for k, v in context.items():
            if k == exclude_key or v is None:
                continue
            if allowed_keys is not None and k not in allowed_keys:
                continue
            filtered[k] = str(v)

        if not filtered:
            return ""

        raw_parts = [f"--- Output of [{key}] ---\n{val}" for key, val in filtered.items()]
        full_text = "\n\n".join(raw_parts)
        total_tokens = self.token_counter(full_text)

        if total_tokens <= self.max_tokens:
            return f"Prior context:\n{full_text}"

        self.was_truncated = True
        logger.warning(
            "Context size (%d tokens) exceeds max_context_tokens limit (%d tokens). Truncating.",
            total_tokens,
            self.max_tokens,
        )

        per_entry_tokens = max(1, self.max_tokens // len(filtered))
        truncated_parts = []
        for key, val in filtered.items():
            val_tokens = self.token_counter(val)
            if val_tokens > per_entry_tokens:
                ratio = per_entry_tokens / val_tokens
                cut_len = max(100, int(len(val) * ratio))
                head = val[: cut_len // 2]
                tail = val[-cut_len // 4 :]
                val = f"{head}\n[TRUNCATED: Context entry '{key}' exceeded token budget]\n{tail}"
            truncated_parts.append(f"--- Output of [{key}] ---\n{val}")

        final_text = "Prior context (truncated):\n" + "\n\n".join(truncated_parts)
        self.truncation_record = {
            "original_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "final_tokens": self.token_counter(final_text),
            "keys_included": list(filtered.keys()),
        }
        return final_text
