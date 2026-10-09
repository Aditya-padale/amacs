"""LLM response cache module — reduces redundant API calls and latency."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from amacs.integrations.llm_providers import LLMResponse


class ResponseCache:
    """In-memory response cache keyed by hash of messages and model parameters."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache: Dict[str, LLMResponse] = {}

    @staticmethod
    def compute_key(messages: Any, model: Optional[str] = None, temperature: float = 0.7) -> str:
        data = {
            "messages": [m.content if hasattr(m, "content") else str(m) for m in messages],
            "model": model or "",
            "temperature": temperature,
        }
        raw = json.dumps(data, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[LLMResponse]:
        if not self.enabled:
            return None
        return self._cache.get(key)

    def set(self, key: str, response: LLMResponse) -> None:
        if self.enabled:
            self._cache[key] = response

    def clear(self) -> None:
        self._cache.clear()
