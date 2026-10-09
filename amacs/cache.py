"""LLM response cache module — reduces redundant API calls and latency."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

from amacs.integrations.llm_providers import LLMResponse


class ResponseCache:
    """In-memory and file-backed response cache keyed by provider, model, messages and parameters."""

    def __init__(self, enabled: bool = True, filepath: Optional[str] = None) -> None:
        self.enabled = enabled
        self.filepath = filepath
        self._cache: Dict[str, LLMResponse] = {}
        if self.filepath and os.path.exists(self.filepath):
            self.load_from_disk(self.filepath)

    @staticmethod
    def compute_key(
        messages: Any,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        data = {
            "provider": provider or "",
            "messages": [m.content if hasattr(m, "content") else str(m) for m in messages],
            "model": model or "",
            "temperature": temperature,
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
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
            if self.filepath:
                self.save_to_disk(self.filepath)

    def clear(self) -> None:
        self._cache.clear()

    def load_from_disk(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            for k, v in raw_data.items():
                self._cache[k] = LLMResponse(
                    content=v.get("content", ""),
                    model=v.get("model", ""),
                    usage=v.get("usage", {}),
                )
        except Exception as exc:
            import logging
            logging.getLogger("amacs.cache").warning("Failed to load cache from disk (%s): %s", path, exc)

    def save_to_disk(self, path: str) -> None:
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            data = {
                k: {
                    "content": v.content,
                    "model": v.model,
                    "usage": v.usage,
                }
                for k, v in self._cache.items()
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            import logging
            logging.getLogger("amacs.cache").warning("Failed to save cache to disk (%s): %s", path, exc)
