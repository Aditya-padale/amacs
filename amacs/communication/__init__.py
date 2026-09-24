"""In-memory communication bus for inter-agent knowledge sharing.

Provides a pub/sub-style shared context that agents read from and write to
during a single orchestrated call.  Conflict resolution uses last-write-wins
with an append-only audit log so nothing is silently lost.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class LogEntry:
    """Single entry in the bus audit log."""

    timestamp: float
    key: str
    value: Any
    writer: str
    overwritten: bool = False


class CommunicationBus:
    """Thread-safe shared-context bus with pub/sub and conflict logging.

    Parameters
    ----------
    merge_strategy:
        Optional callable ``(old_value, new_value) -> merged_value`` used
        instead of last-write-wins when a key is overwritten.
    """

    def __init__(
        self,
        merge_strategy: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        self._store: Dict[str, Any] = {}
        self._log: List[LogEntry] = []
        self._subscribers: Dict[str, List[Callable[[str, Any], None]]] = {}
        self._lock = threading.RLock()
        self._merge_strategy = merge_strategy

    # ── write ──────────────────────────────────────────────────────────
    def publish(self, key: str, value: Any, writer: str = "unknown") -> None:
        """Write *value* under *key*, notifying subscribers."""
        with self._lock:
            overwritten = key in self._store
            if overwritten and self._merge_strategy is not None:
                value = self._merge_strategy(self._store[key], value)
            self._store[key] = value
            self._log.append(
                LogEntry(
                    timestamp=time.time(),
                    key=key,
                    value=value,
                    writer=writer,
                    overwritten=overwritten,
                )
            )
            # notify subscribers
            for cb in self._subscribers.get(key, []):
                cb(key, value)
            for cb in self._subscribers.get("*", []):
                cb(key, value)

    # ── read ───────────────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        """Read the latest value for *key*."""
        with self._lock:
            return self._store.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of the entire shared context."""
        with self._lock:
            return dict(self._store)

    # ── subscribe ──────────────────────────────────────────────────────
    def subscribe(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """Register *callback* to be invoked when *key* is published.

        Use ``key="*"`` to subscribe to all keys.
        """
        with self._lock:
            self._subscribers.setdefault(key, []).append(callback)

    # ── audit log ──────────────────────────────────────────────────────
    def get_log(self) -> List[LogEntry]:
        """Return the full audit log (append-only, never truncated)."""
        with self._lock:
            return list(self._log)

    def get_conflicts(self) -> List[LogEntry]:
        """Return only log entries where a key was overwritten."""
        with self._lock:
            return [e for e in self._log if e.overwritten]

    # ── housekeeping ───────────────────────────────────────────────────
    def clear(self) -> None:
        """Reset the bus (useful between test runs)."""
        with self._lock:
            self._store.clear()
            self._log.clear()
            self._subscribers.clear()

    def __repr__(self) -> str:
        return f"CommunicationBus(keys={list(self._store.keys())}, log_entries={len(self._log)})"
