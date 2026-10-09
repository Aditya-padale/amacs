"""Monitor — tracks latency, token usage, error rates, and agent health.

Provides a real-time view of how each agent is performing during an
orchestrated execution so the :class:`Evaluator` can flag problems.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentMetrics:
    """Accumulated metrics for a single agent instance."""

    agent_id: str
    agent_type: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency: float = 0.0
    total_tokens: int = 0
    last_latency: float = 0.0
    last_error: Optional[str] = None
    timestamps: List[float] = field(default_factory=list)


@dataclass
class SystemSnapshot:
    """Point-in-time snapshot of the entire system's health."""

    timestamp: float
    active_agents: int
    total_calls: int
    total_failures: int
    avg_latency: float
    total_tokens: int
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)


class Monitor:
    """Thread-safe performance monitor for the adaptive control loop."""

    def __init__(self) -> None:
        self._metrics: Dict[str, AgentMetrics] = {}
        self._lock = threading.RLock()
        self._start_time = time.time()

    def record_success(
        self,
        agent_id: str,
        agent_type: str,
        latency: float,
        tokens: int = 0,
    ) -> None:
        """Record a successful agent execution."""
        with self._lock:
            m = self._get_or_create(agent_id, agent_type)
            m.total_calls += 1
            m.successful_calls += 1
            m.total_latency += latency
            m.last_latency = latency
            m.total_tokens += tokens
            m.timestamps.append(time.time())

    def record_failure(
        self,
        agent_id: str,
        agent_type: str,
        latency: float,
        error: str,
    ) -> None:
        """Record a failed agent execution."""
        with self._lock:
            m = self._get_or_create(agent_id, agent_type)
            m.total_calls += 1
            m.failed_calls += 1
            m.total_latency += latency
            m.last_latency = latency
            m.last_error = error
            m.timestamps.append(time.time())

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Return metrics for a specific agent."""
        with self._lock:
            m = self._metrics.get(agent_id)
            if m is None:
                return None
            # return a copy
            return AgentMetrics(
                agent_id=m.agent_id,
                agent_type=m.agent_type,
                total_calls=m.total_calls,
                successful_calls=m.successful_calls,
                failed_calls=m.failed_calls,
                total_latency=m.total_latency,
                total_tokens=m.total_tokens,
                last_latency=m.last_latency,
                last_error=m.last_error,
                timestamps=list(m.timestamps),
            )

    def snapshot(self) -> SystemSnapshot:
        """Capture a point-in-time snapshot of all agent metrics."""
        with self._lock:
            total_calls = sum(m.total_calls for m in self._metrics.values())
            total_failures = sum(m.failed_calls for m in self._metrics.values())
            total_latency = sum(m.total_latency for m in self._metrics.values())
            total_tokens = sum(m.total_tokens for m in self._metrics.values())
            avg_latency = total_latency / total_calls if total_calls > 0 else 0.0

            return SystemSnapshot(
                timestamp=time.time(),
                active_agents=len(self._metrics),
                total_calls=total_calls,
                total_failures=total_failures,
                avg_latency=avg_latency,
                total_tokens=total_tokens,
                agent_metrics={k: self.get_agent_metrics(k) for k in self._metrics},  # type: ignore[misc]
            )

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._start_time = time.time()

    def _get_or_create(self, agent_id: str, agent_type: str) -> AgentMetrics:
        if agent_id not in self._metrics:
            self._metrics[agent_id] = AgentMetrics(
                agent_id=agent_id, agent_type=agent_type
            )
        return self._metrics[agent_id]
