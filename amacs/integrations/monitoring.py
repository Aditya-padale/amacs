"""Optional Prometheus metrics export — enabled via ``pip install amacs[monitoring]``."""

from __future__ import annotations

from typing import Any

from amacs.exceptions import AMACSError

# Sentinel; replaced by real metrics when Prometheus client is available.
_METRICS_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore[import-untyped]

    _METRICS_AVAILABLE = True

    AGENT_CALLS = Counter(
        "amacs_agent_calls_total",
        "Total agent invocations",
        ["agent_type", "status"],
    )
    AGENT_LATENCY = Histogram(
        "amacs_agent_latency_seconds",
        "Agent call latency in seconds",
        ["agent_type"],
        buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
    )
    ACTIVE_AGENTS = Gauge(
        "amacs_active_agents",
        "Number of concurrently running agents",
    )
    TOKEN_USAGE = Counter(
        "amacs_token_usage_total",
        "Total LLM tokens consumed",
        ["agent_type", "token_type"],
    )
except ImportError:
    pass


class MetricsRecorder:
    """Thin wrapper that no-ops gracefully when prometheus_client is absent."""

    def __init__(self) -> None:
        if not _METRICS_AVAILABLE:
            return

    @staticmethod
    def record_call(agent_type: str, status: str = "success") -> None:
        if _METRICS_AVAILABLE:
            AGENT_CALLS.labels(agent_type=agent_type, status=status).inc()

    @staticmethod
    def observe_latency(agent_type: str, seconds: float) -> None:
        if _METRICS_AVAILABLE:
            AGENT_LATENCY.labels(agent_type=agent_type).observe(seconds)

    @staticmethod
    def set_active_agents(count: int) -> None:
        if _METRICS_AVAILABLE:
            ACTIVE_AGENTS.set(count)

    @staticmethod
    def record_tokens(agent_type: str, prompt: int, completion: int) -> None:
        if _METRICS_AVAILABLE:
            TOKEN_USAGE.labels(agent_type=agent_type, token_type="prompt").inc(prompt)
            TOKEN_USAGE.labels(agent_type=agent_type, token_type="completion").inc(completion)

    @staticmethod
    def is_available() -> bool:
        return _METRICS_AVAILABLE


def start_metrics_server(port: int = 8000) -> Any:
    """Start a Prometheus HTTP server on *port*."""
    if not _METRICS_AVAILABLE:
        raise AMACSError(
            "prometheus_client not installed. Run: pip install amacs[monitoring]"
        )
    from prometheus_client import start_http_server  # type: ignore[import-untyped]

    return start_http_server(port)
