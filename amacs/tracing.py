"""Execution tracing and telemetry for AMACS orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceSpan:
    """A single execution span in the orchestration trajectory."""

    name: str
    span_type: str  # e.g. "agent_run", "task_decomposition", "wave_execution"
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


class Tracer:
    """Collects structured trace spans for performance profiling."""

    def __init__(self) -> None:
        self.spans: List[TraceSpan] = []

    def start_span(self, name: str, span_type: str, metadata: Optional[Dict[str, Any]] = None) -> TraceSpan:
        span = TraceSpan(
            name=name,
            span_type=span_type,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self.spans.append(span)
        return span

    def end_span(self, span: TraceSpan, metadata: Optional[Dict[str, Any]] = None) -> None:
        span.end_time = time.time()
        if metadata:
            span.metadata.update(metadata)

    def summary(self) -> Dict[str, Any]:
        total_time = sum(s.duration_seconds for s in self.spans)
        return {
            "total_spans": len(self.spans),
            "total_duration_seconds": total_time,
            "spans": [
                {
                    "name": s.name,
                    "type": s.span_type,
                    "duration": f"{s.duration_seconds:.3f}s",
                }
                for s in self.spans
            ],
        }
