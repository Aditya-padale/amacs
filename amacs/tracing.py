"""Execution tracing and telemetry for AMACS orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TraceSpan:
    """A single execution span in the orchestration trajectory."""

    name: str
    span_type: str  # e.g. "agent_run", "task_decomposition", "wave_execution"
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[TraceSpan] = field(default=None, repr=False)

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


class Tracer:
    """Collects structured trace spans for performance profiling."""

    def __init__(self, on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.spans: List[TraceSpan] = []
        self._on_event = on_event

    def start_span(
        self,
        name: str,
        span_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional[TraceSpan] = None,
    ) -> TraceSpan:
        span = TraceSpan(
            name=name,
            span_type=span_type,
            start_time=time.time(),
            metadata=metadata or {},
            parent=parent,
        )
        self.spans.append(span)
        self._emit("span_started", span)
        return span

    def end_span(self, span: TraceSpan, metadata: Optional[Dict[str, Any]] = None) -> None:
        span.end_time = time.time()
        if metadata:
            span.metadata.update(metadata)
        self._emit("span_finished", span)

    def _emit(self, event: str, span: TraceSpan) -> None:
        if self._on_event:
            self._on_event({
                "event": event,
                "name": span.name,
                "type": span.span_type,
                "metadata": dict(span.metadata),
                "duration_seconds": span.duration_seconds,
            })

    def _span_dict(self, span: TraceSpan) -> Dict[str, Any]:
        return {
            "name": span.name,
            "type": span.span_type,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_seconds": span.duration_seconds,
            "metadata": dict(span.metadata),
            "parent": span.parent.name if span.parent else None,
        }

    def to_json(self) -> str:
        """Return the execution spans as stable, JSON-serializable data."""
        return json.dumps({"spans": [self._span_dict(span) for span in self.spans]}, sort_keys=True)

    def to_mermaid(self) -> str:
        """Render the span hierarchy as a Mermaid flowchart."""
        lines = ["flowchart TD"]
        for index, span in enumerate(self.spans):
            node = f"span{index}"
            label = f"{span.name} ({span.span_type})"
            lines.append(f'    {node}["{label}"]')
            if span.parent in self.spans:
                lines.append(f"    span{self.spans.index(span.parent)} --> {node}")
        return "\n".join(lines)

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
