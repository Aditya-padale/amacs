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
        self._active_spans: List[TraceSpan] = []

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
        self._active_spans.append(span)
        self._emit("span_started", span)
        return span

    def end_span(self, span: TraceSpan, metadata: Optional[Dict[str, Any]] = None) -> None:
        span.end_time = time.time()
        if metadata:
            span.metadata.update(metadata)
        if span in self._active_spans:
            self._active_spans.remove(span)
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

    def record_event(self, event: str, metadata: Optional[Dict[str, Any]] = None, parent: Optional[TraceSpan] = None, **labels: Any) -> TraceSpan:
        """Record an instantaneous, structured event in the same trace stream."""
        span = self.start_span(event, "event", {**(metadata or {}), **labels}, parent=parent or (self._active_spans[-1] if self._active_spans else None))
        self.end_span(span)
        return span

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
            status = span.metadata.get("status") or ("failed" if span.metadata.get("success") is False else "done" if span.end_time else "running")
            label = f"{span.name} ({span.span_type}) [{status}]"
            lines.append(f'    {node}["{label}"]')
            if span.parent in self.spans:
                lines.append(f"    span{self.spans.index(span.parent)} --> {node}")
        return "\n".join(lines)

    def render_execution_dag(self, plan: Any, results: Optional[List[Any]] = None) -> str:
        """Render scheduled dependencies with each executed task's final status."""
        outcomes = {getattr(r, "sub_task_id", ""): getattr(r, "success", False) for r in (results or [])}
        lines = ["flowchart TD"]
        seen: set[str] = set()
        for wave in getattr(plan, "waves", []):
            for task in wave:
                task_id = str(task.id)
                if task_id not in seen:
                    status = "success" if outcomes.get(task_id) else "failed" if task_id in outcomes else "pending"
                    lines.append(f'    {task_id}["{task_id}: {status}"]')
                    seen.add(task_id)
                for dependency in getattr(task, "dependencies", []):
                    lines.append(f"    {dependency} --> {task_id}")
        return "\n".join(lines)

    def export_opentelemetry(self, endpoint: Optional[str] = None) -> bool:
        """Best-effort OTLP export. Optional dependency; never breaks a run."""
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            provider = TracerProvider(resource=Resource.create({"service.name": "amacs"}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
            otel = trace.get_tracer("amacs")
            for span in self.spans:
                with otel.start_as_current_span(span.name) as exported:
                    for key, value in span.metadata.items():
                        exported.set_attribute(str(key), str(value))
            provider.shutdown()
            return True
        except ImportError:
            return False

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
