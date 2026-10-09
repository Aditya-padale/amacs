"""Evaluator — compares agent performance against configurable thresholds.

Flags agents as "underperforming" when they exceed failure, latency, or quality
signal limits, feeding actionable signals to the :class:`AdaptationEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from amacs.adaptive.monitor import AgentMetrics, Monitor

REFUSAL_PATTERNS = [
    "i cannot",
    "i am unable to",
    "as an ai",
    "i apologize, but",
    "i can't fulfill",
    "i'm sorry, but i cannot",
    "as a language model",
]


def is_repeated_text(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 3 and len(set(lines)) == 1:
        return True
    words = text.split()
    if len(words) >= 12:
        chunk_len = 3
        chunks = [tuple(words[i : i + chunk_len]) for i in range(len(words) - chunk_len + 1)]
        if len(chunks) >= 4 and len(set(chunks)) * 3 <= len(chunks):
            return True
    return False


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"


@dataclass
class ThresholdConfig:
    """Configurable thresholds for the evaluator."""

    max_failures: int = 2
    max_latency_seconds: float = 30.0
    max_error_rate: float = 0.5  # 50 %
    min_calls_for_evaluation: int = 1
    min_output_length: int = 20
    min_llm_judge_score: float = 0.6
    max_budget_burn_rate: float = 0.8
    detect_refusal: bool = True
    detect_repeated: bool = True


@dataclass
class AgentEvaluation:
    """Evaluation result for a single agent."""

    agent_id: str
    agent_type: str
    status: HealthStatus
    reasons: List[str] = field(default_factory=list)
    metrics: Optional[AgentMetrics] = None
    trigger: str = ""
    signal_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Full evaluation report across all agents."""

    evaluations: Dict[str, AgentEvaluation] = field(default_factory=dict)
    underperforming: List[str] = field(default_factory=list)  # agent_ids
    system_healthy: bool = True


class Evaluator:
    """Evaluates agent health against thresholds and quality signals."""

    def __init__(
        self,
        thresholds: Optional[ThresholdConfig] = None,
    ) -> None:
        self._thresholds = thresholds or ThresholdConfig()

    def evaluate(self, monitor: Monitor) -> EvaluationReport:
        """Produce an :class:`EvaluationReport` from current monitor state."""
        snap = monitor.snapshot()
        report = EvaluationReport()

        for agent_id, metrics in snap.agent_metrics.items():
            if metrics is None:
                continue
            ev = self._evaluate_agent(metrics)
            report.evaluations[agent_id] = ev
            if ev.status != HealthStatus.HEALTHY:
                report.underperforming.append(agent_id)

        report.system_healthy = len(report.underperforming) == 0
        return report

    def _evaluate_agent(self, m: AgentMetrics) -> AgentEvaluation:
        reasons: List[str] = []
        status = HealthStatus.HEALTHY
        trigger = ""

        if m.total_calls < self._thresholds.min_calls_for_evaluation:
            return AgentEvaluation(
                agent_id=m.agent_id,
                agent_type=m.agent_type,
                status=HealthStatus.HEALTHY,
                metrics=m,
                trigger="healthy",
            )

        # 1. failure count
        if m.failed_calls >= self._thresholds.max_failures:
            reasons.append(
                f"Too many failures: {m.failed_calls} >= {self._thresholds.max_failures}"
            )
            status = HealthStatus.FAILING
            trigger = trigger or "failure:max_failures"

        # 2. error rate
        error_rate = m.failed_calls / m.total_calls if m.total_calls > 0 else 0
        if error_rate >= self._thresholds.max_error_rate:
            reasons.append(
                f"High error rate: {error_rate:.0%} >= {self._thresholds.max_error_rate:.0%}"
            )
            status = HealthStatus.FAILING
            trigger = trigger or "failure:error_rate"

        # 3. latency
        avg_latency = m.total_latency / m.total_calls if m.total_calls > 0 else 0
        if avg_latency > self._thresholds.max_latency_seconds:
            reasons.append(
                f"High avg latency: {avg_latency:.1f}s > {self._thresholds.max_latency_seconds}s"
            )
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            trigger = trigger or "latency:high"

        # 4. Quality Signals (trigger even when nothing failed!)
        if m.last_content is not None and m.successful_calls > 0:
            content_stripped = m.last_content.strip()
            if len(content_stripped) < self._thresholds.min_output_length:
                reasons.append(
                    f"Empty or very short output ({len(content_stripped)} chars < {self._thresholds.min_output_length})"
                )
                status = HealthStatus.FAILING if len(content_stripped) == 0 else HealthStatus.DEGRADED
                trigger = trigger or "quality_signal:short_output"

            if self._thresholds.detect_refusal and content_stripped:
                content_lower = content_stripped.lower()
                if any(pat in content_lower for pat in REFUSAL_PATTERNS):
                    reasons.append("Refusal pattern detected in response")
                    status = HealthStatus.FAILING
                    trigger = trigger or "quality_signal:refusal_pattern"

            if (
                self._thresholds.detect_repeated
                and content_stripped
                and is_repeated_text(content_stripped)
            ):
                reasons.append("Repeated output pattern detected")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                trigger = trigger or "quality_signal:repeated_output"

        if m.quality_scores:
            latest_score = m.quality_scores[-1]
            if latest_score < self._thresholds.min_llm_judge_score:
                reasons.append(
                    f"Low LLM judge score: {latest_score:.2f} < {self._thresholds.min_llm_judge_score:.2f}"
                )
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                trigger = trigger or "quality_signal:low_judge_score"

        if m.burn_rate > self._thresholds.max_budget_burn_rate:
            reasons.append(
                f"High budget burn rate: {m.burn_rate:.2f} > {self._thresholds.max_budget_burn_rate:.2f}"
            )
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            trigger = trigger or "quality_signal:budget_burn_rate"

        signal_values = {
            "failed_calls": m.failed_calls,
            "error_rate": error_rate,
            "avg_latency": avg_latency,
            "content_length": len(m.last_content.strip()) if m.last_content else 0,
            "quality_scores": m.quality_scores,
            "burn_rate": m.burn_rate,
        }

        return AgentEvaluation(
            agent_id=m.agent_id,
            agent_type=m.agent_type,
            status=status,
            reasons=reasons,
            metrics=m,
            trigger=trigger or ("healthy" if status == HealthStatus.HEALTHY else "degraded"),
            signal_values=signal_values,
        )
