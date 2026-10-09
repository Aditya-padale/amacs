"""Evaluator — compares agent performance against configurable thresholds.

Flags agents as "underperforming" when they exceed failure or latency limits,
feeding actionable signals to the :class:`AdaptationEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from amacs.adaptive.monitor import AgentMetrics, Monitor


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


@dataclass
class AgentEvaluation:
    """Evaluation result for a single agent."""

    agent_id: str
    agent_type: str
    status: HealthStatus
    reasons: List[str] = field(default_factory=list)
    metrics: Optional[AgentMetrics] = None


@dataclass
class EvaluationReport:
    """Full evaluation report across all agents."""

    evaluations: Dict[str, AgentEvaluation] = field(default_factory=dict)
    underperforming: List[str] = field(default_factory=list)  # agent_ids
    system_healthy: bool = True


class Evaluator:
    """Evaluates agent health against thresholds."""

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

        if m.total_calls < self._thresholds.min_calls_for_evaluation:
            return AgentEvaluation(
                agent_id=m.agent_id,
                agent_type=m.agent_type,
                status=HealthStatus.HEALTHY,
                metrics=m,
            )

        # failure count
        if m.failed_calls >= self._thresholds.max_failures:
            reasons.append(
                f"Too many failures: {m.failed_calls} >= {self._thresholds.max_failures}"
            )
            status = HealthStatus.FAILING

        # error rate
        error_rate = m.failed_calls / m.total_calls if m.total_calls > 0 else 0
        if error_rate >= self._thresholds.max_error_rate:
            reasons.append(
                f"High error rate: {error_rate:.0%} >= {self._thresholds.max_error_rate:.0%}"
            )
            status = HealthStatus.FAILING

        # latency
        avg_latency = m.total_latency / m.total_calls if m.total_calls > 0 else 0
        if avg_latency > self._thresholds.max_latency_seconds:
            reasons.append(
                f"High avg latency: {avg_latency:.1f}s > {self._thresholds.max_latency_seconds}s"
            )
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED

        return AgentEvaluation(
            agent_id=m.agent_id,
            agent_type=m.agent_type,
            status=status,
            reasons=reasons,
            metrics=m,
        )
