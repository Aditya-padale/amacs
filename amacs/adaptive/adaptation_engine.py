"""Adaptation engine — decides what remedial action to take based on evaluations.

Translates :class:`EvaluationReport` findings into concrete :class:`AdaptationAction`
directives that the :class:`Reconfigurator` can apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from amacs.adaptive.evaluator import EvaluationReport, HealthStatus


class ActionType(str, Enum):
    RETRY_WITH_DIFFERENT_AGENT = "retry_with_different_agent"
    SWITCH_MODEL = "switch_model"
    ADD_AGENT = "add_agent"
    REMOVE_AGENT = "remove_agent"
    REDUCE_TEAM = "reduce_team"
    SWAP_AGENT = "swap_agent"
    SKIP_TASK = "skip_task"
    NO_ACTION = "no_action"


@dataclass
class AdaptationAction:
    """A single remedial action to be executed by the reconfigurator."""

    action_type: ActionType
    target_agent_id: str
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)


class AdaptationEngine:
    """Decides what changes to make based on evaluation results."""

    def decide(
        self,
        report: EvaluationReport,
        total_agents: int = 0,
    ) -> List[AdaptationAction]:
        """Return an ordered list of actions to take."""
        actions: List[AdaptationAction] = []

        if report.system_healthy:
            return [
                AdaptationAction(
                    action_type=ActionType.NO_ACTION,
                    target_agent_id="",
                    reason="System is healthy — no adaptation needed.",
                )
            ]

        failing_count = 0
        for agent_id in report.underperforming:
            ev = report.evaluations[agent_id]
            params = {
                "original_type": ev.agent_type,
                "trigger": ev.trigger,
                "signal_values": ev.signal_values,
            }

            if ev.status == HealthStatus.FAILING:
                failing_count += 1
                actions.append(
                    AdaptationAction(
                        action_type=ActionType.SWAP_AGENT,
                        target_agent_id=agent_id,
                        reason="; ".join(ev.reasons),
                        params=params,
                    )
                )
            elif ev.status == HealthStatus.DEGRADED:
                # Quality triggers lead to RETRY_WITH_DIFFERENT_AGENT / SWITCH_MODEL
                if ev.trigger.startswith("quality_signal:"):
                    if ev.signal_values.get("quality_score", 1.0) < 0.35 and ev.signal_values.get("critical", False):
                        actions.append(
                            AdaptationAction(
                                action_type=ActionType.ADD_AGENT,
                                target_agent_id=agent_id,
                                reason="Critical task has a persistently low quality signal.",
                                params=params,
                            )
                        )
                    actions.append(
                        AdaptationAction(
                            action_type=ActionType.RETRY_WITH_DIFFERENT_AGENT,
                            target_agent_id=agent_id,
                            reason="; ".join(ev.reasons),
                            params=params,
                        )
                    )
                else:
                    actions.append(
                        AdaptationAction(
                            action_type=ActionType.SKIP_TASK,
                            target_agent_id=agent_id,
                            reason="; ".join(ev.reasons),
                            params=params,
                        )
                    )

        # systemic issue: reduce team
        if total_agents > 0 and failing_count > total_agents / 2:
            actions.append(
                AdaptationAction(
                    action_type=ActionType.REDUCE_TEAM,
                    target_agent_id="",
                    reason=f"{failing_count}/{total_agents} agents are failing — reducing team.",
                )
            )

        return actions
