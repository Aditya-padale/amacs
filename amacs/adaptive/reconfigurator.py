"""Reconfigurator — applies adaptation actions mid-run.

Swaps agent instances, adjusts the remaining schedule, or removes agents
from the execution plan based on :class:`AdaptationAction` directives.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from amacs.adaptive.adaptation_engine import ActionType, AdaptationAction
from amacs.agents.base_agent import BaseAgent, SubTask
from amacs.config import AMACSConfig
from amacs.integrations.llm_providers import LLMProvider
from amacs.orchestrator.agent_selector import _AGENT_REGISTRY
from amacs.orchestrator.scheduler import ExecutionPlan

logger = logging.getLogger("amacs.adaptive.reconfigurator")


class ReconfigurationResult:
    """Outcome of applying a set of adaptation actions."""

    def __init__(self) -> None:
        self.applied: List[str] = []
        self.skipped: List[str] = []

    def __repr__(self) -> str:
        return f"ReconfigurationResult(applied={len(self.applied)}, skipped={len(self.skipped)})"


class Reconfigurator:
    """Applies :class:`AdaptationAction` directives to the live execution state."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[AMACSConfig] = None,
    ) -> None:
        self._provider = provider
        self._config = config

    def apply(
        self,
        actions: List[AdaptationAction],
        agents: Dict[str, BaseAgent],
        plan: ExecutionPlan,
        remaining_wave_idx: int = 0,
    ) -> ReconfigurationResult:
        """Apply actions, mutating *agents* and *plan* in place."""
        result = ReconfigurationResult()

        for action in actions:
            if action.action_type == ActionType.NO_ACTION:
                continue

            if action.action_type == ActionType.SWAP_AGENT:
                swapped = self._swap_agent(action, agents)
                if swapped:
                    result.applied.append(
                        f"Swapped agent '{action.target_agent_id}': {action.reason}"
                    )
                else:
                    result.skipped.append(
                        f"Could not swap '{action.target_agent_id}': no alternative available"
                    )

            elif action.action_type == ActionType.SKIP_TASK:
                self._remove_from_plan(action.target_agent_id, plan, remaining_wave_idx)
                agents.pop(action.target_agent_id, None)
                result.applied.append(
                    f"Skipped task '{action.target_agent_id}': {action.reason}"
                )

            elif action.action_type == ActionType.REDUCE_TEAM:
                removed = self._reduce_team(agents, plan, remaining_wave_idx)
                result.applied.append(
                    f"Reduced team by {removed} non-critical agents"
                )

            elif action.action_type in (
                ActionType.RETRY_WITH_DIFFERENT_AGENT,
                ActionType.ADD_AGENT,
                ActionType.REMOVE_AGENT,
            ):
                # handled identically to swap for now
                swapped = self._swap_agent(action, agents)
                if swapped:
                    result.applied.append(
                        f"{action.action_type.value} for '{action.target_agent_id}'"
                    )
                else:
                    result.skipped.append(
                        f"Could not apply {action.action_type.value} "
                        f"for '{action.target_agent_id}'"
                    )

        logger.info("Reconfiguration complete: %s", result)
        return result

    # ── Internal helpers ──────────────────────────────────────────────

    def _swap_agent(
        self, action: AdaptationAction, agents: Dict[str, BaseAgent]
    ) -> bool:
        """Replace the agent for *target_agent_id* with a different type."""
        current = agents.get(action.target_agent_id)
        if current is None:
            return False

        original_type = action.params.get("original_type", current.agent_type)

        # pick a different agent type
        alternatives = [
            cls
            for name, cls in _AGENT_REGISTRY.items()
            if name != original_type
        ]
        if not alternatives:
            return False

        # use the first alternative (could be smarter in the future)
        new_cls = alternatives[0]
        agents[action.target_agent_id] = new_cls(
            provider=self._provider, config=self._config
        )
        logger.info(
            "Swapped agent %s from %s → %s",
            action.target_agent_id,
            original_type,
            new_cls.__name__,
        )
        return True

    @staticmethod
    def _remove_from_plan(
        agent_id: str, plan: ExecutionPlan, start_wave: int
    ) -> None:
        """Remove a task from remaining waves."""
        for wave in plan.waves[start_wave:]:
            wave[:] = [st for st in wave if st.id != agent_id]

    @staticmethod
    def _reduce_team(
        agents: Dict[str, BaseAgent],
        plan: ExecutionPlan,
        start_wave: int,
    ) -> int:
        """Remove non-critical tasks from the remaining plan."""
        removed = 0
        for wave in plan.waves[start_wave:]:
            to_keep: List[SubTask] = []
            for st in wave:
                if st.critical:
                    to_keep.append(st)
                else:
                    agents.pop(st.id, None)
                    removed += 1
            wave[:] = to_keep
        return removed
