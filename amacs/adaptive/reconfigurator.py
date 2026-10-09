"""Reconfigurator — applies adaptation actions mid-run.

Swaps agent instances, adjusts the remaining schedule, or removes agents
from the execution plan based on :class:`AdaptationAction` directives.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from amacs.adaptive.adaptation_engine import ActionType, AdaptationAction
from amacs.agents.base_agent import BaseAgent, SubTask
from amacs.config import AMACSConfig
from amacs.integrations.llm_providers import LLMProvider
from amacs.orchestrator.agent_selector import get_alternative_agent_classes
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

    MAX_SWAPS_PER_TASK = 2

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[AMACSConfig] = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._swap_counts: Dict[str, int] = {}

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

            if action.action_type in (
                ActionType.SWAP_AGENT,
                ActionType.RETRY_WITH_DIFFERENT_AGENT,
            ):
                swapped = self._swap_agent(action, agents, plan, remaining_wave_idx)
                if swapped:
                    result.applied.append(
                        f"Swapped agent for '{action.target_agent_id}': {action.reason}"
                    )
                else:
                    result.skipped.append(
                        f"Could not swap '{action.target_agent_id}': no alternative available or max swaps reached"
                    )

            elif action.action_type == ActionType.SWITCH_MODEL:
                swapped = self._swap_agent(action, agents, plan, remaining_wave_idx)
                if swapped:
                    result.applied.append(
                        f"Switched model/agent for '{action.target_agent_id}': {action.reason}"
                    )
                else:
                    result.skipped.append(
                        f"Could not switch model for '{action.target_agent_id}'"
                    )

            elif action.action_type == ActionType.ADD_AGENT:
                added = self._add_agent(action, agents, plan, remaining_wave_idx)
                if added:
                    result.applied.append(
                        f"Added supplementary agent for '{action.target_agent_id}': {action.reason}"
                    )
                else:
                    result.skipped.append(
                        f"Could not add supplementary agent for '{action.target_agent_id}'"
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

            elif action.action_type == ActionType.REMOVE_AGENT:
                self._remove_from_plan(action.target_agent_id, plan, remaining_wave_idx)
                agents.pop(action.target_agent_id, None)
                result.applied.append(
                    f"Removed agent for '{action.target_agent_id}'"
                )

        logger.info("Reconfiguration complete: %s", result)
        return result

    # ── Internal helpers ──────────────────────────────────────────────

    def _swap_agent(
        self,
        action: AdaptationAction,
        agents: Dict[str, BaseAgent],
        plan: ExecutionPlan,
        remaining_wave_idx: int = 0,
    ) -> bool:
        """Replace agent with a different type."""
        target_id = action.target_agent_id

        # If target_id is not in agents, check remaining tasks
        if target_id not in agents:
            remaining_task_ids = set()
            if plan and plan.waves:
                for w in plan.waves[remaining_wave_idx:]:
                    for st in w:
                        remaining_task_ids.add(st.id)

            orig_type = action.params.get("original_type")
            for tid in remaining_task_ids:
                ag = agents.get(tid)
                if ag and orig_type and ag.agent_type.lower() == str(orig_type).lower():
                    target_id = tid
                    break
            else:
                if remaining_task_ids:
                    target_id = next(iter(remaining_task_ids))

        current = agents.get(target_id)
        if current is None:
            return False

        # Limit swaps per task to prevent infinite swap loops
        swap_count = self._swap_counts.get(target_id, 0)
        if swap_count >= self.MAX_SWAPS_PER_TASK:
            logger.warning(
                "Max swap limit (%d) reached for subtask '%s'",
                self.MAX_SWAPS_PER_TASK,
                target_id,
            )
            return False

        original_type = action.params.get("original_type", current.agent_type)
        alternatives = get_alternative_agent_classes(original_type)
        if not alternatives:
            return False

        target_type = action.params.get("new_type")
        new_cls = None
        if target_type:
            for cls in alternatives:
                inst = cls(provider=self._provider, config=self._config)
                if inst.agent_type.lower() == str(target_type).lower():
                    new_cls = cls
                    break
        if not new_cls:
            new_cls = alternatives[swap_count % len(alternatives)]

        agents[target_id] = new_cls(
            provider=self._provider, config=self._config
        )
        self._swap_counts[target_id] = swap_count + 1

        logger.info(
            "Swapped agent %s from %s → %s (swap #%d)",
            target_id,
            original_type,
            new_cls.__name__,
            swap_count + 1,
        )
        return True

    def _add_agent(
        self,
        action: AdaptationAction,
        agents: Dict[str, BaseAgent],
        plan: ExecutionPlan,
        remaining_wave_idx: int = 0,
    ) -> bool:
        """Add a secondary agent for a critical sub-task to remaining waves."""
        target_id = action.target_agent_id
        current = agents.get(target_id)
        if current is None or not plan or not plan.waves:
            return False

        new_id = f"{target_id}_aux"
        original_type = current.agent_type
        alternatives = get_alternative_agent_classes(original_type)
        new_cls = alternatives[0] if alternatives else current.__class__

        new_agent = new_cls(provider=self._provider, config=self._config)
        agents[new_id] = new_agent

        new_st = SubTask(
            id=new_id,
            label=f"{target_id}_aux",
            description=f"Supplementary agent task for {target_id}",
            critical=False,
        )

        target_wave_idx = min(remaining_wave_idx, len(plan.waves) - 1)
        plan.waves[target_wave_idx].append(new_st)
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
