"""Result aggregator — collects, validates, and synthesises sub-task outputs.

Runs a final validation pass and merges all agent outputs into a single
coherent result that the ``@amacs`` decorator returns to the caller.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from amacs.agents.base_agent import AgentResult, BaseAgent, SubTask
from amacs.agents.validator_agent import ValidatorAgent
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.exceptions import AggregationError
from amacs.integrations.llm_providers import LLMProvider, get_provider

logger = logging.getLogger("amacs.aggregation")


class Aggregator:
    """Collects sub-task results and produces a final, validated output."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[AMACSConfig] = None,
    ) -> None:
        self._provider = provider or get_provider()
        self._config = config

    def aggregate(
        self,
        results: List[AgentResult],
        sub_tasks: List[SubTask],
        bus: CommunicationBus,
    ) -> str:
        """Merge results into a single coherent string.

        Steps:
        1. Filter to successful results.
        2. Order by the sub-task list (preserves logical flow).
        3. Run a Validator pass for final consistency.
        4. Merge into a single output.
        """
        task_map = {st.id: st for st in sub_tasks}
        success_map: Dict[str, AgentResult] = {}
        for r in results:
            if r.success and r.content:
                success_map[r.sub_task_id] = r

        if not success_map:
            raise AggregationError("No successful sub-task results to aggregate.")

        # Order by original sub-task list
        ordered_contents: List[str] = []
        for st in sub_tasks:
            if st.id in success_map:
                ordered_contents.append(success_map[st.id].content)

        merged = "\n\n".join(ordered_contents)

        # Final validation pass
        try:
            merged = self._validate(merged, bus)
        except Exception as exc:
            logger.warning("Final validation failed (using unvalidated merge): %s", exc)

        return merged

    def _validate(self, content: str, bus: CommunicationBus) -> str:
        """Run the Validator agent on the merged content."""
        validator = ValidatorAgent(provider=self._provider, config=self._config)

        validation_task = SubTask(
            id="final_validation",
            label="validate",
            description=(
                "Review the following aggregated content for factual accuracy, "
                "internal consistency, and overall quality. If issues are found, "
                "produce a corrected version. If the content is sound, return it "
                "as-is with a brief quality confirmation at the end."
            ),
            critical=False,
        )

        context = bus.snapshot()
        context["aggregated_content"] = content

        result = validator.run(validation_task, context, bus)
        if result.success and result.content:
            return result.content
        return content
