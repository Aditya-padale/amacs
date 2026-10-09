"""Result aggregator — collects, validates, and synthesises sub-task outputs.

Runs a final validation pass and merges all agent outputs into a single
coherent result that the ``@amacs`` decorator returns to the caller.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from amacs.agents.base_agent import AgentResult, SubTask
from amacs.agents.validator_agent import ValidatorAgent
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.exceptions import AggregationError
from amacs.integrations.llm_providers import LLMProvider, get_provider

logger = logging.getLogger("amacs.aggregation")


class ValidationReport(BaseModel):
    """Structured report returned by the validator agent."""

    verdict: Literal["pass", "revise"] = "pass"
    issues: List[str] = Field(default_factory=list)
    revised_text: Optional[str] = None


class Aggregator:
    """Collects sub-task results and produces a final, validated output."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[AMACSConfig] = None,
    ) -> None:
        self._provider = provider or get_provider()
        self._config = config
        self.raw_merge: str = ""
        self.validation_report: Optional[Dict[str, Any]] = None

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
        self.raw_merge = merged

        min_ratio = self._config.min_revision_ratio if self._config else 0.6
        val_report = ValidationReport(verdict="pass", issues=[], revised_text=None)

        # Check strategy rules before running optional validation pass
        skip_validation = False
        if self._config and self._config.strategy:
            from amacs.strategy import StrategyPolicy
            strat_val = self._config.strategy.value if hasattr(self._config.strategy, "value") else str(self._config.strategy)
            rules = StrategyPolicy.get_rules(strat_val)
            if not rules.enable_validation_pass or not rules.enable_critique_revision:
                skip_validation = True

        if not skip_validation:
            # Final validation pass (non-destructive)
            try:
                val_report = self._validate(merged, bus)
                if val_report.verdict == "revise" and val_report.revised_text:
                    rev_text = val_report.revised_text.strip()
                    if len(rev_text) >= len(merged) * min_ratio:
                        merged = rev_text
                    else:
                        logger.warning(
                            "Revised text length (%d) below min_revision_ratio (%f of %d). Keeping raw merge.",
                            len(rev_text),
                            min_ratio,
                            len(merged),
                        )
            except Exception as exc:
                logger.warning("Final validation failed (using unvalidated merge): %s", exc)

        self.validation_report = val_report.model_dump()
        return merged

    def _validate(self, content: str, bus: CommunicationBus) -> ValidationReport:
        """Run the Validator agent on the merged content non-destructively."""
        validator = ValidatorAgent(provider=self._provider, config=self._config)

        validation_task = SubTask(
            id="final_validation",
            label="validate",
            description=(
                "Review the following aggregated content for factual accuracy and quality. "
                "Respond in valid JSON format matching schema: "
                '{"verdict": "pass" | "revise", "issues": [...], "revised_text": str | null}'
            ),
            critical=False,
        )

        context = bus.snapshot()
        context["aggregated_content"] = content

        # Run validator without writing back to the communication bus
        result = validator.run(validation_task, context, bus=None)
        if result.success and result.content and result.content.strip():
            raw_json = result.content.strip()
            if raw_json.startswith("```"):
                lines = raw_json.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_json = "\n".join(lines).strip()
            try:
                data = json.loads(raw_json)
                return ValidationReport.model_validate(data)
            except Exception as parse_err:
                logger.warning("Failed to parse validator JSON response: %s", parse_err)

        return ValidationReport(verdict="pass", issues=["Parse failure or empty output"], revised_text=None)
