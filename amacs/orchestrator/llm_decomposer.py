"""LLM-driven task decomposer — uses an LLM to dynamically generate task DAGs.

Produces structured SubTask instances from unstructured or complex prompts.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from amacs.agents.base_agent import SubTask
from amacs.config import AMACSConfig
from amacs.integrations.llm_providers import LLMProvider, Message, get_provider
from amacs.orchestrator.task_analyzer import TaskProfile

logger = logging.getLogger("amacs.orchestrator.llm_decomposer")


from pydantic import BaseModel, Field


class LLMSubTaskSchema(BaseModel):
    id: str
    label: str = Field(default="general")
    description: str = Field(default="")
    dependencies: List[str] = Field(default_factory=list)
    critical: bool = Field(default=True)


class LLMTaskDecomposer:
    """Uses LLM structured generation to decompose complex prompts into sub-tasks."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[AMACSConfig] = None,
    ) -> None:
        self._provider = provider or get_provider()
        self._config = config

    def decompose(self, profile: TaskProfile) -> List[SubTask]:
        """Decompose profile using LLM structured JSON response."""
        prompt = (
            f"Analyze the following task and decompose it into {profile.estimated_sub_tasks} sub-tasks.\n"
            f"Task Domain: {profile.domain}\n"
            f"Context: {profile.arg_summary}\n\n"
            "Return ONLY a JSON array of objects with keys:\n"
            '- "id": string (e.g. "search_0", "analyze_1")\n'
            '- "label": string (one of "search", "analyze", "write", "validate")\n'
            '- "description": string\n'
            '- "dependencies": list of string IDs\n'
            '- "critical": boolean\n'
        )

        messages = [
            Message(role="system", content="You are an expert AI task planner. Output valid JSON only."),
            Message(role="user", content=prompt),
        ]

        try:
            resp = self._provider.chat(messages, max_tokens=1024)
            cleaned = resp.content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            items = json.loads(cleaned)
            if not isinstance(items, list):
                raise TypeError("Expected JSON array of sub-tasks")

            sub_tasks: List[SubTask] = []
            for item in items:
                validated = LLMSubTaskSchema.model_validate(item)
                sub_tasks.append(
                    SubTask(
                        id=validated.id,
                        label=validated.label,
                        description=validated.description,
                        dependencies=validated.dependencies,
                        critical=validated.critical,
                    )
                )

            if sub_tasks:
                from amacs.orchestrator.scheduler import Scheduler
                Scheduler(config=self._config).plan(sub_tasks)
                return sub_tasks
        except Exception as exc:
            logger.warning("LLM task decomposition failed (%s). Falling back to rule-based decomposer.", exc)

        # Fallback to rule-based TaskDecomposer
        from amacs.orchestrator.task_decomposer import TaskDecomposer
        return TaskDecomposer().decompose(profile)
