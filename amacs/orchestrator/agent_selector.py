"""Agent selector — maps each sub-task to the best-fit agent type.

Also manages the global agent registry so users can register custom agents
via ``amacs.register_agent(name, agent_class)``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from amacs.agents.analysis_agent import AnalysisAgent
from amacs.agents.base_agent import BaseAgent, SubTask
from amacs.agents.search_agent import SearchAgent
from amacs.agents.validator_agent import ValidatorAgent
from amacs.agents.writer_agent import WriterAgent
from amacs.config import AMACSConfig
from amacs.integrations.llm_providers import LLMProvider

# ── Global registry ──────────────────────────────────────────────────────

_AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "search": SearchAgent,
    "analyze": AnalysisAgent,
    "analysis": AnalysisAgent,
    "write": WriterAgent,
    "writer": WriterAgent,
    "validate": ValidatorAgent,
    "validator": ValidatorAgent,
}


def register_agent(name: str, agent_class: Type[BaseAgent]) -> None:
    """Register a custom agent class under *name*."""
    _AGENT_REGISTRY[name.lower()] = agent_class


def get_registered_agents() -> Dict[str, Type[BaseAgent]]:
    """Return a copy of the current agent registry."""
    return dict(_AGENT_REGISTRY)


def get_alternative_agent_classes(current_type: str) -> List[Type[BaseAgent]]:
    """Return registered agent classes excluding the given current type."""
    alternatives: List[Type[BaseAgent]] = []
    seen: set[Type[BaseAgent]] = set()
    for name, cls in _AGENT_REGISTRY.items():
        if name.lower() != current_type.lower() and cls not in seen:
            seen.add(cls)
            alternatives.append(cls)
    return alternatives


# ── Selector ──────────────────────────────────────────────────────────────

class AgentSelector:
    """Picks the best-fit agent for each sub-task based on its label."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[AMACSConfig] = None,
    ) -> None:
        self._provider = provider
        self._config = config

    def select(self, sub_tasks: List[SubTask]) -> Dict[str, BaseAgent]:
        """Return a mapping of ``sub_task.id → agent instance``."""
        assignments: Dict[str, BaseAgent] = {}
        for st in sub_tasks:
            agent_cls = _AGENT_REGISTRY.get(st.label.lower())
            if agent_cls is None:
                # Fall back to the closest match or default to WriterAgent
                agent_cls = self._fuzzy_match(st.label) or WriterAgent
            assignments[st.id] = agent_cls(
                provider=self._provider,
                config=self._config,
            )
        return assignments

    @staticmethod
    def _fuzzy_match(label: str) -> Optional[Type[BaseAgent]]:
        """Simple substring match against registry keys."""
        label_lower = label.lower()
        for key, cls in _AGENT_REGISTRY.items():
            if key in label_lower or label_lower in key:
                return cls
        return None
