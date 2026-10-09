"""Coordinator base abstraction for multi-agent interaction protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from amacs.agents.base_agent import AgentResult, SubTask
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.integrations.llm_providers import LLMProvider, get_provider


class Coordinator(ABC):
    """Abstract base class for multi-agent coordination protocols."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[AMACSConfig] = None,
    ) -> None:
        self._provider = provider or get_provider()
        self._config = config

    @abstractmethod
    def coordinate(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: CommunicationBus,
    ) -> AgentResult:
        """Execute the coordination protocol for the given sub-task."""
