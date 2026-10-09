"""Coordinator base abstraction for multi-agent interaction protocols."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from amacs.agents.base_agent import AgentResult, BaseAgent, SubTask
from amacs.cache import ResponseCache
from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.integrations.llm_providers import LLMProvider, get_provider
from amacs.tracing import Tracer


class Coordinator(ABC):
    """Abstract base class for multi-agent coordination protocols."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[AMACSConfig] = None,
    ) -> None:
        self._provider = provider or get_provider()
        self._config = config
        self._tracer: Optional[Tracer] = None
        self._cache: Optional[ResponseCache] = None

    def attach_runtime(
        self, tracer: Optional[Tracer] = None, cache: Optional[ResponseCache] = None
    ) -> None:
        """Share pipeline-owned observability and caching with nested agents."""
        self._tracer = tracer
        self._cache = cache

    def prepare_agent(self, agent: BaseAgent) -> BaseAgent:
        if self._tracer is not None:
            agent.attach_tracer(self._tracer)
        if self._cache is not None:
            agent.attach_cache(self._cache)
        return agent

    @abstractmethod
    def coordinate(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: CommunicationBus,
    ) -> AgentResult:
        """Execute the coordination protocol for the given sub-task."""

    async def acoordinate(
        self, sub_task: SubTask, context: Dict[str, Any], bus: CommunicationBus
    ) -> AgentResult:
        """Async protocol hook; subclasses should avoid blocking the event loop."""
        return await asyncio.to_thread(self.coordinate, sub_task, context, bus)
