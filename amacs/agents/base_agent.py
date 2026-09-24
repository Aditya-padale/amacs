"""Abstract base agent — all AMACS agents inherit from this."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.exceptions import AgentError, LLMProviderError, RetryExhaustedError
from amacs.integrations.llm_providers import LLMProvider, LLMResponse, Message, get_provider
from amacs.integrations.monitoring import MetricsRecorder


@dataclass
class SubTask:
    """A single sub-task produced by the :class:`TaskDecomposer`."""

    id: str
    label: str  # e.g. "search", "analyze", "write", "validate"
    description: str
    dependencies: list[str] = field(default_factory=list)
    critical: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result returned by an agent's ``run`` method."""

    sub_task_id: str
    agent_name: str
    content: str
    success: bool = True
    error: Optional[str] = None
    latency_seconds: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract agent with LLM integration, retry logic, and metrics.

    Subclasses must implement :meth:`system_prompt` and optionally override
    :meth:`build_user_prompt` to customise per-agent behaviour.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[AMACSConfig] = None,
    ) -> None:
        self._provider = provider or get_provider()
        self._config = config
        self._metrics = MetricsRecorder()

    # ── Subclass hooks ────────────────────────────────────────────────

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Short label such as ``'search'``, ``'analysis'``, etc."""

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt that defines this agent's role."""

    def build_user_prompt(self, sub_task: SubTask, context: Dict[str, Any]) -> str:
        """Build the user-message from the sub-task and shared context.

        Default implementation includes the sub-task description and any
        prior agent outputs found in *context*.
        """
        parts = [f"Task: {sub_task.description}"]
        if context:
            prior = "\n".join(f"- {k}: {v}" for k, v in context.items() if k != sub_task.id)
            if prior:
                parts.append(f"\nPrior context:\n{prior}")
        return "\n".join(parts)

    # ── Public API ────────────────────────────────────────────────────

    def run(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: Optional[CommunicationBus] = None,
    ) -> AgentResult:
        """Execute the sub-task synchronously with retries."""
        retry_limit = self._config.retry_limit if self._config else 3
        start = time.time()
        try:
            result = self._run_with_retry(sub_task, context, retry_limit)
            latency = time.time() - start
            self._metrics.record_call(self.agent_type, "success")
            self._metrics.observe_latency(self.agent_type, latency)
            if result.token_usage:
                self._metrics.record_tokens(
                    self.agent_type,
                    result.token_usage.get("prompt_tokens", 0),
                    result.token_usage.get("completion_tokens", 0),
                )
            result.latency_seconds = latency
            # publish to bus
            if bus is not None:
                bus.publish(sub_task.id, result.content, writer=self.agent_type)
            return result
        except Exception as exc:
            latency = time.time() - start
            self._metrics.record_call(self.agent_type, "failure")
            self._metrics.observe_latency(self.agent_type, latency)
            return AgentResult(
                sub_task_id=sub_task.id,
                agent_name=self.agent_type,
                content="",
                success=False,
                error=str(exc),
                latency_seconds=latency,
            )

    async def arun(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: Optional[CommunicationBus] = None,
    ) -> AgentResult:
        """Execute the sub-task asynchronously with retries."""
        retry_limit = self._config.retry_limit if self._config else 3
        start = time.time()
        try:
            result = await self._arun_with_retry(sub_task, context, retry_limit)
            latency = time.time() - start
            self._metrics.record_call(self.agent_type, "success")
            self._metrics.observe_latency(self.agent_type, latency)
            if result.token_usage:
                self._metrics.record_tokens(
                    self.agent_type,
                    result.token_usage.get("prompt_tokens", 0),
                    result.token_usage.get("completion_tokens", 0),
                )
            result.latency_seconds = latency
            if bus is not None:
                bus.publish(sub_task.id, result.content, writer=self.agent_type)
            return result
        except Exception as exc:
            latency = time.time() - start
            self._metrics.record_call(self.agent_type, "failure")
            self._metrics.observe_latency(self.agent_type, latency)
            return AgentResult(
                sub_task_id=sub_task.id,
                agent_name=self.agent_type,
                content="",
                success=False,
                error=str(exc),
                latency_seconds=latency,
            )

    # ── Internal ──────────────────────────────────────────────────────

    def _call_llm(self, sub_task: SubTask, context: Dict[str, Any]) -> AgentResult:
        messages = [
            Message(role="system", content=self.system_prompt()),
            Message(role="user", content=self.build_user_prompt(sub_task, context)),
        ]
        model = self._config.llm_model if self._config else None
        resp: LLMResponse = self._provider.chat(messages, model=model)
        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name=self.agent_type,
            content=resp.content,
            success=True,
            token_usage=resp.usage,
        )

    async def _acall_llm(self, sub_task: SubTask, context: Dict[str, Any]) -> AgentResult:
        messages = [
            Message(role="system", content=self.system_prompt()),
            Message(role="user", content=self.build_user_prompt(sub_task, context)),
        ]
        model = self._config.llm_model if self._config else None
        resp: LLMResponse = await self._provider.achat(messages, model=model)
        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name=self.agent_type,
            content=resp.content,
            success=True,
            token_usage=resp.usage,
        )

    def _run_with_retry(
        self, sub_task: SubTask, context: Dict[str, Any], max_attempts: int
    ) -> AgentResult:
        @retry(
            retry=retry_if_exception_type((LLMProviderError, AgentError)),
            stop=stop_after_attempt(max(max_attempts, 1)),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
            reraise=True,
        )
        def _inner() -> AgentResult:
            return self._call_llm(sub_task, context)

        try:
            return _inner()
        except Exception as exc:
            raise RetryExhaustedError(
                f"Agent '{self.agent_type}' exhausted {max_attempts} retries: {exc}"
            ) from exc

    async def _arun_with_retry(
        self, sub_task: SubTask, context: Dict[str, Any], max_attempts: int
    ) -> AgentResult:
        # tenacity's @retry works with sync only; manual async retry loop
        last_exc: Optional[Exception] = None
        for attempt in range(max(max_attempts, 1)):
            try:
                return await self._acall_llm(sub_task, context)
            except (LLMProviderError, AgentError) as exc:
                last_exc = exc
                import asyncio

                await asyncio.sleep(min(0.5 * (2**attempt), 10))
        raise RetryExhaustedError(
            f"Agent '{self.agent_type}' exhausted {max_attempts} async retries: {last_exc}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.agent_type!r})"
