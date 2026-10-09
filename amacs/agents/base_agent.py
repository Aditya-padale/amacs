"""Abstract base agent — all AMACS agents inherit from this."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from amacs.communication import CommunicationBus
from amacs.config import AMACSConfig
from amacs.context_builder import ContextBuilder
from amacs.exceptions import (
    RETRYABLE_ERRORS,
    LLMProviderError,
    is_retryable_provider_error,
)
from amacs.integrations.llm_providers import LLMProvider, LLMResponse, Message, get_provider
from amacs.integrations.monitoring import MetricsRecorder
from amacs.tracing import Tracer

logger = logging.getLogger("amacs.agents.base_agent")


def _should_retry_exception(exc: BaseException) -> bool:
    """Predicate for retry logic to check if an exception is retryable."""
    if isinstance(exc, LLMProviderError):
        return is_retryable_provider_error(exc)
    return isinstance(exc, RETRYABLE_ERRORS)


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
        self._tracer: Optional[Tracer] = None

    def attach_tracer(self, tracer: Tracer) -> None:
        """Attach the execution tracer owned by the current pipeline run."""
        self._tracer = tracer

    # ── Subclass hooks ────────────────────────────────────────────────

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Short label such as ``'search'``, ``'analysis'``, etc."""

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt that defines this agent's role."""

    def build_user_prompt(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: Optional[CommunicationBus] = None,
    ) -> str:
        """Build the user-message from the sub-task and shared context.

        Includes only declared dependencies + original_input, token-budgeted.
        """
        parts = [f"Task: {sub_task.description}"]
        if context:
            max_tokens = self._config.max_context_tokens if self._config else None
            builder = ContextBuilder(max_tokens=max_tokens)
            allowed = set(sub_task.dependencies)
            allowed.add("original_input")
            ctx_str = builder.build_context_string(
                context, exclude_key=sub_task.id, allowed_keys=list(allowed)
            )
            if ctx_str:
                parts.append(ctx_str)
        return "\n\n".join(parts)

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
            result = self._run_with_retry(sub_task, context, retry_limit, bus=bus)
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
            result = await self._arun_with_retry(sub_task, context, retry_limit, bus=bus)
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

    def _call_llm(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: Optional[CommunicationBus] = None,
    ) -> AgentResult:
        user_prompt = self.build_user_prompt(sub_task, context, bus=bus)
        messages = [
            Message(role="system", content=self.system_prompt()),
            Message(role="user", content=user_prompt),
        ]
        from amacs.cache import ResponseCache
        from amacs.strategy import StrategyPolicy

        model = (
            StrategyPolicy.resolve_model(
                self._config.strategy.value if hasattr(self._config.strategy, "value") else str(self._config.strategy),
                self._provider.name(),
                self._config.llm_model,
                agent_type=self.agent_type,
                config=self._config,
            )
            if self._config
            else None
        )
        timeout = self._config.timeout if self._config else None
        if self._config and self._config.strategy:
            rules = StrategyPolicy.get_rules(
                self._config.strategy.value if hasattr(self._config.strategy, "value") else str(self._config.strategy)
            )
            if (timeout is None or timeout == 120.0) and rules.timeout_seconds is not None:
                timeout = rules.timeout_seconds

        # Check Cache
        cache_enabled = bool(self._config and self._config.cache)
        cache_path = self._config.cache if (self._config and isinstance(self._config.cache, str)) else None
        cache_inst = ResponseCache(enabled=cache_enabled, filepath=cache_path) if cache_enabled else None

        if cache_inst:
            ckey = ResponseCache.compute_key(messages, provider=self._provider.name(), model=model)
            cached_resp = cache_inst.get(ckey)
            if cached_resp:
                return AgentResult(
                    sub_task_id=sub_task.id,
                    agent_name=self.agent_type,
                    content=cached_resp.content,
                    success=True,
                    token_usage=cached_resp.usage,
                    metadata={"model": cached_resp.model, "cached": True},
                )

        def _do_call(msgs: list[Message], temp: float = 0.7) -> LLMResponse:
            span = self._tracer.start_span(
                f"{self.agent_type}:{sub_task.id}:llm",
                "llm_call",
                {"agent": self.agent_type, "task": sub_task.id, "model": model},
            ) if self._tracer else None
            try:
                if timeout is not None and timeout > 0:
                    import concurrent.futures

                    from amacs.exceptions import AgentTimeoutError

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        fut = executor.submit(self._provider.chat, msgs, model=model, temperature=temp, timeout=timeout)
                        try:
                            response = fut.result(timeout=timeout)
                        except concurrent.futures.TimeoutError:
                            raise AgentTimeoutError(
                                f"Agent '{self.agent_type}' for sub-task '{sub_task.id}' timed out after {timeout}s"
                            ) from None
                else:
                    response = self._provider.chat(msgs, model=model, temperature=temp, timeout=timeout)
            except Exception as exc:
                if span and self._tracer:
                    self._tracer.end_span(span, {"success": False, "error": str(exc)})
                raise
            if span and self._tracer:
                self._tracer.end_span(span, {"success": True, "total_tokens": response.usage.get("total_tokens", 0)})
            return response

        resp = _do_call(messages)
        if cache_inst:
            ckey = ResponseCache.compute_key(messages, provider=self._provider.name(), model=model)
            cache_inst.set(ckey, resp)

        content = resp.content
        token_usage = dict(resp.usage)

        # Candidate selection
        candidates_k = self._config.candidates_k if self._config else 1
        candidates_list = [content]
        if candidates_k > 1 and sub_task.critical:
            p_toks = token_usage.get("prompt_tokens", 0)
            c_toks = token_usage.get("completion_tokens", 0)
            for i in range(1, candidates_k):
                c_resp = _do_call(messages, temp=0.3 + 0.2 * i)
                candidates_list.append(c_resp.content)
                p_toks += c_resp.usage.get("prompt_tokens", 0)
                c_toks += c_resp.usage.get("completion_tokens", 0)
            token_usage = {"prompt_tokens": p_toks, "completion_tokens": c_toks, "total_tokens": p_toks + c_toks}

            # Select best candidate
            best_content = candidates_list[0]
            try:
                judge_prompt = (
                    f"Select the best candidate response for task: {sub_task.description}\n\n"
                    + "\n\n".join(f"Candidate {idx+1}:\n{c}" for idx, c in enumerate(candidates_list))
                    + "\n\nOutput only the integer candidate number."
                )
                judge_resp = self._provider.chat([Message(role="user", content=judge_prompt)], temperature=0.0)
                import re
                m = re.search(r"\b([1-9]\d*)\b", judge_resp.content)
                if m:
                    val = int(m.group(1)) - 1
                    if 0 <= val < len(candidates_list):
                        best_content = candidates_list[val]
            except Exception as exc:
                logger.debug("Candidate selection judge failed (%s). Using first candidate.", exc)
            content = best_content

        # Critic/reviser loop
        max_revs = self._config.max_revisions if self._config else 0
        threshold = self._config.critic_score_threshold if self._config else 0.8
        revisions = []
        if max_revs > 0 and self.agent_type in ("write", "analysis"):
            current_draft = content
            for rev in range(max_revs):
                critic_prompt = (
                    f"Evaluate output for task: {sub_task.description}\n\nDraft:\n{current_draft}\n\n"
                    "Output a score between 0.0 and 1.0 (format 'Score: X.XX') and feedback."
                )
                try:
                    c_resp = self._provider.chat([Message(role="user", content=critic_prompt)])
                    c_text = c_resp.content.strip()
                    import re
                    m = re.search(r"Score:\s*([0-1]?\.\d+|1\.0|0|\d+/10)", c_text, re.IGNORECASE)
                    score = 0.5
                    if m:
                        s_str = m.group(1)
                        score = float(s_str.split("/")[0]) / 10.0 if "/10" in s_str else float(s_str)
                    revisions.append({"revision": rev + 1, "score": score, "critic": c_text})
                    if bus:
                        bus.publish(f"{sub_task.id}_critic_{rev+1}", f"Score: {score:.2f} | {c_text}", writer="critic")
                    if score >= threshold:
                        break

                    w_prompt = f"Revise draft based on critic feedback:\n{c_text}\n\nDraft:\n{current_draft}"
                    r_resp = self._provider.chat([Message(role="user", content=w_prompt)])
                    rev_text = r_resp.content.strip()
                    if rev_text:
                        ratio = len(rev_text) / max(len(current_draft), 1)
                        if ratio >= (self._config.min_revision_ratio if self._config else 0.6):
                            current_draft = rev_text
                except Exception:
                    break
            content = current_draft

        meta: Dict[str, Any] = {"model": resp.model}
        if len(candidates_list) > 1:
            meta["candidates"] = candidates_list
        if revisions:
            meta["revisions"] = revisions

        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name=self.agent_type,
            content=content,
            success=True,
            token_usage=token_usage,
            metadata=meta,
        )

    async def _acall_llm(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: Optional[CommunicationBus] = None,
    ) -> AgentResult:
        user_prompt = self.build_user_prompt(sub_task, context, bus=bus)
        messages = [
            Message(role="system", content=self.system_prompt()),
            Message(role="user", content=user_prompt),
        ]
        from amacs.cache import ResponseCache
        from amacs.strategy import StrategyPolicy

        model = (
            StrategyPolicy.resolve_model(
                self._config.strategy.value if hasattr(self._config.strategy, "value") else str(self._config.strategy),
                self._provider.name(),
                self._config.llm_model,
                agent_type=self.agent_type,
                config=self._config,
            )
            if self._config
            else None
        )
        timeout = self._config.timeout if self._config else None
        if self._config and self._config.strategy:
            rules = StrategyPolicy.get_rules(
                self._config.strategy.value if hasattr(self._config.strategy, "value") else str(self._config.strategy)
            )
            if (timeout is None or timeout == 120.0) and rules.timeout_seconds is not None:
                timeout = rules.timeout_seconds

        # Check Cache
        cache_enabled = bool(self._config and self._config.cache)
        cache_path = self._config.cache if (self._config and isinstance(self._config.cache, str)) else None
        cache_inst = ResponseCache(enabled=cache_enabled, filepath=cache_path) if cache_enabled else None

        if cache_inst:
            ckey = ResponseCache.compute_key(messages, provider=self._provider.name(), model=model)
            cached_resp = cache_inst.get(ckey)
            if cached_resp:
                return AgentResult(
                    sub_task_id=sub_task.id,
                    agent_name=self.agent_type,
                    content=cached_resp.content,
                    success=True,
                    token_usage=cached_resp.usage,
                    metadata={"model": cached_resp.model, "cached": True},
                )

        async def _do_acall(msgs: list[Message], temp: float = 0.7) -> LLMResponse:
            span = self._tracer.start_span(
                f"{self.agent_type}:{sub_task.id}:llm",
                "llm_call",
                {"agent": self.agent_type, "task": sub_task.id, "model": model},
            ) if self._tracer else None
            try:
                if timeout is not None and timeout > 0:
                    import asyncio

                    from amacs.exceptions import AgentTimeoutError

                    try:
                        response = await asyncio.wait_for(
                            self._provider.achat(msgs, model=model, temperature=temp, timeout=timeout),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        raise AgentTimeoutError(
                            f"Agent '{self.agent_type}' for sub-task '{sub_task.id}' timed out after {timeout}s"
                        ) from None
                else:
                    response = await self._provider.achat(msgs, model=model, temperature=temp, timeout=timeout)
            except Exception as exc:
                if span and self._tracer:
                    self._tracer.end_span(span, {"success": False, "error": str(exc)})
                raise
            if span and self._tracer:
                self._tracer.end_span(span, {"success": True, "total_tokens": response.usage.get("total_tokens", 0)})
            return response

        resp = await _do_acall(messages)
        if cache_inst:
            ckey = ResponseCache.compute_key(messages, provider=self._provider.name(), model=model)
            cache_inst.set(ckey, resp)

        content = resp.content
        token_usage = dict(resp.usage)

        # Candidate selection
        candidates_k = self._config.candidates_k if self._config else 1
        candidates_list = [content]
        if candidates_k > 1 and sub_task.critical:
            p_toks = token_usage.get("prompt_tokens", 0)
            c_toks = token_usage.get("completion_tokens", 0)
            for i in range(1, candidates_k):
                c_resp = await _do_acall(messages, temp=0.3 + 0.2 * i)
                candidates_list.append(c_resp.content)
                p_toks += c_resp.usage.get("prompt_tokens", 0)
                c_toks += c_resp.usage.get("completion_tokens", 0)
            token_usage = {"prompt_tokens": p_toks, "completion_tokens": c_toks, "total_tokens": p_toks + c_toks}

            best_content = candidates_list[0]
            try:
                judge_prompt = (
                    f"Select the best candidate response for task: {sub_task.description}\n\n"
                    + "\n\n".join(f"Candidate {idx+1}:\n{c}" for idx, c in enumerate(candidates_list))
                    + "\n\nOutput only the integer candidate number."
                )
                judge_resp = await self._provider.achat([Message(role="user", content=judge_prompt)], temperature=0.0)
                import re
                m = re.search(r"\b([1-9]\d*)\b", judge_resp.content)
                if m:
                    val = int(m.group(1)) - 1
                    if 0 <= val < len(candidates_list):
                        best_content = candidates_list[val]
            except Exception as exc:
                logger.debug("Async candidate selection judge failed (%s). Using first candidate.", exc)
            content = best_content

        # Critic/reviser loop
        max_revs = self._config.max_revisions if self._config else 0
        threshold = self._config.critic_score_threshold if self._config else 0.8
        revisions = []
        if max_revs > 0 and self.agent_type in ("write", "analysis"):
            current_draft = content
            for rev in range(max_revs):
                critic_prompt = (
                    f"Evaluate output for task: {sub_task.description}\n\nDraft:\n{current_draft}\n\n"
                    "Output a score between 0.0 and 1.0 (format 'Score: X.XX') and feedback."
                )
                try:
                    c_resp = await self._provider.achat([Message(role="user", content=critic_prompt)])
                    c_text = c_resp.content.strip()
                    import re
                    m = re.search(r"Score:\s*([0-1]?\.\d+|1\.0|0|\d+/10)", c_text, re.IGNORECASE)
                    score = 0.5
                    if m:
                        s_str = m.group(1)
                        score = float(s_str.split("/")[0]) / 10.0 if "/10" in s_str else float(s_str)
                    revisions.append({"revision": rev + 1, "score": score, "critic": c_text})
                    if bus:
                        bus.publish(f"{sub_task.id}_critic_{rev+1}", f"Score: {score:.2f} | {c_text}", writer="critic")
                    if score >= threshold:
                        break

                    w_prompt = f"Revise draft based on critic feedback:\n{c_text}\n\nDraft:\n{current_draft}"
                    r_resp = await self._provider.achat([Message(role="user", content=w_prompt)])
                    rev_text = r_resp.content.strip()
                    if rev_text:
                        ratio = len(rev_text) / max(len(current_draft), 1)
                        if ratio >= (self._config.min_revision_ratio if self._config else 0.6):
                            current_draft = rev_text
                except Exception:
                    break
            content = current_draft

        meta: Dict[str, Any] = {"model": resp.model}
        if len(candidates_list) > 1:
            meta["candidates"] = candidates_list
        if revisions:
            meta["revisions"] = revisions

        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name=self.agent_type,
            content=content,
            success=True,
            token_usage=token_usage,
            metadata=meta,
        )

    def _run_with_retry(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        max_attempts: int,
        bus: Optional[CommunicationBus] = None,
    ) -> AgentResult:
        from amacs.exceptions import run_with_retry
        return run_with_retry(
            lambda: self._call_llm(sub_task, context, bus=bus),
            max_attempts=max_attempts,
        )

    async def _arun_with_retry(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        max_attempts: int,
        bus: Optional[CommunicationBus] = None,
    ) -> AgentResult:
        from amacs.exceptions import arun_with_retry
        return await arun_with_retry(
            lambda: self._acall_llm(sub_task, context, bus=bus),
            max_attempts=max_attempts,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.agent_type!r})"
