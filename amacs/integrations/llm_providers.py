"""LLM provider abstraction — uniform interface for OpenAI, Anthropic, Gemini, and Ollama.

The framework never imports an SDK directly; it goes through :class:`LLMProvider`.
Concrete providers are loaded lazily so missing optional deps don't crash the import.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from amacs.exceptions import LLMProviderError

# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class Message:
    """A single chat-completion message."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    """Normalised response from any provider."""

    content: str
    model: str
    usage: Dict[str, int]  # {"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}
    raw: Any = None  # provider-specific response object
    tool_calls: Optional[List[Dict[str, Any]]] = None  # normalized {id, name, arguments}
    structured: Any = None  # decoded structured output when the provider returns JSON


def _value(source: Any, key: str, default: Any = None) -> Any:
    """Read a field from either an SDK object or a mapping."""
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _normalize_tool_calls(calls: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize SDK-specific tool call objects into the public AMACS shape."""
    if not calls:
        return None
    normalized: List[Dict[str, Any]] = []
    for index, call in enumerate(calls):
        function = _value(call, "function", call)
        name = _value(function, "name", _value(call, "name", ""))
        arguments = _value(function, "arguments", _value(call, "arguments", {}))
        if not isinstance(arguments, (str, dict, list)):
            arguments = str(arguments)
        normalized.append({"id": _value(call, "id", f"tool_call_{index}"), "name": name, "arguments": arguments})
    return normalized


def _structured_value(content: str) -> Any:
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return None


def _schema_parameters(kwargs: Dict[str, Any], provider: str) -> Dict[str, Any]:
    """Translate the common ``output_schema`` option to provider parameters."""
    schema = kwargs.pop("output_schema", None)
    if schema is None:
        return kwargs
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    elif not isinstance(schema, dict):
        schema = getattr(schema, "schema", dict)()
    if provider == "openai":
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "amacs_output", "strict": True, "schema": schema}}
    elif provider == "ollama":
        kwargs["format"] = schema
    elif provider == "gemini":
        kwargs["response_schema"] = schema
        kwargs["response_mime_type"] = "application/json"
    else:
        kwargs["output_schema"] = schema
    return kwargs


def _usage(source: Any) -> Dict[str, int]:
    usage = _value(source, "usage", {}) or {}
    prompt = _value(usage, "prompt_tokens", _value(usage, "input_tokens", 0)) or 0
    completion = _value(usage, "completion_tokens", _value(usage, "output_tokens", 0)) or 0
    total = _value(usage, "total_tokens", prompt + completion) or prompt + completion
    return {"prompt_tokens": int(prompt), "completion_tokens": int(completion), "total_tokens": int(total)}


def _response(content: Any, model: str, usage: Dict[str, int], raw: Any = None, tool_calls: Any = None) -> LLMResponse:
    text = content if isinstance(content, str) else json.dumps(content) if content is not None else ""
    return LLMResponse(text, model, usage, raw=raw, tool_calls=_normalize_tool_calls(tool_calls), structured=_structured_value(text))


# ── Abstract base ─────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous chat completion."""

    @abstractmethod
    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async chat completion."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""


# ── OpenAI ────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """Provider backed by the ``openai`` SDK."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: Optional[str] = None, **client_kwargs: Any) -> None:
        try:
            import openai
        except ImportError:
            raise LLMProviderError(
                "openai package not installed. Run: pip install amacs[openai]"
            ) from None
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = openai.OpenAI(api_key=self._api_key, **client_kwargs)
        self._async_client = openai.AsyncOpenAI(api_key=self._api_key, **client_kwargs)

    def name(self) -> str:
        return "openai"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        model_name = model or self.DEFAULT_MODEL
        params: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if timeout is not None:
            params["timeout"] = timeout
        params.update(_schema_parameters(kwargs, "openai"))
        try:
            resp: Any = self._client.chat.completions.create(**params)
            message = resp.choices[0].message
            return _response(_value(message, "content", ""), _value(resp, "model", model_name), _usage(resp), resp, _value(message, "tool_calls"))
        except Exception as exc:
            raise LLMProviderError(f"OpenAI call failed: {exc}") from exc

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        model_name = model or self.DEFAULT_MODEL
        params: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if timeout is not None:
            params["timeout"] = timeout
        params.update(_schema_parameters(kwargs, "openai"))
        try:
            resp: Any = await self._async_client.chat.completions.create(**params)
            message = resp.choices[0].message
            return _response(_value(message, "content", ""), _value(resp, "model", model_name), _usage(resp), resp, _value(message, "tool_calls"))
        except Exception as exc:
            raise LLMProviderError(f"OpenAI async call failed: {exc}") from exc


# ── Anthropic ─────────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """Provider backed by the ``anthropic`` SDK."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: Optional[str] = None, **client_kwargs: Any) -> None:
        try:
            import anthropic
        except ImportError:
            raise LLMProviderError(
                "anthropic package not installed. Run: pip install amacs[anthropic]"
            ) from None
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=self._api_key, **client_kwargs)
        self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key, **client_kwargs)

    def name(self) -> str:
        return "anthropic"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        model_name = model or self.DEFAULT_MODEL
        system_msg: Optional[str] = None
        user_msgs: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                user_msgs.append({"role": m.role, "content": m.content})

        params: Dict[str, Any] = {
            "model": model_name,
            "messages": user_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            params["system"] = system_msg
        if timeout is not None:
            params["timeout"] = timeout
        params.update(_schema_parameters(kwargs, "anthropic"))

        try:
            resp: Any = self._client.messages.create(**params)
            blocks = _value(resp, "content", []) or []
            text = "".join(_value(block, "text", "") for block in blocks if _value(block, "type", "text") != "tool_use")
            tools = [block for block in blocks if _value(block, "type", "") == "tool_use"]
            return _response(text, _value(resp, "model", model_name), _usage({"usage": {"input_tokens": _value(_value(resp, "usage", {}), "input_tokens", 0), "output_tokens": _value(_value(resp, "usage", {}), "output_tokens", 0)}}), resp, tools)
        except Exception as exc:
            raise LLMProviderError(f"Anthropic call failed: {exc}") from exc

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        model_name = model or self.DEFAULT_MODEL
        system_msg: Optional[str] = None
        user_msgs: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                user_msgs.append({"role": m.role, "content": m.content})

        params: Dict[str, Any] = {
            "model": model_name,
            "messages": user_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            params["system"] = system_msg
        if timeout is not None:
            params["timeout"] = timeout
        params.update(_schema_parameters(kwargs, "anthropic"))

        try:
            resp: Any = await self._async_client.messages.create(**params)
            blocks = _value(resp, "content", []) or []
            text = "".join(_value(block, "text", "") for block in blocks if _value(block, "type", "text") != "tool_use")
            tools = [block for block in blocks if _value(block, "type", "") == "tool_use"]
            return _response(text, _value(resp, "model", model_name), _usage({"usage": {"input_tokens": _value(_value(resp, "usage", {}), "input_tokens", 0), "output_tokens": _value(_value(resp, "usage", {}), "output_tokens", 0)}}), resp, tools)
        except Exception as exc:
            raise LLMProviderError(f"Anthropic async call failed: {exc}") from exc


# ── Ollama (local LLM) ───────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """Provider backed by a local Ollama instance."""

    DEFAULT_MODEL = "llama3"

    def __init__(self, host: Optional[str] = None, **client_kwargs: Any) -> None:
        try:
            import ollama as _ollama
        except ImportError:
            raise LLMProviderError(
                "ollama package not installed. Run: pip install amacs[ollama]"
            ) from None
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._ollama = _ollama
        self._client = _ollama.Client(host=self._host, **client_kwargs)
        self._async_client = _ollama.AsyncClient(host=self._host, **client_kwargs)

    def name(self) -> str:
        return "ollama"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        model_name = model or self.DEFAULT_MODEL
        try:
            resp = self._client.chat(
                model=model_name,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                options={"temperature": temperature, "num_predict": max_tokens},
                **_schema_parameters(kwargs, "ollama"),
            )
            usage = {"prompt_tokens": resp.get("prompt_eval_count", 0), "completion_tokens": resp.get("eval_count", 0), "total_tokens": resp.get("prompt_eval_count", 0) + resp.get("eval_count", 0)}
            message = resp.get("message", {})
            return _response(message.get("content", ""), model_name, usage, resp, message.get("tool_calls"))
        except Exception as exc:
            raise LLMProviderError(f"Ollama call failed: {exc}") from exc

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        model_name = model or self.DEFAULT_MODEL
        try:
            resp = await self._async_client.chat(
                model=model_name,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                options={"temperature": temperature, "num_predict": max_tokens},
                **_schema_parameters(kwargs, "ollama"),
            )
            usage = {"prompt_tokens": resp.get("prompt_eval_count", 0), "completion_tokens": resp.get("eval_count", 0), "total_tokens": resp.get("prompt_eval_count", 0) + resp.get("eval_count", 0)}
            message = resp.get("message", {})
            return _response(message.get("content", ""), model_name, usage, resp, message.get("tool_calls"))
        except Exception as exc:
            raise LLMProviderError(f"Ollama async call failed: {exc}") from exc


# ── Gemini (Google) ───────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """Provider backed by the ``google-genai`` SDK (Gemini API)."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None, **client_kwargs: Any) -> None:
        try:
            from google import genai
        except ImportError:
            raise LLMProviderError(
                "google-genai package not installed. Run: pip install amacs[gemini]"
            ) from None
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        self._genai = genai
        self._client = genai.Client(api_key=self._api_key, **client_kwargs)

    def name(self) -> str:
        return "gemini"

    def _build_contents(self, messages: Sequence[Message]) -> tuple[Optional[str], List[Any]]:
        """Split messages into a system instruction and contents list."""
        system_instruction: Optional[str] = None
        contents: List[Any] = []
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                role = "model" if m.role == "assistant" else m.role
                contents.append({"role": role, "parts": [{"text": m.content}]})
        return system_instruction, contents

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        model_name = model or self.DEFAULT_MODEL
        system_instruction, contents = self._build_contents(messages)
        try:
            config = self._genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
                **_schema_parameters(kwargs, "gemini"),
            )
            resp = self._client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            usage_meta = getattr(resp, "usage_metadata", None)
            prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
            calls = []
            for candidate in getattr(resp, "candidates", []) or []:
                calls.extend([part.function_call for part in getattr(candidate.content, "parts", []) if getattr(part, "function_call", None)])
            return _response(resp.text or "", model_name, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}, resp, calls)
        except Exception as exc:
            raise LLMProviderError(f"Gemini call failed: {exc}") from exc

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        model_name = model or self.DEFAULT_MODEL
        system_instruction, contents = self._build_contents(messages)
        try:
            config = self._genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
                **_schema_parameters(kwargs, "gemini"),
            )
            resp = await self._client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            usage_meta = getattr(resp, "usage_metadata", None)
            prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
            calls = []
            for candidate in getattr(resp, "candidates", []) or []:
                calls.extend([part.function_call for part in getattr(candidate.content, "parts", []) if getattr(part, "function_call", None)])
            return _response(resp.text or "", model_name, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}, resp, calls)
        except Exception as exc:
            raise LLMProviderError(f"Gemini async call failed: {exc}") from exc


# ── Stub / Mock provider (used when no LLM is configured) ────────────────

class StubProvider(LLMProvider):
    """Deterministic stub for testing and offline use."""

    def name(self) -> str:
        return "stub"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_user = next(
            (m.content for m in reversed(list(messages)) if m.role == "user"), ""
        )
        return LLMResponse(
            content=f"[stub] Processed: {last_user}",
            model="stub",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)


# ── Scripted & Flaky Providers for Behavioral Testing ────────────────────

class FakeProvider(LLMProvider):
    """Scripted provider for behavioral testing.

    Returns programmed responses per call, records every call, and can raise exceptions
    on specific call numbers.
    """

    def __init__(
        self,
        responses: Optional[Sequence[Any]] = None,
        default_response: str = "[fake] default response",
        raise_on_call: Optional[Dict[int, Exception]] = None,
        provider_name: str = "stub",
    ) -> None:
        self._responses = list(responses) if responses is not None else []
        self._default_response = default_response
        self._raise_on_call = raise_on_call or {}
        self._provider_name = provider_name
        self.calls: List[Dict[str, Any]] = []

    def name(self) -> str:
        return self._provider_name

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        call_idx = len(self.calls) + 1
        call_record = {
            "index": call_idx,
            "messages": list(messages),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "kwargs": kwargs,
        }
        self.calls.append(call_record)

        if call_idx in self._raise_on_call:
            raise self._raise_on_call[call_idx]

        if self._responses:
            next_resp = self._responses.pop(0)
            if isinstance(next_resp, Exception):
                raise next_resp
            if isinstance(next_resp, LLMResponse):
                return next_resp
            return LLMResponse(
                content=str(next_resp),
                model=model or self._provider_name,
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            )

        last_user = next(
            (m.content for m in reversed(list(messages)) if m.role == "user"), ""
        )
        content = (
            self._default_response.format(user=last_user, idx=call_idx)
            if ("{user}" in self._default_response or "{idx}" in self._default_response)
            else self._default_response
        )
        return LLMResponse(
            content=content,
            model=model or self._provider_name,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )


class FlakyProvider(LLMProvider):
    """Wrapper provider simulating latency, random failures, and empty outputs."""

    def __init__(
        self,
        base_provider: Optional[LLMProvider] = None,
        failure_rate: float = 0.0,
        latency_seconds: float = 0.0,
        empty_output_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        import random
        self._base = base_provider or StubProvider()
        self._failure_rate = failure_rate
        self._latency_seconds = latency_seconds
        self._empty_output_rate = empty_output_rate
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self.calls = 0
        self.failures = 0

    def name(self) -> str:
        return self._base.name()

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        import time
        self.calls += 1
        if self._latency_seconds > 0:
            time.sleep(self._latency_seconds)

        if self._failure_rate > 0 and self._rng.random() < self._failure_rate:
            self.failures += 1
            raise LLMProviderError("FlakyProvider simulated 503 service unavailable error")

        if self._empty_output_rate > 0 and self._rng.random() < self._empty_output_rate:
            return LLMResponse(
                content="",
                model=model or self.name(),
                usage={"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
            )

        return self._base.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        import asyncio
        self.calls += 1
        if self._latency_seconds > 0:
            await asyncio.sleep(self._latency_seconds)

        if self._failure_rate > 0 and self._rng.random() < self._failure_rate:
            self.failures += 1
            raise LLMProviderError("FlakyProvider simulated 503 service unavailable error")

        if self._empty_output_rate > 0 and self._rng.random() < self._empty_output_rate:
            return LLMResponse(
                content="",
                model=model or self.name(),
                usage={"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
            )

        return await self._base.achat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )


# ── Factory ───────────────────────────────────────────────────────────────

_PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "stub": StubProvider,
    "fake": FakeProvider,
    "flaky": FlakyProvider,
}


def get_provider(
    name: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """Resolve and instantiate an LLM provider by name."""
    raw_name: str = name or os.getenv("AMACS_LLM_PROVIDER") or "stub"
    provider_name = raw_name.lower()
    cls = _PROVIDERS.get(provider_name)
    if cls is None:
        raise LLMProviderError(
            f"Unknown LLM provider '{provider_name}'. Available: {list(_PROVIDERS.keys())}"
        )
    return cls(**kwargs)


def register_provider(name: str, provider_class: Type[LLMProvider]) -> None:
    """Register a custom LLM provider class."""
    _PROVIDERS[name.lower()] = provider_class
