"""LLM provider abstraction — uniform interface for OpenAI, Anthropic, Gemini, and Ollama.

The framework never imports an SDK directly; it goes through :class:`LLMProvider`.
Concrete providers are loaded lazily so missing optional deps don't crash the import.
"""

from __future__ import annotations

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
        params.update(kwargs)
        try:
            resp: Any = self._client.chat.completions.create(**params)
            usage = resp.usage
            return LLMResponse(
                content=resp.choices[0].message.content or "",
                model=resp.model,
                usage={
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
                raw=resp,
            )
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
        params.update(kwargs)
        try:
            resp: Any = await self._async_client.chat.completions.create(**params)
            usage = resp.usage
            return LLMResponse(
                content=resp.choices[0].message.content or "",
                model=resp.model,
                usage={
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
                raw=resp,
            )
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
        params.update(kwargs)

        try:
            resp: Any = self._client.messages.create(**params)
            return LLMResponse(
                content=resp.content[0].text if resp.content else "",
                model=resp.model,
                usage={
                    "prompt_tokens": resp.usage.input_tokens,
                    "completion_tokens": resp.usage.output_tokens,
                    "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
                },
                raw=resp,
            )
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
        params.update(kwargs)

        try:
            resp: Any = await self._async_client.messages.create(**params)
            return LLMResponse(
                content=resp.content[0].text if resp.content else "",
                model=resp.model,
                usage={
                    "prompt_tokens": resp.usage.input_tokens,
                    "completion_tokens": resp.usage.output_tokens,
                    "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
                },
                raw=resp,
            )
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
                **kwargs,
            )
            return LLMResponse(
                content=resp.get("message", {}).get("content", ""),
                model=model_name,
                usage={
                    "prompt_tokens": resp.get("prompt_eval_count", 0),
                    "completion_tokens": resp.get("eval_count", 0),
                    "total_tokens": resp.get("prompt_eval_count", 0)
                    + resp.get("eval_count", 0),
                },
                raw=resp,
            )
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
                **kwargs,
            )
            return LLMResponse(
                content=resp.get("message", {}).get("content", ""),
                model=model_name,
                usage={
                    "prompt_tokens": resp.get("prompt_eval_count", 0),
                    "completion_tokens": resp.get("eval_count", 0),
                    "total_tokens": resp.get("prompt_eval_count", 0)
                    + resp.get("eval_count", 0),
                },
                raw=resp,
            )
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
                **kwargs,
            )
            resp = self._client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            usage_meta = getattr(resp, "usage_metadata", None)
            prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
            return LLMResponse(
                content=resp.text or "",
                model=model_name,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                raw=resp,
            )
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
                **kwargs,
            )
            resp = await self._client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            usage_meta = getattr(resp, "usage_metadata", None)
            prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
            return LLMResponse(
                content=resp.text or "",
                model=model_name,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                raw=resp,
            )
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


# ── Factory ───────────────────────────────────────────────────────────────

_PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "stub": StubProvider,
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
