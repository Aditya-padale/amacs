"""AMACS exception hierarchy.

All AMACS-specific exceptions inherit from AMACSError so callers can
catch everything with a single ``except AMACSError``.
"""

from __future__ import annotations


class AMACSError(Exception):
    """Base exception for all AMACS errors."""


class ConfigurationError(AMACSError):
    """Raised when decorator or runtime configuration is invalid."""


class OrchestrationError(AMACSError):
    """Raised when the orchestration pipeline encounters an unrecoverable error."""


class AgentError(AMACSError):
    """Raised when an individual agent fails irrecoverably."""


class AgentTimeoutError(AgentError):
    """Raised when an agent exceeds the configured timeout.

    This is a retryable error — the adaptive loop may re-execute with
    a different agent or extended timeout.
    """


class CommunicationError(AMACSError):
    """Raised when the communication bus encounters an error."""


class LLMProviderError(AMACSError):
    """Raised when the LLM provider returns an error or is unavailable."""


class AggregationError(AMACSError):
    """Raised when result aggregation fails."""


class AdaptationError(AMACSError):
    """Raised when the adaptive control loop encounters an error."""


class RetryExhaustedError(AgentError):
    """Raised when all retry attempts for an agent have been exhausted."""


class BudgetExceededError(AMACSError):
    """Raised when cost or token budget is exhausted."""


# ── Retryable error classification ──────────────────────────────────────

from typing import Any, Callable, Coroutine, TypeVar

from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

T = TypeVar("T")

#: Errors that should be retried with backoff (transient failures).
RETRYABLE_ERRORS = (
    AgentTimeoutError,
)

#: LLM provider error substrings that indicate a retryable condition.
RETRYABLE_PROVIDER_PATTERNS = frozenset({
    "rate limit",
    "rate_limit",
    "429",
    "timeout",
    "timed out",
    "connection",
    "502",
    "503",
    "504",
    "server error",
    "internal server error",
    "overloaded",
    "capacity",
})

#: LLM provider error substrings that indicate a fatal (non-retryable) condition.
FATAL_PROVIDER_PATTERNS = frozenset({
    "authentication",
    "auth",
    "401",
    "403",
    "invalid api key",
    "invalid_api_key",
    "permission",
    "bad request",
    "400",
    "not found",
    "404",
    "invalid model",
})


def should_retry_exception(exc: BaseException) -> bool:
    """Predicate for retry logic to check if an exception is retryable."""
    if isinstance(exc, RETRYABLE_ERRORS):
        return True

    target_exc = exc
    if getattr(exc, "__cause__", None) is not None and not hasattr(exc, "status_code"):
        target_exc = exc.__cause__  # type: ignore[assignment]

    status_code = (
        getattr(target_exc, "status_code", None)
        or getattr(target_exc, "http_status", None)
        or getattr(target_exc, "code", None)
    )
    if isinstance(status_code, int):
        if status_code in (400, 401, 403, 404) or (400 <= status_code < 500 and status_code != 429):
            return False
        if status_code == 429 or status_code >= 500:
            return True

    class_name = target_exc.__class__.__name__
    if any(fatal in class_name for fatal in ("Authentication", "Permission", "BadRequest", "NotFound", "InvalidKey")):
        return False
    if any(retryable in class_name for retryable in ("RateLimit", "Connection", "Timeout", "InternalServer", "ServiceUnavailable", "Overloaded")):
        return True

    msg = str(target_exc).lower()
    for pattern in FATAL_PROVIDER_PATTERNS:
        if pattern in msg:
            return False
    for pattern in RETRYABLE_PROVIDER_PATTERNS:
        if pattern in msg:
            return True

    return False


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Determine if an LLM provider error is retryable."""
    return should_retry_exception(exc)


def run_with_retry(
    fn: Callable[[], T],
    max_attempts: int = 3,
) -> T:
    """Execute a synchronous function with unified tenacity retry logic and jitter."""
    attempts = max(max_attempts, 1)
    retryer = Retrying(
        retry=retry_if_exception(should_retry_exception),
        stop=stop_after_attempt(attempts),
        wait=wait_random_exponential(multiplier=0.5, min=0.5, max=10),
        reraise=True,
    )
    try:
        return retryer(fn)
    except Exception as exc:
        if not should_retry_exception(exc):
            raise
        raise RetryExhaustedError(
            f"exhausted {attempts} retries: {exc}"
        ) from exc


async def arun_with_retry(
    afn: Callable[[], Coroutine[Any, Any, T]],
    max_attempts: int = 3,
) -> T:
    """Execute an asynchronous coroutine with unified tenacity retry logic and jitter."""
    attempts = max(max_attempts, 1)
    retryer = AsyncRetrying(
        retry=retry_if_exception(should_retry_exception),
        stop=stop_after_attempt(attempts),
        wait=wait_random_exponential(multiplier=0.5, min=0.5, max=10),
        reraise=True,
    )
    try:
        async for attempt in retryer:
            with attempt:
                return await afn()
        raise RuntimeError("Unreachable")
    except Exception as exc:
        if not should_retry_exception(exc):
            raise
        raise RetryExhaustedError(
            f"exhausted {attempts} async retries: {exc}"
        ) from exc
