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


def is_retryable_provider_error(exc: LLMProviderError) -> bool:
    """Determine if an LLM provider error is retryable based on message content."""
    msg = str(exc).lower()
    # Check for fatal patterns first (they take priority)
    for pattern in FATAL_PROVIDER_PATTERNS:
        if pattern in msg:
            return False
    # Check for retryable patterns
    for pattern in RETRYABLE_PROVIDER_PATTERNS:
        if pattern in msg:
            return True
    # Default: not retryable (fail fast on unknown errors)
    return False
