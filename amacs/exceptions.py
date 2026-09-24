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
    """Raised when an agent exceeds the configured timeout."""


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
