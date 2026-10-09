"""AMACS — Adaptive Multi-Agent Coordination System.

Add multi-agent AI coordination to any function with a single decorator::

    from amacs import amacs

    @amacs(max_agents=8, strategy="performance")
    def research_task(topic: str) -> str:
        return f"Research on {topic}"

    result = research_task("Renewable Energy")
"""

from __future__ import annotations

__version__ = "0.1.0"

# ── Public API ────────────────────────────────────────────────────────────

from amacs.agents.base_agent import AgentResult, BaseAgent, SubTask
from amacs.communication import CommunicationBus, LogEntry
from amacs.config import AMACSConfig, Strategy
from amacs.decorator import amacs
from amacs.exceptions import (
    AdaptationError,
    AgentError,
    AgentTimeoutError,
    AggregationError,
    AMACSError,
    CommunicationError,
    ConfigurationError,
    LLMProviderError,
    OrchestrationError,
    RetryExhaustedError,
)
from amacs.integrations.llm_providers import (
    LLMProvider,
    LLMResponse,
    Message,
    get_provider,
    register_provider,
)
from amacs.orchestrator.agent_selector import register_agent
from amacs.results import AMACSResult

__all__ = [
    # Decorator
    "amacs",
    # Config
    "AMACSConfig",
    "Strategy",
    # Results & Inspection
    "AMACSResult",
    "AgentResult",
    "LogEntry",
    "SubTask",
    # Agents
    "BaseAgent",
    "register_agent",
    # LLM
    "get_provider",
    "register_provider",
    "LLMProvider",
    "Message",
    "LLMResponse",
    # Communication
    "CommunicationBus",
    # Exceptions
    "AMACSError",
    "AgentError",
    "AgentTimeoutError",
    "AggregationError",
    "CommunicationError",
    "ConfigurationError",
    "LLMProviderError",
    "OrchestrationError",
    "RetryExhaustedError",
    "AdaptationError",
    # Meta
    "__version__",
]
