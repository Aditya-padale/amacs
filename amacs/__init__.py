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

from amacs.decorator import amacs  # noqa: F401, E402
from amacs.config import AMACSConfig, Strategy  # noqa: F401, E402
from amacs.exceptions import (  # noqa: F401, E402
    AMACSError,
    AgentError,
    AgentTimeoutError,
    AggregationError,
    CommunicationError,
    ConfigurationError,
    LLMProviderError,
    OrchestrationError,
    RetryExhaustedError,
    AdaptationError,
)
from amacs.agents.base_agent import BaseAgent, SubTask, AgentResult  # noqa: F401, E402
from amacs.orchestrator.agent_selector import register_agent  # noqa: F401, E402
from amacs.integrations.llm_providers import (  # noqa: F401, E402
    get_provider,
    register_provider,
    LLMProvider,
    Message,
    LLMResponse,
)
from amacs.communication import CommunicationBus  # noqa: F401, E402

__all__ = [
    # Decorator
    "amacs",
    # Config
    "AMACSConfig",
    "Strategy",
    # Agents
    "BaseAgent",
    "SubTask",
    "AgentResult",
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
