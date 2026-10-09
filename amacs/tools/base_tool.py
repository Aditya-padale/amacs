"""Base tool abstraction for AMACS agents.

Allows agents to execute deterministic functions or external APIs during their
execution loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    """Abstract base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for the tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description of what the tool does and when to use it."""

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON schema defining the expected arguments for the tool."""
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Synchronously execute the tool with given keyword arguments."""

    async def aexecute(self, **kwargs: Any) -> str:
        """Asynchronously execute the tool. Default calls sync execute."""
        return self.execute(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Export tool metadata for LLM tool-use schemas."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }
