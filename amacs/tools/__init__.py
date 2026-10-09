"""AMACS Tools module — exposes Tool base class, built-in tools, and ToolRegistry."""

from __future__ import annotations

from typing import Dict, List, Optional

from amacs.tools.base_tool import Tool
from amacs.tools.builtins import PythonCalcTool, VectorSearchTool, WebSearchTool


class ToolRegistry:
    """Registry for discovering and managing agent tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        # Register defaults
        self.register(PythonCalcTool())
        self.register(WebSearchTool())
        self.register(VectorSearchTool())

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name.lower()] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name.lower())

    def list_tools(self) -> List[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())


__all__ = [
    "Tool",
    "PythonCalcTool",
    "WebSearchTool",
    "VectorSearchTool",
    "ToolRegistry",
]
