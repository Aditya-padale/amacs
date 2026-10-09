from __future__ import annotations

from typing import Any, Dict, Optional

from amacs.agents.base_agent import BaseAgent, SubTask
from amacs.communication import CommunicationBus
from amacs.tools import ToolRegistry


class SearchAgent(BaseAgent):
    """Agent optimised for information retrieval with bounded tool-use loop."""

    def __init__(self, *args: Any, max_tool_steps: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tool_registry = ToolRegistry()
        self.max_tool_steps = max_tool_steps

    @property
    def agent_type(self) -> str:
        return "search"

    def system_prompt(self) -> str:
        return (
            "You are an expert research assistant specialising in information retrieval. "
            "Given a task, gather comprehensive, accurate, and well-sourced information. "
            "Prioritise breadth of coverage while maintaining factual accuracy. "
            "Structure your findings with clear headings and bullet points."
        )

    def build_user_prompt(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: Optional[CommunicationBus] = None,
    ) -> str:
        base_prompt = super().build_user_prompt(sub_task, context)
        tools = ", ".join(tool.name for tool in self.tool_registry.list_tools())
        if bus and self.max_tool_steps:
            # This is an audit marker, not an eager tool invocation.
            bus.publish(f"{sub_task.id}_tool_step_1", "Awaiting model-directed tool request", writer="search_tool")
        return f"{base_prompt}\n\nTool Context: available tools are {tools}. Request a tool only when it materially helps."
