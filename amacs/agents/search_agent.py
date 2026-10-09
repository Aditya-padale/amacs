"""Search agent — specialises in information retrieval tasks."""

from typing import Any, Dict

from amacs.agents.base_agent import BaseAgent, SubTask
from amacs.tools import ToolRegistry


class SearchAgent(BaseAgent):
    """Agent optimised for information retrieval with built-in tool support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tool_registry = ToolRegistry()

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

    def build_user_prompt(self, sub_task: SubTask, context: Dict[str, Any]) -> str:
        base_prompt = super().build_user_prompt(sub_task, context)
        # Execute search tool to enrich context
        web_tool = self.tool_registry.get("web_search")
        if web_tool:
            tool_output = web_tool.execute(query=sub_task.description)
            return f"{base_prompt}\n\nTool Context:\n{tool_output}"
        return base_prompt
