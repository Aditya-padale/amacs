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
        web_tool = self.tool_registry.get("web_search")
        vector_tool = self.tool_registry.get("vector_search")

        tool_outputs = []
        steps_run = 0

        if web_tool and steps_run < self.max_tool_steps:
            steps_run += 1
            out = web_tool.execute(query=sub_task.description)
            tool_outputs.append(out)
            if bus:
                bus.publish(f"{sub_task.id}_tool_step_{steps_run}", out, writer="search_tool")

        if vector_tool and steps_run < self.max_tool_steps:
            steps_run += 1
            out = vector_tool.execute(query=sub_task.description)
            tool_outputs.append(out)
            if bus:
                bus.publish(f"{sub_task.id}_tool_step_{steps_run}", out, writer="search_tool")

        if tool_outputs:
            combined = "\n\n".join(tool_outputs)
            return f"{base_prompt}\n\nTool Context:\n{combined}"
        return base_prompt
