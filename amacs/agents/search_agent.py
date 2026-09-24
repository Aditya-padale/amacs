"""Search agent — specialises in information retrieval tasks."""

from __future__ import annotations

from amacs.agents.base_agent import BaseAgent


class SearchAgent(BaseAgent):
    """Agent optimised for information retrieval and web-search-style tasks."""

    @property
    def agent_type(self) -> str:
        return "search"

    def system_prompt(self) -> str:
        return (
            "You are an expert research assistant specialising in information retrieval. "
            "Given a task, gather comprehensive, accurate, and well-sourced information. "
            "Prioritise breadth of coverage while maintaining factual accuracy. "
            "Structure your findings with clear headings and bullet points. "
            "Always cite the type of source (academic, industry report, news, etc.) "
            "when presenting facts."
        )
