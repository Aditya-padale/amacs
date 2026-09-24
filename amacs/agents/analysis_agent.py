"""Analysis agent — specialises in data analysis and reasoning tasks."""

from __future__ import annotations

from amacs.agents.base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    """Agent optimised for analytical reasoning, data interpretation, and synthesis."""

    @property
    def agent_type(self) -> str:
        return "analysis"

    def system_prompt(self) -> str:
        return (
            "You are an expert data analyst and critical thinker. "
            "Given information and a task, perform rigorous analysis: identify patterns, "
            "draw insights, evaluate evidence quality, and provide well-reasoned conclusions. "
            "Use structured reasoning — state assumptions, present evidence, and qualify "
            "your confidence level. When dealing with quantitative data, include relevant "
            "statistics and comparisons."
        )
