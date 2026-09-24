"""Writer agent — specialises in content generation and synthesis."""

from __future__ import annotations

from amacs.agents.base_agent import BaseAgent


class WriterAgent(BaseAgent):
    """Agent optimised for generating well-structured written content."""

    @property
    def agent_type(self) -> str:
        return "write"

    def system_prompt(self) -> str:
        return (
            "You are an expert writer and content synthesiser. "
            "Given research findings, analysis, and context, produce clear, engaging, "
            "and well-structured written content. Maintain a professional tone, use "
            "logical flow between sections, and ensure the output reads as a cohesive "
            "piece rather than a collection of fragments. Adapt your writing style to "
            "match the audience implied by the task."
        )
