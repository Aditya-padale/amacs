"""Validator agent — specialises in fact-checking and verification."""

from __future__ import annotations

from amacs.agents.base_agent import BaseAgent


class ValidatorAgent(BaseAgent):
    """Agent optimised for fact-checking, consistency verification, and quality control."""

    @property
    def agent_type(self) -> str:
        return "validate"

    def system_prompt(self) -> str:
        return (
            "You are an expert fact-checker and quality reviewer. "
            "Given content and its supporting research, verify factual accuracy, "
            "check for internal contradictions, identify unsupported claims, and "
            "ensure logical consistency. "
            "Respond ONLY in valid JSON format matching this schema:\n"
            '{"verdict": "pass" | "revise", "issues": ["issue 1", ...], "revised_text": "string or null"}'
        )
