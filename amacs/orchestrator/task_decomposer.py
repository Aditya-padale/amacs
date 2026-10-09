"""Task decomposer — breaks a :class:`TaskProfile` into a DAG of sub-tasks.

The decomposition is deterministic and rule-based: the domain drives a
template of sub-task labels, and the complexity controls how many are emitted.
"""

from __future__ import annotations

from typing import Any, Dict, List

from amacs.agents.base_agent import SubTask
from amacs.orchestrator.task_analyzer import TaskProfile

# ── Domain → sub-task templates ───────────────────────────────────────────

_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "research": [
        {"label": "search", "desc": "Gather comprehensive information on the topic",
         "deps": [], "critical": True},
        {"label": "analyze", "desc": "Analyse and synthesise the gathered information",
         "deps": ["search_0"], "critical": True},
        {"label": "write", "desc": "Compose a well-structured report from the analysis",
         "deps": ["analyze_1"], "critical": True},
        {"label": "validate", "desc": "Fact-check and verify the final report",
         "deps": ["write_2"], "critical": False},
    ],
    "analysis": [
        {"label": "search", "desc": "Collect relevant data and background information",
         "deps": [], "critical": True},
        {"label": "analyze", "desc": "Perform detailed analysis on the collected data",
         "deps": ["search_0"], "critical": True},
        {"label": "analyze", "desc": "Identify patterns, trends, and insights",
         "deps": ["analyze_1"], "critical": True},
        {"label": "write", "desc": "Summarise findings into a clear report",
         "deps": ["analyze_2"], "critical": True},
        {"label": "validate", "desc": "Verify analytical conclusions for accuracy",
         "deps": ["write_3"], "critical": False},
    ],
    "content_creation": [
        {"label": "search", "desc": "Research background material for the content",
         "deps": [], "critical": True},
        {"label": "write", "desc": "Draft the main content piece",
         "deps": ["search_0"], "critical": True},
        {"label": "validate", "desc": "Review the draft for quality and accuracy",
         "deps": ["write_1"], "critical": False},
    ],
    "coding": [
        {"label": "search", "desc": "Research relevant APIs, libraries, and patterns",
         "deps": [], "critical": True},
        {"label": "analyze", "desc": "Design the solution architecture",
         "deps": ["search_0"], "critical": True},
        {"label": "write", "desc": "Implement the solution",
         "deps": ["analyze_1"], "critical": True},
        {"label": "validate", "desc": "Review and test the implementation",
         "deps": ["write_2"], "critical": True},
    ],
    "planning": [
        {"label": "search", "desc": "Gather context and constraints",
         "deps": [], "critical": True},
        {"label": "analyze", "desc": "Evaluate options and trade-offs",
         "deps": ["search_0"], "critical": True},
        {"label": "write", "desc": "Draft the plan document",
         "deps": ["analyze_1"], "critical": True},
        {"label": "validate", "desc": "Review plan feasibility",
         "deps": ["write_2"], "critical": False},
    ],
    "general": [
        {"label": "search", "desc": "Gather relevant information",
         "deps": [], "critical": True},
        {"label": "analyze", "desc": "Process and analyse the information",
         "deps": ["search_0"], "critical": True},
        {"label": "write", "desc": "Produce the final output",
         "deps": ["analyze_1"], "critical": True},
        {"label": "validate", "desc": "Verify the output quality",
         "deps": ["write_2"], "critical": False},
    ],
}


import copy


class TaskDecomposer:
    """Splits a :class:`TaskProfile` into an ordered list of :class:`SubTask` objects."""

    def decompose(self, profile: TaskProfile) -> List[SubTask]:
        """Return sub-tasks as a simple DAG (list with dependency IDs)."""
        raw_template = _TEMPLATES.get(profile.domain, _TEMPLATES["general"])
        # Deep copy to avoid mutating module-level templates
        template = copy.deepcopy(raw_template)

        # Trim or extend based on estimated sub-task count
        target = profile.estimated_sub_tasks
        if target < len(template):
            template = template[:target]
        elif target > len(template) and profile.secondary_domains:
            for sd in profile.secondary_domains[: target - len(template)]:
                extra = {
                    "label": "search",
                    "desc": f"Gather additional information related to {sd}",
                    "deps": [],
                    "critical": False,
                }
                template.insert(1, extra)

        sub_tasks: List[SubTask] = []
        task_ids: List[str] = [f"{tmpl['label']}_{idx}" for idx, tmpl in enumerate(template)]
        valid_id_set: set[str] = set(task_ids)

        for idx, tmpl in enumerate(template):
            task_id = task_ids[idx]
            desc = tmpl["desc"]
            if profile.arg_summary:
                desc = f"{desc} (context: {profile.arg_summary})"

            # Sanitize and validate dependencies
            raw_deps = tmpl.get("deps", [])
            valid_deps: List[str] = []
            for dep in raw_deps:
                if dep in valid_id_set and dep != task_id:
                    valid_deps.append(dep)

            # If a dependency reference was broken by trimming/expansion and we are not step 0,
            # fall back to depending on the immediately preceding task
            if raw_deps and not valid_deps and idx > 0:
                valid_deps.append(task_ids[idx - 1])

            sub_tasks.append(
                SubTask(
                    id=task_id,
                    label=tmpl["label"],
                    description=desc,
                    dependencies=valid_deps,
                    critical=tmpl["critical"],
                )
            )

        return sub_tasks
