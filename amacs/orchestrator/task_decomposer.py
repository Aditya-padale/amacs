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
         "after": [], "critical": True},
        {"label": "analyze", "desc": "Analyse and synthesise the gathered information",
         "after": ["search"], "critical": True},
        {"label": "write", "desc": "Compose a well-structured report from the analysis",
         "after": ["analyze"], "critical": True},
        {"label": "validate", "desc": "Fact-check and verify the final report",
         "after": ["write"], "critical": False},
    ],
    "analysis": [
        {"label": "search", "desc": "Collect relevant data and background information",
         "after": [], "critical": True},
        {"label": "analyze", "desc": "Perform detailed analysis on the collected data",
         "after": ["search"], "critical": True},
        {"label": "analyze", "desc": "Identify patterns, trends, and insights",
         "after": ["analyze"], "critical": True},
        {"label": "write", "desc": "Summarise findings into a clear report",
         "after": ["analyze"], "critical": True},
        {"label": "validate", "desc": "Verify analytical conclusions for accuracy",
         "after": ["write"], "critical": False},
    ],
    "content_creation": [
        {"label": "search", "desc": "Research background material for the content",
         "after": [], "critical": True},
        {"label": "write", "desc": "Draft the main content piece",
         "after": ["search"], "critical": True},
        {"label": "validate", "desc": "Review the draft for quality and accuracy",
         "after": ["write"], "critical": False},
    ],
    "coding": [
        {"label": "search", "desc": "Research relevant APIs, libraries, and patterns",
         "after": [], "critical": True},
        {"label": "analyze", "desc": "Design the solution architecture",
         "after": ["search"], "critical": True},
        {"label": "write", "desc": "Implement the solution",
         "after": ["analyze"], "critical": True},
        {"label": "validate", "desc": "Review and test the implementation",
         "after": ["write"], "critical": True},
    ],
    "planning": [
        {"label": "search", "desc": "Gather context and constraints",
         "after": [], "critical": True},
        {"label": "analyze", "desc": "Evaluate options and trade-offs",
         "after": ["search"], "critical": True},
        {"label": "write", "desc": "Draft the plan document",
         "after": ["analyze"], "critical": True},
        {"label": "validate", "desc": "Review plan feasibility",
         "after": ["write"], "critical": False},
    ],
    "general": [
        {"label": "search", "desc": "Gather relevant information",
         "after": [], "critical": True},
        {"label": "analyze", "desc": "Process and analyse the information",
         "after": ["search"], "critical": True},
        {"label": "write", "desc": "Produce the final output",
         "after": ["analyze"], "critical": True},
        {"label": "validate", "desc": "Verify the output quality",
         "after": ["write"], "critical": False},
    ],
}


import copy
from collections import defaultdict

from amacs.exceptions import OrchestrationError


class TaskDecomposer:
    """Splits a :class:`TaskProfile` into an ordered list of :class:`SubTask` objects."""

    def decompose(self, profile: TaskProfile) -> List[SubTask]:
        """Return sub-tasks as a simple DAG (list with dependency IDs)."""
        raw_template = _TEMPLATES.get(profile.domain, _TEMPLATES["general"])
        template = copy.deepcopy(raw_template)

        target = profile.estimated_sub_tasks
        if target < len(template):
            template = template[:target]
        elif target > len(template) and profile.secondary_domains:
            for sd in profile.secondary_domains[: target - len(template)]:
                extra = {
                    "label": "search",
                    "desc": f"Gather additional information related to {sd}",
                    "after": [],
                    "critical": False,
                }
                template.insert(1, extra)

        task_ids: List[str] = [f"{tmpl['label']}_{idx}" for idx, tmpl in enumerate(template)]
        role_to_ids: Dict[str, List[str]] = defaultdict(list)
        for idx, tmpl in enumerate(template):
            role_to_ids[tmpl["label"]].append(task_ids[idx])

        sub_tasks: List[SubTask] = []

        for idx, tmpl in enumerate(template):
            task_id = task_ids[idx]
            label = tmpl["label"]
            desc = tmpl["desc"]
            if profile.arg_summary:
                desc = f"{desc} (context: {profile.arg_summary})"

            after_roles = tmpl.get("after", [])
            deps: List[str] = []

            if label == "analyze" and "search" in role_to_ids:
                # Rule: analyze task depends on ALL search tasks
                deps.extend(role_to_ids["search"])
            else:
                for role in after_roles:
                    preceding = [
                        tid for tid in role_to_ids.get(role, [])
                        if task_ids.index(tid) < idx
                    ]
                    if not preceding:
                        raise OrchestrationError(
                            f"Task '{task_id}' specifies dependency on role '{role}', but no preceding sub-task with that role exists."
                        )
                    deps.extend(preceding)

            resolved_deps = list(dict.fromkeys([d for d in deps if d != task_id]))

            sub_tasks.append(
                SubTask(
                    id=task_id,
                    label=label,
                    description=desc,
                    dependencies=resolved_deps,
                    critical=tmpl["critical"],
                )
            )

        return sub_tasks
