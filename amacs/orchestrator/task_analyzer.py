"""Task analyzer — inspects a decorated function to infer task domain and complexity.

Uses the function's name, docstring, parameter names, and the actual arguments
passed at call-time to produce a :class:`TaskProfile` that downstream stages
(decomposer, agent selector) can act on.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

# ── Domain keyword map ────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "research": [
        "research", "investigate", "study", "explore", "survey", "literature",
        "review", "examine", "discover", "find",
    ],
    "analysis": [
        "analyze", "analyse", "compare", "evaluate", "assess", "measure",
        "calculate", "statistics", "data", "metrics", "benchmark",
    ],
    "content_creation": [
        "write", "draft", "compose", "generate", "create", "author",
        "summarize", "summarise", "report", "article", "blog", "essay",
    ],
    "coding": [
        "code", "implement", "develop", "program", "debug", "refactor",
        "test", "deploy", "build", "compile",
    ],
    "planning": [
        "plan", "strategy", "roadmap", "schedule", "organise", "organize",
        "design", "architect", "outline",
    ],
}

_COMPLEXITY_SIGNALS: Dict[str, float] = {
    # words / phrases that raise perceived complexity
    "comprehensive": 0.3,
    "detailed": 0.2,
    "thorough": 0.3,
    "in-depth": 0.3,
    "multi-step": 0.4,
    "complex": 0.4,
    "advanced": 0.3,
    "compare": 0.2,
    "contrast": 0.2,
    "all": 0.1,
    "every": 0.1,
}


@dataclass
class TaskProfile:
    """Result of analysing a decorated function call."""

    domain: str  # primary domain (e.g. "research")
    secondary_domains: List[str] = field(default_factory=list)
    complexity: float = 0.5  # 0.0 – 1.0
    description: str = ""
    function_name: str = ""
    arg_summary: str = ""
    estimated_sub_tasks: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskAnalyzer:
    """Analyses a function + its call-time arguments to produce a :class:`TaskProfile`."""

    def analyze(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> TaskProfile:
        """Return a :class:`TaskProfile` describing the inferred task."""
        text = self._extract_text(func, args, kwargs)
        domain, secondary = self._infer_domains(text)
        complexity = self._estimate_complexity(text)
        arg_summary = self._summarise_args(func, args, kwargs)

        # heuristic for sub-task count
        estimated = max(2, min(int(complexity * 6) + 2, 8))

        return TaskProfile(
            domain=domain,
            secondary_domains=secondary,
            complexity=complexity,
            description=text,
            function_name=func.__name__,
            arg_summary=arg_summary,
            estimated_sub_tasks=estimated,
        )

    # ── internals ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> str:
        """Concatenate all textual signals from the function and its arguments."""
        parts: List[str] = [func.__name__]
        doc = inspect.getdoc(func)
        if doc:
            parts.append(doc)
        # parameter names
        sig = inspect.signature(func)
        parts.extend(sig.parameters.keys())
        # stringified args
        for a in args:
            parts.append(str(a))
        for v in kwargs.values():
            parts.append(str(v))
        return " ".join(parts).lower()

    @staticmethod
    def _infer_domains(text: str) -> Tuple[str, List[str]]:
        scores: Dict[str, float] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = sum(1.0 for kw in keywords if kw in text)
            if score > 0:
                scores[domain] = score
        if not scores:
            return "general", []
        ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        return ranked[0], ranked[1:]

    @staticmethod
    def _estimate_complexity(text: str) -> float:
        base = 0.3
        for signal, weight in _COMPLEXITY_SIGNALS.items():
            if signal in text:
                base += weight
        # longer text → slightly higher complexity
        word_count = len(text.split())
        base += min(word_count / 200, 0.2)
        return min(base, 1.0)

    @staticmethod
    def _summarise_args(
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> str:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        parts: List[str] = []
        for i, a in enumerate(args):
            name = params[i] if i < len(params) else f"arg{i}"
            parts.append(f"{name}={a!r}")
        for k, v in kwargs.items():
            parts.append(f"{k}={v!r}")
        return ", ".join(parts)
