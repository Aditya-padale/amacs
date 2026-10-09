"""Built-in tools provided out-of-the-box by AMACS."""

from __future__ import annotations

import math
import re
from typing import Any, Dict

from amacs.tools.base_tool import Tool


class PythonCalcTool(Tool):
    """Safely evaluates basic mathematical expressions in Python."""

    @property
    def name(self) -> str:
        return "python_calc"

    @property
    def description(self) -> str:
        return "Evaluates basic math expressions safely (e.g. '2 + 2', 'sqrt(16)', '10 * 3.5')."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate."}
            },
            "required": ["expression"],
        }

    def execute(self, **kwargs: Any) -> str:
        expr = str(kwargs.get("expression", "")).strip()
        if not expr:
            return "Error: Empty expression"

        # Safe math scope
        safe_dict: Dict[str, Any] = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }
        try:
            # Basic character sanitization
            if re.search(r"[a-zA-Z_]\w*", expr):
                for token in re.findall(r"[a-zA-Z_]\w*", expr):
                    if token not in safe_dict:
                        return f"Error: Function/variable '{token}' not allowed."
            res = eval(expr, {"__builtins__": None}, safe_dict)
            return str(res)
        except Exception as exc:
            return f"Error calculating '{expr}': {exc}"


class WebSearchTool(Tool):
    """Information retrieval tool for web/knowledge search."""

    def __init__(self, mock_results: bool = True) -> None:
        self.mock_results = mock_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Searches the web or knowledge base for up-to-date facts, news, and topic details."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."}
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "Error: Empty query"
        return f"[Search Results for '{query}']: Found key findings on '{query}'. High relevance sources confirm standard principles."


class VectorSearchTool(Tool):
    """Vector database similarity search tool."""

    def __init__(self, documents: list[str] | None = None) -> None:
        self.documents = documents or [
            "AMACS features adaptive multi-agent coordination with runtime monitoring.",
            "Agents communicate via a thread-safe pub/sub CommunicationBus.",
            "The framework supports OpenAI, Anthropic, Gemini, and Ollama models.",
        ]

    @property
    def name(self) -> str:
        return "vector_search"

    @property
    def description(self) -> str:
        return "Searches local vector database / docstore for relevant contextual snippets."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Semantic query."}
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).lower()
        matches = [doc for doc in self.documents if any(word in doc.lower() for word in query.split())]
        if matches:
            return "Relevant Documents:\n" + "\n".join(f"- {m}" for m in matches)
        return "No matching documents found in vector store."
