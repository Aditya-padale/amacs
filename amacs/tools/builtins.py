"""Built-in tools provided out-of-the-box by AMACS."""

from __future__ import annotations

import ast
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from amacs.integrations.vector_db import SimpleVectorStore, VectorStore
from amacs.tools.base_tool import Tool

ALLOWED_FUNCS: Dict[str, Callable[..., Any]] = {
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
}

ALLOWED_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Constant of type '{type(node.value).__name__}' is not allowed")
    elif isinstance(node, ast.Name):
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Variable '{node.id}' is not allowed")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unary operator '{type(node.op).__name__}' is not allowed")
    elif isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        elif isinstance(node.op, ast.FloorDiv):
            return left // right
        elif isinstance(node.op, ast.Mod):
            return left % right
        elif isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError(f"Binary operator '{type(node.op).__name__}' is not allowed")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise TypeError("Call target must be a simple function name")
        func_name = node.func.id
        if func_name not in ALLOWED_FUNCS:
            raise ValueError(f"Function '{func_name}' is not allowed")
        args = [_eval_ast_node(arg) for arg in node.args]
        return ALLOWED_FUNCS[func_name](*args)
    else:
        raise TypeError(f"AST node '{type(node).__name__}' is forbidden")


class PythonCalcTool(Tool):
    """Safely evaluates basic mathematical expressions using AST parsing without eval."""

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

        try:
            parsed = ast.parse(expr, mode="eval")
            res = _eval_ast_node(parsed)
            return str(res)
        except Exception as exc:
            return f"Error calculating '{expr}': {exc}"


# ── Search Backends ─────────────────────────────────────────────────────────

class SearchBackend(ABC):
    """Abstract interface for web search backends."""

    @abstractmethod
    def search(self, query: str) -> str:
        """Execute search for a query string."""


class MockSearchBackend(SearchBackend):
    """Mock search backend explicitly labeling output as mock data for testing."""

    def search(self, query: str) -> str:
        return (
            f"[MOCK DATA] Search Results for '{query}': "
            f"Found key findings on '{query}'. High relevance sources confirm standard principles."
        )


class WebSearchTool(Tool):
    """Information retrieval tool for web/knowledge search with pluggable backend."""

    def __init__(self, backend: Optional[SearchBackend] = None, mock_results: bool = True) -> None:
        self.backend = backend or MockSearchBackend()
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
        return self.backend.search(query)


# ── Vector Search Tool ──────────────────────────────────────────────────────

class VectorSearchTool(Tool):
    """Vector database similarity search tool wired to VectorStore adapters."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        if vector_store is not None:
            self.store = vector_store
        else:
            default_docs = documents or [
                "AMACS features adaptive multi-agent coordination with runtime monitoring.",
                "Agents communicate via a thread-safe pub/sub CommunicationBus.",
                "The framework supports OpenAI, Anthropic, Gemini, and Ollama models.",
            ]
            self.store = SimpleVectorStore(documents=default_docs)

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
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "Error: Empty query"

        if isinstance(self.store, SimpleVectorStore):
            results = self.store.search_text(query)
            if results:
                return "Relevant Documents:\n" + "\n".join(f"- {doc}" for doc in results)
            return "No matching documents found in vector store."

        res_dicts = self.store.query(vector=[0.1] * 128, top_k=3)
        if res_dicts:
            return "Relevant Vector Matches:\n" + "\n".join(f"- {r}" for r in res_dicts)
        return "No matching documents found in vector store."
