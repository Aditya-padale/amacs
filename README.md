# AMACS — Adaptive Multi-Agent Coordination System

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/amacs.svg)](https://pypi.org/project/amacs/)

**Add multi-agent AI coordination to any function with a single decorator.**

AMACS is a Python framework that transparently orchestrates multiple AI agents to
collaboratively solve complex tasks. Just decorate your function — AMACS handles
task decomposition, agent selection, parallel execution, adaptive monitoring, and
result aggregation behind the scenes.

---

## ✨ Features

- **Single-decorator API** — wrap any sync or async function with `@amacs(...)`
- **Automatic orchestration** — task analysis → decomposition → agent selection → scheduling → execution → aggregation
- **4 built-in agents** — Search, Analysis, Writer, Validator (all LLM-powered)
- **Adaptive control loop** — real-time monitoring with automatic agent swapping and task skipping
- **Multi-provider LLM support** — OpenAI, Anthropic, Ollama (local), with a common interface
- **Communication bus** — in-memory pub/sub for inter-agent knowledge sharing
- **Configurable strategies** — optimize for `performance`, `cost`, or `speed`
- **Extensible** — register custom agents, LLM providers, and merge strategies
- **Production-ready** — retry logic (tenacity), graceful degradation, optional Prometheus metrics

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────┐
│                   @amacs decorator                   │
│   (auto-detects sync/async, manages full pipeline)   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Task     │→ │    Task      │→ │   Agent      │ │
│  │  Analyzer  │  │  Decomposer  │  │  Selector    │ │
│  └────────────┘  └──────────────┘  └──────────────┘ │
│         │               │                │           │
│         ▼               ▼                ▼           │
│  ┌─────────────────────────────────────────────────┐ │
│  │                  Scheduler                      │ │
│  │  (DAG-based parallel execution, max_agents)     │ │
│  └─────────────────────────────────────────────────┘ │
│         │                                            │
│         ▼                                            │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ │
│  │  Search  │ │ Analysis │ │ Writer │ │Validator │ │
│  │  Agent   │ │  Agent   │ │ Agent  │ │  Agent   │ │
│  └──────────┘ └──────────┘ └────────┘ └──────────┘ │
│         │           │           │           │        │
│         ▼           ▼           ▼           ▼        │
│  ┌─────────────────────────────────────────────────┐ │
│  │            Communication Bus                    │ │
│  │  (shared context, pub/sub, conflict resolution) │ │
│  └─────────────────────────────────────────────────┘ │
│         │                                            │
│         ▼                                            │
│  ┌─────────────────────────────────────────────────┐ │
│  │   Adaptive Control Loop (optional)              │ │
│  │  Monitor → Evaluator → Engine → Reconfigurator  │ │
│  └─────────────────────────────────────────────────┘ │
│         │                                            │
│         ▼                                            │
│  ┌─────────────────────────────────────────────────┐ │
│  │              Aggregator                         │ │
│  │  (collect → validate → merge → return)          │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Core (uses stub LLM by default for testing)
pip install amacs

# With OpenAI
pip install amacs[openai]

# With Anthropic
pip install amacs[anthropic]

# With local LLM (Ollama)
pip install amacs[ollama]

# All LLM providers
pip install amacs[all-llm]

# Optional integrations
pip install amacs[pinecone]      # Vector DB support
pip install amacs[monitoring]    # Prometheus metrics
```

### Basic Usage

```python
from amacs import amacs

@amacs(max_agents=8, strategy="performance")
def research_task(topic: str) -> str:
    """Conduct comprehensive research on the given topic."""
    return f"Research on {topic}"

# Just call it — AMACS handles everything
result = research_task("Renewable Energy")
print(result)
```

### Async Support

```python
import asyncio
from amacs import amacs

@amacs(max_agents=4, strategy="speed", adaptive=True)
async def async_research(topic: str) -> str:
    """Async research with adaptive monitoring."""
    return f"Research on {topic}"

result = asyncio.run(async_research("Climate Change"))
```

### Custom Agents

```python
from amacs import amacs, BaseAgent, SubTask, register_agent

class DomainExpertAgent(BaseAgent):
    @property
    def agent_type(self) -> str:
        return "domain_expert"

    def system_prompt(self) -> str:
        return "You are a domain expert specialising in renewable energy..."

# Register it globally
register_agent("domain_expert", DomainExpertAgent)
```

---

## ⚙️ Configuration

### Decorator Parameters

| Parameter          | Type    | Default         | Description                                    |
|--------------------|---------|-----------------|------------------------------------------------|
| `max_agents`       | `int`   | `4`             | Maximum concurrent agents (1–64)               |
| `strategy`         | `str`   | `"performance"` | `"performance"`, `"cost"`, or `"speed"`        |
| `adaptive`         | `bool`  | `False`         | Enable the adaptive control loop               |
| `retry_limit`      | `int`   | `3`             | Per-agent retry limit (0–10)                   |
| `timeout`          | `float` | `120.0`         | Per-agent timeout in seconds                   |
| `llm_provider`     | `str`   | `None`          | LLM provider (`"openai"`, `"anthropic"`, etc.) |
| `llm_model`        | `str`   | `None`          | Model name (e.g. `"gpt-4o"`)                   |
| `skip_non_critical`| `bool`  | `True`          | Skip failed non-critical sub-tasks             |

### Environment Variables

| Variable              | Description                                    |
|-----------------------|------------------------------------------------|
| `AMACS_LLM_PROVIDER`  | Default LLM provider (falls back to `"stub"`)  |
| `OPENAI_API_KEY`       | OpenAI API key                                 |
| `ANTHROPIC_API_KEY`    | Anthropic API key                              |
| `OLLAMA_HOST`          | Ollama server URL (default `localhost:11434`)   |

---

## 🔄 Orchestration Flow

Every `@amacs`-decorated call follows this pipeline:

1. **TaskAnalyzer** — inspects the function's name, docstring, and arguments to infer the task domain (research, analysis, coding, etc.) and complexity.

2. **TaskDecomposer** — splits the task into an ordered list of sub-tasks with dependency edges (a simple DAG). The domain drives the template; complexity controls how many sub-tasks are emitted.

3. **AgentSelector** — maps each sub-task to the best-fit agent based on its label (`"search"` → SearchAgent, `"analyze"` → AnalysisAgent, etc.).

4. **Scheduler** — topologically sorts the DAG into parallel waves, respecting the `max_agents` limit. Uses `ThreadPoolExecutor` for sync, `asyncio.gather` for async.

5. **Execution** — each agent calls the configured LLM provider with a role-specific system prompt. Results are published to the Communication Bus.

6. **Adaptive Control** *(optional)* — monitors latency, token usage, and error rates. Flags underperforming agents and can swap, skip, or reduce the team mid-run.

7. **Aggregator** — collects all successful sub-task outputs, runs a final Validator pass, and merges them into a single coherent result.

---

## 📦 Package Structure

```
amacs/
├── amacs/
│   ├── __init__.py              # Public API
│   ├── decorator.py             # @amacs decorator
│   ├── config.py                # Pydantic configuration
│   ├── cli.py                   # CLI (amacs --version, amacs run)
│   ├── exceptions.py            # Exception hierarchy
│   ├── orchestrator/
│   │   ├── task_analyzer.py     # Domain/complexity inference
│   │   ├── task_decomposer.py   # DAG-based task splitting
│   │   ├── agent_selector.py    # Agent-to-task mapping
│   │   └── scheduler.py         # Parallel execution planning
│   ├── agents/
│   │   ├── base_agent.py        # Abstract Agent class
│   │   ├── search_agent.py      # Information retrieval
│   │   ├── analysis_agent.py    # Data analysis/reasoning
│   │   ├── writer_agent.py      # Content generation
│   │   └── validator_agent.py   # Fact-checking/verification
│   ├── communication/
│   │   └── bus.py               # Pub/sub + shared context
│   ├── adaptive/
│   │   ├── monitor.py           # Performance tracking
│   │   ├── evaluator.py         # Threshold evaluation
│   │   ├── adaptation_engine.py # Decision engine
│   │   └── reconfigurator.py    # Live reconfiguration
│   ├── aggregation/
│   │   └── aggregator.py        # Result synthesis
│   └── integrations/
│       ├── llm_providers.py     # OpenAI, Anthropic, Ollama
│       ├── vector_db.py         # Pinecone/FAISS adapters
│       └── monitoring.py        # Prometheus hooks
├── tests/
├── examples/
│   └── basic_usage.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 🧪 Testing

```bash
# Install dev dependencies
pip install amacs[dev]

# Run all tests
pytest

# With coverage
pytest --cov=amacs --cov-report=term-missing

# Type checking
mypy amacs/
```

---

## 📝 CLI

```bash
# Check version
amacs --version

# Run a script with AMACS
amacs run examples/basic_usage.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install dev dependencies (`pip install -e ".[dev]"`)
4. Make your changes and add tests
5. Run the test suite (`pytest`)
6. Submit a pull request

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
# amacs
