# AMACS — Adaptive Multi-Agent Coordination System

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Add multi-agent AI coordination to any function with a single decorator.**

AMACS is a Python framework that transparently orchestrates multiple AI agents to
collaboratively solve complex tasks. Just decorate your function — AMACS handles
task decomposition, agent selection, parallel execution, adaptive monitoring, and
result aggregation behind the scenes.

---

## ✨ Features

- **Single-decorator API** — wrap any sync or async function with `@amacs(...)`
- **Real-Time Inter-Wave Adaptation** — `WaveExecutor` monitors and adapts execution plan between task waves before subsequent waves run
- **Token & Budget Control** — token-aware prompt context budgeting (`ContextBuilder`) and hard token/cost budget limits (`max_cost_usd`, `max_total_tokens`)
- **Strategy-Driven Execution** — `StrategyPolicy` dynamically maps strategies (`performance`, `cost`, `speed`) to models and concurrency rules
- **Tool System** — agents execute tools (`WebSearchTool`, `PythonCalcTool`, `VectorSearchTool`) via `ToolRegistry`
- **Genuine Multi-Agent Protocols** — `DebateCoordinator` (proposer-critic consensus) and `ManagerWorkerCoordinator` (hierarchical delegation)
- **Dynamic LLM Task Decomposition** — `LLMTaskDecomposer` generates structured sub-task DAGs via LLMs
- **Multi-Provider LLM Support** — OpenAI, Anthropic, Gemini, and Ollama (local) with uniform interface
- **Telemetry & Benchmarking** — built-in execution tracing (`Tracer`), response caching (`ResponseCache`), and `amacs benchmark` CLI

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────┐
│                   @amacs decorator                   │
│   (auto-detects sync/async, manages full pipeline)   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Task     │→ │    Task      │→ │   Agent      │  │
│  │  Analyzer  │  │  Decomposer  │  │  Selector    │  │
│  └────────────┘  └──────────────┘  └──────────────┘  │
│         │               │                │           │
│         ▼               ▼                ▼           │
│  ┌─────────────────────────────────────────────────┐ │
│  │                  Scheduler                      │ │
│  │  (DAG-based parallel execution, max_agents)     │ │
│  └─────────────────────────────────────────────────┘ │
│         │                                            │
│         ▼                                            │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐   │
│  │  Search  │ │ Analysis │ │ Writer │ │Validator │   │
│  │  Agent   │ │  Agent   │ │ Agent  │ │  Agent   │   │
│  └──────────┘ └──────────┘ └────────┘ └──────────┘   │
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
# Install from this checkout (the project is not claiming a PyPI release here)
pip install .

# With OpenAI
pip install .[openai]

# With Anthropic
pip install .[anthropic]

# With local LLM (Ollama)
pip install .[ollama]

# All LLM providers
pip install .[all-llm]

# Optional integrations
pip install .[pinecone]         # Vector DB support
pip install .[monitoring]       # Prometheus metrics
```

The default provider is the deterministic `stub`; it echoes prompts for offline tests and is not a quality benchmark. `WebSearchTool` likewise uses an explicitly labeled mock backend unless an application supplies a real `SearchBackend`.

Run the reproducible local benchmark with `make bench-fixture`. It writes measured JSON and Markdown outputs under `benchmarks/results/`; no real API calls are made.

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

### Inspecting Agent Responses & Inter-Agent Communication

```python
from amacs import amacs

# Option 1: Live printing with verbose=True and returning details object
@amacs(max_agents=4, verbose=True, return_details=True)
def research_task(topic: str):
    return f"Research topic: {topic}"

result = research_task("Quantum Computing")

# Access individual agent outputs
for agent_res in result.agent_results:
    print(f"[{agent_res.agent_name}] -> {agent_res.content}")

# Access inter-agent pub/sub communication log
for log in result.communication_log:
    print(f"💬 [{log.writer}] published '{log.key}': {log.value}")

# Pretty-print formatted reports
result.print_agent_responses()
result.print_communication_log()

# Option 2: Inspect via wrapper attribute anytime
print(research_task.last_result.agent_results)
print(research_task.last_result.communication_log)
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
| `verbose`          | `bool`  | `False`         | Print agent responses & bus communications live|
| `return_details`   | `bool`  | `False`         | Return `AMACSResult` instead of plain string   |

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

## 🧠 Deep Dive: Responses, Aggregation, Adaptive Storage, Answer Selection & Validation

### 1. 🤖 Agent Responses (`amacs/agents/`)
Each agent inherits from `BaseAgent` and generates responses tailored to its role:

* **`SearchAgent` (`agent_type="search"`)**: Retrieves factual information, citing source types (academic, news, industry) with bullet points and clear headings.
* **`AnalysisAgent` (`agent_type="analysis"`)**: Interprets data, evaluates evidence quality, lists explicit assumptions, and quantifies confidence levels.
* **`WriterAgent` (`agent_type="write"`)**: Synthesises research and analysis into cohesive, professional written content.
* **`ValidatorAgent` (`agent_type="validate"`)**: Verifies facts, checks for internal contradictions, and generates either corrections or a quality confirmation.

**Output Structure**: Every agent returns an `AgentResult` object containing:
```python
AgentResult(
    sub_task_id="task_1",
    agent_name="search",
    content="...",                 # Textual response payload
    success=True,                   # Status flag
    error=None,                     # Error details if failed
    latency_seconds=1.24,           # Execution time
    token_usage={"total_tokens": 320},
    metadata={}
)
```

---

### 2. 🧩 How Answers Are Combined (`amacs/aggregation/`)
Result combination is handled by the `Aggregator` (`amacs/aggregation/__init__.py`):
1. **Filtering**: Successful results (`result.success == True` and non-empty `content`) are extracted.
2. **DAG Sequencing**: Outputs are sorted according to the original `SubTask` execution DAG sequence to preserve logical progression (Search → Analysis → Writer → Validator).
3. **Joining**: Ordered content strings are joined with double line breaks (`"\n\n"`).
4. **Final Synthesis Pass**: The merged text is passed to `ValidatorAgent` with full thread context snapshot from `CommunicationBus` to polish and eliminate inconsistencies.

---

### 3. 💾 Where Adaptive Answers & State Are Stored (`amacs/communication/` & `amacs/adaptive/`)
AMACS stores intermediate states, metrics, and adaptive decisions across three specialized layers:

* **Shared Execution Bus (`CommunicationBus`)**:
  * **State Store** (`_store: Dict[str, Any]`): Holds published outputs mapped by `sub_task_id` and `"original_input"`.
  * **Audit Log** (`_log: List[LogEntry]`): Append-only log recording timestamp, writer, value, and overwrite status for full auditability.
* **Adaptive Monitor (`Monitor`)**:
  * **`AgentMetrics`**: Real-time thread-safe metrics dictionary (`total_calls`, `failed_calls`, `total_latency`, `total_tokens`, `last_error`).
  * **`SystemSnapshot`**: Captures system health snapshots across execution waves.
* **Evaluation & Actions**:
  * `Evaluator` produces an `EvaluationReport` flagging agent health (`HEALTHY`, `DEGRADED`, `FAILING`).
  * `AdaptationEngine` translates flags into `AdaptationAction` directives (`SWAP_AGENT`, `SKIP_TASK`, `REDUCE_TEAM`).
  * `Reconfigurator` logs execution results in `ReconfigurationResult`.

---

### 4. 🏆 How the Best Answer is Selected (`amacs/communication/` & `amacs/adaptive/`)
Selection and optimization occur at three distinct levels:

* **Conflict Resolution on Bus**: Last-write-wins by default when duplicate keys are published, or a custom `merge_strategy(old_value, new_value)` function passed to `CommunicationBus`. All overwrites are recorded in `get_conflicts()`.
* **Dynamic Agent Selection & Swapping**: When `Evaluator` detects an underperforming or failing agent, `AdaptationEngine` triggers `SWAP_AGENT`. `Reconfigurator` dynamically swaps the failing agent with an alternative agent class from `_AGENT_REGISTRY` for remaining waves.
* **Validator Synthesis**: `ValidatorAgent` cross-references all context stored in `CommunicationBus`, resolves remaining discrepancies, and yields the final best output.

---

### 5. 🛡️ How Validation Works (`amacs/agents/`, `amacs/orchestrator/`, `amacs/aggregation/`)
Validation is enforced continuously throughout the execution lifecycle:

1. **Retry Logic**: `BaseAgent` uses `tenacity` exponential backoff retries (`retry_limit`, default 3) on transient LLM/provider failures.
2. **DAG Execution Integrity**: `Scheduler` tracks dependency waves and checks `SubTask.critical`. Critical failures raise `OrchestrationError` when `skip_non_critical=False`.
3. **Adaptive Threshold Checks**: `Evaluator` compares execution performance against `ThresholdConfig` (`max_failures`, `max_latency_seconds`, `max_error_rate`).
4. **Final Pass Validation**: `Aggregator._validate()` executes a final `ValidatorAgent` check over the merged payload to ensure factual consistency and quality before returning the result.


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
