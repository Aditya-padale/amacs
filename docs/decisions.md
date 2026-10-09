# AMACS Architecture & Technical Decisions Log

## Session 2 Decisions (2026-10-09)

### 1. Sync Thread Cancellation & Timeout Handling
- **Context**: Python OS threads executed inside `ThreadPoolExecutor` cannot be forcefully killed from user space without native thread signals.
- **Decision**: In `BaseAgent._call_llm`, sync timeouts are enforced via `future.result(timeout=timeout)`. When a timeout occurs, AMACS immediately raises `AgentTimeoutError` and releases caller flow control to retry/adapt, while documenting that the underlying thread finishes in the background.

### 2. Pricing & Cost Calculations
- **Context**: Hardcoded rates (`0.000002` / `0.000006`) in `executor.py` were model-agnostic and stale.
- **Decision**: Created `amacs/pricing.py` containing `PRICE_TABLE` with per-model rates (USD per 1M tokens) as of 2026-10-09 (OpenAI, Anthropic, Gemini, Ollama, Stub), supporting user-supplied overrides and safe unknown defaults.

### 3. Clean Budget Degradation
- **Context**: When budgets are near exhaustion, crashing abruptly can waste partial progress.
- **Decision**: When budget consumption passes 75% threshold, `WaveExecutor` removes non-critical sub-tasks from remaining waves and records a structured `AdaptationEvent`. If hard budget ceilings are strictly exceeded without remaining optional tasks, `BudgetExceededError` is raised.

### 4. Strategy Rules & Per-Agent Model Routing
- **Context**: `StrategyPolicy` was limited to model lookups without behavioral differentiation for `performance`, `cost`, and `speed`.

## Session 3 Decisions (2026-10-09)

### 1. Unified `Pipeline` Runner & Decorator Refactoring
- **Context**: `amacs/decorator.py` previously contained ~150 lines of duplicate orchestration code between `_run_pipeline_sync` and `_run_pipeline_async`.
- **Decision**: Extracted `Pipeline` in `amacs/pipeline.py` which owns the 8 pipeline stages and exposes `run()` and `arun()` sharing identical preparation and finalization logic. `@amacs` delegates to `Pipeline`. Parity was verified with `test_pipeline_parity.py`.

### 2. Immediate Mid-Wave Adaptation & Re-execution
- **Context**: Inter-wave adaptation previously ran after wave completion without re-running failed or underperforming tasks in the current wave, allowing bad or empty outputs to propagate to subsequent waves.
- **Decision**: Enhanced `WaveExecutor` to run `Monitor` -> `Evaluator` -> `AdaptationEngine` -> `Reconfigurator` immediately after wave execution. For `RETRY_WITH_DIFFERENT_AGENT`, `SWAP_AGENT`, and `SWITCH_MODEL`, the failed or underperforming task is swapped to a fallback agent class and re-executed mid-wave before proceeding.

### 3. Comprehensive Quality Signals Beyond Failures
- **Context**: Evaluation previously triggered only on hard exceptions or latency thresholds.
- **Decision**: Added quality signal checks in `Evaluator` for empty/very short outputs, refusal pattern detection (`"I cannot..."`), repeated text outputs, low LLM-judge scores in metadata, and high budget burn rate. Adaptation triggers on these signals even when provider execution returned `success=True`.

### 4. Rich Adaptation Observability
- **Context**: `AdaptationEvent` lacked standard fields for event telemetry.
- **Decision**: Enhanced `AdaptationEvent` with `trigger`, `signal_values`, `action`, `target`, and `outcome` fields while maintaining backward compatibility with `action_type` and `target_agent_id`.

## Session 4 Decisions (2026-10-09)

### 1. Coordination Modes Integration
- **Context**: `DebateCoordinator` and `ManagerWorkerCoordinator` existed but were unwired and skipped bus publishing and token accounting.
- **Decision**: Wired `mode="pipeline" | "debate" | "manager_worker"` on `@amacs` decorator and `AMACSConfig`. Updated coordinators to pass communication bus to worker agents and aggregate token usage across multi-turn debate/worker sub-steps.

### 2. LLM Planner with Pydantic & DAG Validation
- **Context**: LLM task decomposition lacked schema enforcement and DAG safety checks.
- **Decision**: Wired `planner="template" | "llm"` into `Pipeline` with Pydantic `LLMSubTaskSchema` parsing and `Scheduler().plan(...)` DAG validation. Automatically logs an event and falls back to rule-based `TaskDecomposer` on invalid JSON or cycle detection.

### 3. Hardened Tool Execution & Pluggable Backends
- **Context**: `PythonCalcTool` used unsafe `eval()`, `WebSearchTool` returned hardcoded strings, and `VectorSearchTool` was unwired from vector DB adapters.
- **Decision**: Replaced `eval()` with a strict AST-based evaluator allowing only arithmetic nodes and whitelisted math functions with tests for escape attempts. Introduced `SearchBackend` interface with `MockSearchBackend` (explicitly labeling mock output) and wired `VectorSearchTool` to `VectorStore` adapters (`SimpleVectorStore`). Added bounded tool-use loop (`max_tool_steps`) in `SearchAgent` with bus publishing.

### 4. Candidate Selection & Critic/Reviser Loop
- **Context**: Critical sub-tasks lacked candidate exploration and iterative refinement.
- **Decision**: Added `candidates_k > 1` support in `BaseAgent` generating k candidate completions evaluated by an LLM judge. Implemented a rubric-scoring critic and writer revision loop (`max_revisions`, `critic_score_threshold`) bounded by iteration limits and score thresholds.

### 5. Decorator Contract & Output Validation
- **Context**: Decorator prompt contract was ambiguous, and structured Pydantic outputs were not validated.
- **Decision**: Added `task` (explicit prompt) and `input_mode` ("return_value_as_prompt" vs "context") parameters. Added `output_schema=<PydanticModel>` parsing with automated 1-step retry feeding validation error back to LLM.

### 6. ResponseCache & Offline Build System
- **Context**: `ResponseCache` was unwired and pyproject.toml / Makefile failed build verification offline.
- **Decision**: Wired `ResponseCache` behind `cache=True|path` keyed by provider, model, messages, and params. Updated `pyproject.toml` build system to `setuptools` and updated `Makefile` to use `--no-isolation` and `--system-site-packages` for clean offline verification.

