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
- **Decision**: Expanded `StrategyRules` with `max_tokens`, `timeout_seconds`, `enable_validation_pass`, `max_concurrency_multiplier`, and `enable_critique_revision`. Added top-level `models` parameter for per-agent routing (e.g., `models={"search": "gpt-4o-mini", "write": "gpt-4o"}`) and `DEFAULT_MODELS` environment overrides.
