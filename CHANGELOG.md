# Changelog

## Unreleased

- Wired execution tracing, structured trace rendering, event hooks, and per-agent/stage cost reports into pipeline results.
- Added `amacs trace` and `amacs bench`; `amacs run` now accepts provider, model, and strategy overrides.
- Added an offline fixture benchmark with seeded fault injection, a single-call baseline, JSON results, and markdown reporting.
- Made the benchmark package installable and corrected the aggregation compatibility import.

# AMACS Changelog

## [Unreleased] - Session 3 (2026-10-09)

### Added
- **Unified `Pipeline` Class**: Extracted `Pipeline` class in `amacs.pipeline` that encapsulates end-to-end orchestration logic, exposing identical `run()` and `arun()` methods.
- **Wave Loop Mid-Wave Re-execution**: Implemented live mid-wave adaptation and re-execution in `WaveExecutor` for `RETRY_WITH_DIFFERENT_AGENT`, `SWAP_AGENT`, and `SWITCH_MODEL`.
- **Quality Signal Detection**: Added quality signal evaluations to `Evaluator` and `Monitor` for empty/very short outputs, refusal patterns, repeated text outputs, LLM judge scores, and budget burn rate.
- **Rich Observability for Adaptation**: Enhanced `AdaptationEvent` with `trigger`, `signal_values`, `action`, `target`, and `outcome` fields for detailed tracing.
- **Comprehensive Behavioral Test Suite**: Added `tests/test_pipeline_parity.py` and `tests/test_adaptation_behavioral.py` with 9 behavioral adaptation tests covering retries, quality signals, swap limits, and budget degradation.

### Changed
- **Decorator Refactoring**: Simplified `@amacs` in `amacs.decorator` to delegate sync and async execution to `Pipeline`, eliminating duplicate orchestration code.

## [Unreleased] - Session 2 (2026-10-09)

### Added
- **Scripted & Flaky Testing Providers**: Added `FakeProvider` (programmed call sequence, call history recording, index-based error injection) and `FlakyProvider` (latency, failure rate, empty output, seeded RNG) to `amacs.integrations.llm_providers`.
- **Dynamic Pricing Engine**: Added `amacs.pricing` with `PRICE_TABLE` (as of 2026-10-09 rates for OpenAI, Anthropic, Gemini, Ollama), user overrides support, and `calculate_cost()` helper.
- **Per-Agent Model Routing**: Added `models={"search": "...", "write": "..."}` parameter to `AMACSConfig` and `@amacs` decorator.
- **`DEFAULT_MODELS` Mapping**: Added environment variable overridable default model mapping (`AMACS_DEFAULT_OPENAI_MODEL`, etc.).
- **Clean Budget Degradation**: Added automatic optional task trimming and `AdaptationEvent` logging when token/cost budgets approach threshold.

### Changed
- **Real Timeout Enforcement**: Updated `BaseAgent` sync and async execution to enforce per-agent timeouts via `ThreadPoolExecutor` and `asyncio.wait_for`, raising retryable `AgentTimeoutError`.
- **Expanded Strategy Rules**: Enhanced `StrategyPolicy` to enforce strategy-specific max tokens, concurrency multipliers, validation pass skipping, and critique/revision toggles across `performance`, `cost`, and `speed`.
- **Aggregator Validation Pass**: Updated `Aggregator` to respect strategy rules and skip optional validation passes under `cost` and `speed` strategies.
