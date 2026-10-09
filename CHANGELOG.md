# AMACS Changelog

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
