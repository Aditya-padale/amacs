# Architecture

AMACS is a library pipeline, not a distributed agent runtime. One decorated call creates one `Pipeline`, one communication bus, one tracer, and one set of agents. Providers are called through the `LLMProvider` interface.

## Pipeline

```mermaid
flowchart LR
    A[decorated function call] --> B[TaskAnalyzer]
    B --> C[TaskDecomposer]
    C --> D[AgentSelector]
    D --> E[Scheduler]
    E --> F[WaveExecutor]
    F --> G[Aggregator]
    G --> H[AMACSResult]
    F --> I[CommunicationBus]
    F --> J[Tracer]
```

`planner="llm"` replaces the template decomposer with `LLMTaskDecomposer`; invalid or unsafe plans fall back to the template path. The default `pipeline` mode uses `WaveExecutor`. `debate` and `manager_worker` are alternate coordinator paths.

## Wave Loop And Adaptation

```mermaid
flowchart TD
    A[ExecutionPlan] --> B{Remaining wave?}
    B -- no --> C[Aggregate results]
    B -- yes --> D[Run ready tasks concurrently]
    D --> E[Record result, metrics, bus, trace]
    E --> F{adaptive?}
    F -- no --> B
    F -- yes --> G[Monitor snapshot]
    G --> H[Evaluator thresholds and quality signals]
    H --> I[AdaptationEngine actions]
    I --> J[Reconfigurator mutates agents and remaining plan]
    J --> K[Optional current-wave retry/re-execution]
    K --> B
```

Adaptation is bounded by swap limits and the configured budget. A synchronous timeout releases the caller after the timeout, although its worker thread may finish later; this is a Python executor limitation.

## Coordination Modes

```mermaid
flowchart TD
    A[Pipeline] --> B{mode}
    B -->|pipeline| C[Scheduler and WaveExecutor]
    B -->|debate| D[DebateCoordinator]
    B -->|manager_worker| E[ManagerWorkerCoordinator]
    C --> F[Aggregator]
    D --> F
    E --> F
```

The coordination modes share providers, configuration, bus, and result types, but their quality has not been compared in a real-task study. Agent role prompts are distinct, but a provider still determines the generated content.

## Ownership Boundaries

- `decorator.py` preserves the public API and dispatches sync versus async calls.
- `pipeline.py` prepares, executes, and finalizes one call.
- `orchestrator/` owns task understanding, decomposition, selection, and scheduling.
- `agents/` owns prompts, provider calls, retries, timeouts, tool context, and revisions.
- `executor.py` owns waves, budgets, metrics, and adaptation.
- `aggregation/` owns ordering, validation, and final merge.
- `cache.py` and `tracing.py` are opt-in runtime services.

The compatibility modules `communication/bus.py` and `aggregation/aggregator.py` re-export implementations; they are not separate implementations.
