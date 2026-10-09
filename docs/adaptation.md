# Adaptation

Adaptive execution is enabled with `@amacs(adaptive=True)`. It is a bounded control loop over the current call, not a learned policy.

## Signals

`Monitor` records calls, failures, latency, tokens, last content, quality scores, and budget burn rate per agent. `Evaluator` compares those values with `ThresholdConfig`:

| Signal | Default trigger |
|---|---|
| Failed calls | `failed_calls >= 2` |
| Error rate | `failed_calls / total_calls >= 0.5` |
| Average latency | `> 30` seconds |
| Short output | fewer than 20 characters |
| Refusal output | configured refusal phrases |
| Repeated output | repeated lines or word chunks |
| Judge score | latest score `< 0.6` |
| Budget burn | `> 0.8` of available budget |

An agent is `HEALTHY`, `DEGRADED`, or `FAILING`. Evaluation requires at least one call by default. The values are implementation defaults and are not calibrated against a benchmark.

## Policy And Actions

The `AdaptationEngine` maps evaluations to actions:

- `FAILING` -> swap the agent when an alternative exists.
- `DEGRADED` due to a quality signal -> retry with a different agent.
- Other degraded cases -> skip the task when policy permits.
- More than half of the team failing -> reduce the remaining team.
- Healthy -> no action.

`Reconfigurator` applies actions to remaining waves, limits swaps per task, and records applied or skipped outcomes. Budget degradation can remove optional work before a hard token or cost ceiling is exceeded. All adaptation events are returned in `AMACSResult.adaptation_events` and are included in tracing/events where configured.

## Pseudo-code

```text
for wave in remaining_waves:
    results = execute_ready_tasks(wave)
    record_results(results)

    if adaptive:
        report = evaluate(monitor.snapshot())
        actions = policy(report, total_agents)
        outcome = reconfigure(actions, agents, plan, next_wave)
        if outcome requests a retry:
            run the affected remaining task with its replacement

    enforce token and cost budgets
return aggregate(successful_results)
```

## Interpretation

Adaptation changes execution after observed signals. It does not prove that a replacement agent is better, and it does not repair arbitrary semantic errors unless they produce a configured signal or a provider-side failure. The fixture benchmark demonstrates seeded fault recovery, not general adaptive quality.
