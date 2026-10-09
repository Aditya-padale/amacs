# Paper Notes

## Problem Statement

Multi-step tasks often need retrieval, analysis, drafting, and checking, but a fixed multi-agent plan can waste calls or amplify a failed component. AMACS explores whether lightweight runtime signals can reconfigure a dependency-aware agent pipeline during one execution while preserving a simple decorator API.

This repository currently establishes an implementation and offline fault-injection harness. It does not establish a quality improvement over a single model call.

## Formal Adaptation Description

Let the execution state at step $t$ be

$$s_t = (P_t, A_t, M_t, B_t, R_t),$$

where $P_t$ is the remaining wave plan, $A_t$ maps task IDs to agents, $M_t$ is the monitor snapshot, $B_t$ is remaining token/cost budget, and $R_t$ is the accumulated result and event history. For each completed call, the monitor records latency, success/failure, token use, output text, optional judge score, and burn rate.

The evaluator computes signals $x_t$ and assigns each agent a status using fixed thresholds. The policy is a deterministic function

$$a_t = \pi(x_t, s_t),$$

where actions include no-op, retry with another agent, swap agent, switch model, skip task, add agent, or reduce team. The reconfigurator applies valid actions to $P_t$ and $A_t$, subject to swap and budget bounds, producing $s_{t+1}$. The loop terminates when no waves remain, a hard budget is exceeded, or an orchestration error is surfaced.

The current policy is rule-based. It is not trained, and thresholds are not learned from a quality dataset.

## Related Work

The comparison points for a future study are AutoGen, LangGraph, CrewAI, and MetaGPT as agent orchestration or software-agent frameworks; multi-agent debate as a deliberation pattern; and self-consistency as repeated-sample answer selection. AMACS differs in emphasis rather than claiming a new foundation model method: its central artifact is a dependency-aware wave executor with runtime monitoring and bounded reconfiguration behind a decorator.

A related-work comparison must pin versions, APIs, providers, prompts, and workloads. No such comparison is reported in this repository.

## Research Questions

- **RQ1:** Does adaptive reconfiguration improve success under controlled provider faults?
- **RQ2:** What does each component contribute: decomposition, scheduling, communication, adaptation, validation, caching, and coordination mode?
- **RQ3:** What is the cost-quality trade-off by strategy (`performance`, `cost`, and `speed`)?

## Threats To Validity

The fixture is tiny, synthetic, and deterministic. Its success predicate is string containment, the stub provider echoes prompts, and injected failures may not represent provider outages or semantic errors. Latency is machine-dependent. Provider SDKs and model behavior change over time. The default templates, thresholds, and prompts may favor AMACS. The repository does not yet include blinded human grading, a held-out dataset, confidence intervals, or a pre-registered analysis.

## Limitations

The default search backend is mock data, the template pipeline is often sequential, and model selection is configuration-driven rather than learned. Adaptive signals are proxies for quality and can produce false positives or false negatives. Sync timeout enforcement cannot kill an already-running Python worker thread. The current benchmark cannot support claims about factuality, generalization, production reliability, or superiority over a single call.
