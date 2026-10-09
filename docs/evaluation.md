# Evaluation

## What Is Measured

The repository contains an offline fixture harness in `benchmarks/run.py`. It uses a deterministic stub provider and `FlakyProvider` to inject seeded provider failures. Each cell runs the same nine fixture cases across three repetitions. A case is successful when the expected fixture string occurs in the output. Recovery is the fraction of injected failures followed by a successful expected-string result.

This methodology measures orchestration plumbing, retry/recovery behavior, and wall-clock overhead. It does not measure factuality, helpfulness, task correctness beyond the fixture string, cost quality, or user preference.

## Recorded Fixture Result

The checked-in report is `benchmarks/results/report.md`. Its measured cells are:

- Single-call baseline at fault rate 0.00: success `1.000`, mean latency `0.000008` seconds.
- AMACS, non-adaptive, fault rate 0.00: success `1.000`, mean latency `0.059296` seconds.
- AMACS, adaptive, fault rate 0.00: success `1.000`, mean latency `0.014351` seconds.
- AMACS, non-adaptive, fault rates 0.25 and 0.50: success `1.000`, recovery `1.000`.
- AMACS, adaptive, fault rates 0.25 and 0.50: success `1.000`, recovery `1.000`; mean latency was `0.459288` and `1.183298` seconds respectively.

These values are fixture observations, not a claim that adaptive execution is faster or better in general.

## Reproduction

From the repository root:

```bash
make bench-fixture
cat benchmarks/results/report.md
```

The default verification command is offline:

```bash
make verify
```

A full benchmark command exists, but it exits with a credentials-required message. Live provider experiments should be added as explicitly marked tests with pinned datasets, provider/model identifiers, API cost accounting, and recorded seeds.

## Required Future Study

A defensible quality study needs a held-out task set, a single-call baseline using the same provider/model, fixed prompts and budgets, repeated seeds, blinded human or task-specific grading, cost and latency distributions, and ablations for adaptation, decomposition, coordination, caching, and validation. No such study is included here.
