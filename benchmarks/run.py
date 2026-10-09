"""Run the deterministic fixture benchmark or a configured full benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

from amacs.integrations.llm_providers import FlakyProvider, LLMProvider, StubProvider, get_provider
from amacs.pricing import calculate_cost

from .report import write_report
from .systems.baselines import (
    b0_single_call,
    b1_sequential,
    b2_amacs,
    b3_debate,
    b4_manager_worker,
    b5_adaptive,
)

ROOT = Path(__file__).parent
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"
SYSTEMS = {"B0": b0_single_call, "B1": b1_sequential, "B2": b2_amacs, "B3": b3_debate, "B4": b4_manager_worker, "B5": b5_adaptive}


def load_dataset(name: str = "fixture") -> List[Dict[str, str]]:
    """Load a checked-in JSON dataset by name or an explicit path."""
    path = Path(name)
    if not path.suffix:
        path = DATASETS / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError(f"Dataset must be a JSON list of task objects: {path}")
    return cast(List[Dict[str, str]], data)


def _load_fixture() -> List[Dict[str, str]]:
    return load_dataset("fixture")


def _summary(values: Sequence[float]) -> Dict[str, float]:
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.mean(values) if values else 0.0, "std": standard_deviation, "ci95": 1.96 * standard_deviation / math.sqrt(len(values)) if len(values) > 1 else 0.0}


def _run_system(provider: LLMProvider, system: str, task: str) -> Dict[str, Any]:
    result: Any = SYSTEMS[system.upper()](provider, task)
    if isinstance(result, str):
        return {"output": result, "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "cost": 0.0}
    execution = result.get("result")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost = 0.0
    if execution:
        for agent_result in execution.agent_results:
            for key in usage:
                usage[key] += agent_result.token_usage.get(key, 0)
            cost += calculate_cost(agent_result.metadata.get("model", "stub"), agent_result.token_usage.get("prompt_tokens", 0), agent_result.token_usage.get("completion_tokens", 0))
    return {"output": result.get("output", ""), "tokens": usage, "cost": cost, "result": execution}


def _run_records(dataset: List[Dict[str, str]], systems: Sequence[str], provider_factory: Any, repetitions: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for system in systems:
        observations: Dict[str, List[float]] = {"latency": [], "success": [], "cost": [], "tokens": []}
        injected = recovered = 0
        for repetition in range(repetitions):
            for index, item in enumerate(dataset):
                provider = provider_factory(repetition, index)
                started = time.perf_counter()
                result = _run_system(provider, system, item["task"])
                observations["latency"].append(time.perf_counter() - started)
                success = item.get("expected", "").lower() in result["output"].lower()
                observations["success"].append(float(success))
                observations["cost"].append(float(result["cost"]))
                observations["tokens"].append(float(result["tokens"]["total_tokens"]))
                failures = int(getattr(provider, "failures", 0))
                injected += failures
                recovered += failures if success else 0
        record: Dict[str, Any] = {"system": system, "runs": len(observations["latency"]), "repetitions": repetitions, "injected_failures": injected, "recovery_rate": recovered / injected if injected else 0.0}
        for metric, values in observations.items():
            summary = _summary(values)
            record[f"{metric}_mean"] = summary["mean"]
            record[f"{metric}_std"] = summary["std"]
            record[f"{metric}_ci95"] = summary["ci95"]
        record["success_rate"] = record["success_mean"]
        record["mean_latency"] = record["latency_mean"]
        record["latency_ci95"] = record["latency_ci95"]
        records.append(record)
    return records


def run_fixture(repetitions: int = 3, systems: Sequence[str] = ("B0", "B1", "B2", "B3", "B4", "B5")) -> Dict[str, Any]:
    """Run an entirely offline, deterministic benchmark."""
    def factory(repetition: int, index: int) -> LLMProvider:
        return FlakyProvider(StubProvider(), failure_rate=0.0, seed=1000 + repetition * 100 + index)
    records = _run_records(_load_fixture(), systems, factory, repetitions)
    return write_report(records, RESULTS, title="AMACS Fixture Benchmark", filename="fixture.json")


def run_full(dataset: str = "reasoning", systems: Sequence[str] = ("B0", "B1", "B2", "B3", "B4", "B5"), provider: str = "stub", repetitions: int = 1) -> Dict[str, Any]:
    """Run selected checked-in tasks using the requested provider."""
    def factory(_repetition: int, _index: int) -> LLMProvider:
        return get_provider(provider)
    records = _run_records(load_dataset(dataset), systems, factory, repetitions)
    return write_report(records, RESULTS, title="AMACS Full Benchmark", filename="full.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", help="Run the deterministic offline fixture.")
    parser.add_argument("--full", action="store_true", help="Run a selected dataset with the selected provider.")
    parser.add_argument("--dataset", default="reasoning", help="Dataset name under benchmarks/datasets or a JSON path.")
    parser.add_argument("--systems", default=",".join(SYSTEMS), help="Comma-separated systems: B0,B1,B2,B3,B4,B5.")
    parser.add_argument("--provider", default="stub", help="Provider name (stub, fake, openai, anthropic, gemini, ollama).")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions per task.")
    parser.add_argument("--ablation", action="append", choices=("adaptation", "coordination", "caching", "strategy"), help="Record an ablation label for this run.")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    selected = [item.strip().upper() for item in args.systems.split(",") if item.strip()]
    unknown = [item for item in selected if item not in SYSTEMS]
    if unknown:
        parser.error(f"unknown systems: {', '.join(unknown)}")
    if args.fixture == args.full:
        parser.error("choose exactly one of --fixture or --full")
    result = run_fixture(args.repetitions, selected) if args.fixture else run_full(args.dataset, selected, args.provider, args.repetitions)
    if args.ablation:
        result["ablations"] = args.ablation
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
