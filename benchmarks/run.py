"""Run the deterministic fixture benchmark and optional live benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, cast

from amacs.integrations.llm_providers import FlakyProvider, StubProvider

from .report import write_report
from .systems.baselines import single_call

ROOT = Path(__file__).parent
DATASET = ROOT / "datasets" / "fixture.json"
RESULTS = ROOT / "results"


def _load_fixture() -> List[Dict[str, str]]:
    return cast(List[Dict[str, str]], json.loads(DATASET.read_text(encoding="utf-8")))


def _run_cell(fault_rate: float, adaptive: bool, repetitions: int) -> Dict[str, Any]:
    latencies: List[float] = []
    successes = 0
    injected_failures = 0
    recovered_failures = 0
    total = 0
    for repetition in range(repetitions):
        for item in _load_fixture():
            total += 1
            provider = FlakyProvider(
                base_provider=StubProvider(),
                failure_rate=fault_rate,
                seed=1000 + repetition * 100 + total,
            )
            started = time.perf_counter()
            from .systems.baselines import amacs_pipeline

            result = amacs_pipeline(provider, item["task"], adaptive)
            latencies.append(time.perf_counter() - started)
            if item["expected"].lower() in result["output"].lower():
                successes += 1
                injected_failures += provider.failures
                if provider.failures:
                    recovered_failures += provider.failures
            else:
                injected_failures += provider.failures

    return {
        "system": "amacs",
        "fault_rate": fault_rate,
        "adaptive": adaptive,
        "runs": total,
        "success_rate": successes / total if total else 0.0,
        "recovery_rate": (
            recovered_failures / injected_failures if injected_failures else 0.0
        ),
        "injected_failures": injected_failures,
        "mean_latency": statistics.mean(latencies) if latencies else 0.0,
    }


def _run_baseline(repetitions: int) -> Dict[str, Any]:
    latencies: List[float] = []
    successes = 0
    fixture = _load_fixture()
    for _ in range(repetitions):
        for item in fixture:
            provider = StubProvider()
            started = time.perf_counter()
            output = single_call(provider, item["task"])
            latencies.append(time.perf_counter() - started)
            successes += item["expected"].lower() in output.lower()
    total = len(fixture) * repetitions
    return {
        "system": "single-call",
        "fault_rate": 0.0,
        "adaptive": False,
        "runs": total,
        "success_rate": successes / total if total else 0.0,
        "recovery_rate": 0.0,
        "injected_failures": 0,
        "mean_latency": statistics.mean(latencies) if latencies else 0.0,
    }


def run_fixture(repetitions: int = 3) -> Dict[str, Any]:
    """Run the offline cells and write ``benchmarks/results`` artifacts."""
    records: List[Dict[str, Any]] = []
    records.append(_run_baseline(repetitions))
    for fault_rate in (0.0, 0.25, 0.5):
        for adaptive in (False, True):
            records.append(_run_cell(fault_rate, adaptive, repetitions))
    return write_report(records, RESULTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", help="Run the offline fixture.")
    parser.add_argument("--full", action="store_true", help="Reserved for API-key benchmark runs.")
    args = parser.parse_args()
    if args.full:
        raise SystemExit("Full benchmark requires provider credentials and is not run by default.")
    if not args.fixture:
        parser.error("choose --fixture or --full")
    result = run_fixture()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()