"""AMACS Benchmark Suite — evaluates latency, throughput, and adaptation efficiency.

Provides synthetic benchmark workloads to measure:
- Speed (latency overhead of DAG decomposition + scheduling)
- Multi-agent coordination efficacy
- Real-time inter-wave adaptation overhead
"""

from __future__ import annotations

import time
from typing import Any, Dict

from amacs import amacs
from amacs.results import AMACSResult


def run_benchmark() -> Dict[str, Any]:
    """Run standard benchmark suite and report performance metrics."""

    @amacs(max_agents=4, strategy="speed")
    def benchmark_speed_task(topic: str) -> str:
        return f"Speed benchmark for {topic}"

    @amacs(max_agents=4, strategy="performance", adaptive=True, return_details=True)
    def benchmark_adaptive_task(topic: str) -> AMACSResult:
        return f"Adaptive benchmark for {topic}"

    # 1. Benchmark speed execution
    start = time.time()
    res_speed = benchmark_speed_task("Quantum Computing")
    speed_latency = time.time() - start

    # 2. Benchmark adaptive execution
    start = time.time()
    res_adaptive: AMACSResult = benchmark_adaptive_task("AI Ethics")
    adaptive_latency = time.time() - start

    metrics = {
        "speed_latency_seconds": round(speed_latency, 4),
        "adaptive_latency_seconds": round(adaptive_latency, 4),
        "sub_task_count": len(res_adaptive.sub_tasks),
        "adaptation_events": len(res_adaptive.adaptation_events),
    }

    print("\n================ 📊 AMACS BENCHMARK SUMMARY ================")
    print(f"Speed Task Latency:    {metrics['speed_latency_seconds']}s")
    print(f"Adaptive Task Latency: {metrics['adaptive_latency_seconds']}s")
    print(f"Sub-tasks Generated:   {metrics['sub_task_count']}")
    print(f"Adaptation Events:     {metrics['adaptation_events']}")
    print("============================================================\n")

    return metrics


if __name__ == "__main__":
    run_benchmark()
