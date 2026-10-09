"""Report generation for the offline benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def write_report(records: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    """Write raw JSON and a markdown summary without inventing missing metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "fixture.json"
    raw_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        "# AMACS Fixture Benchmark",
        "",
        (
            "All values below come from the recorded offline run. The fixture uses the deterministic "
            "stub provider; it is not evidence of quality against a real LLM."
        ),
        "",
        "| system | fault rate | adaptive | runs | success rate | recovery rate | mean latency (s) |",
        "|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for record in records:
        report_lines.append(
            "| {system} | {fault_rate:.2f} | {adaptive} | {runs} | {success_rate:.3f} | "
            "{recovery_rate:.3f} | {mean_latency:.6f} |".format(**record)
        )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        for metric, filename, ylabel in (
            ("success_rate", "success_rate_vs_fault_rate.png", "Success rate"),
            ("mean_latency", "latency_vs_fault_rate.png", "Mean latency (s)"),
        ):
            plt.figure(figsize=(6, 4))
            for adaptive in (False, True):
                points = [
                    record for record in records
                    if record["adaptive"] == adaptive and record["system"] == "amacs"
                ]
                plt.plot(
                    [point["fault_rate"] for point in points],
                    [point[metric] for point in points],
                    marker="o",
                    label=f"adaptive={adaptive}",
                )
            plt.xlabel("Fault rate")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / filename)
            plt.close()
    except ImportError:
        (output_dir / "plots-unavailable.txt").write_text(
            "matplotlib is optional and was not installed; no plots were generated.\n",
            encoding="utf-8",
        )
    return {"records": records, "report": str(output_dir / "report.md")}