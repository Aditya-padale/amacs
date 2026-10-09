"""Report generation for the offline benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def write_report(
    records: List[Dict[str, Any]],
    output_dir: Path,
    title: str = "AMACS Benchmark",
    filename: str = "fixture.json",
) -> Dict[str, Any]:
    """Write raw JSON and a markdown summary without inventing missing metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / filename
    raw_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        f"# {title}",
        "",
        (
            "All values below come from the recorded offline run. The fixture uses the deterministic "
            "stub provider; it is not evidence of quality against a real LLM."
        ),
        "",
        "| system | runs | success mean | success std | success CI95 | recovery rate | latency mean (s) | latency std | latency CI95 | cost mean | tokens mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        row = dict(record)
        row.setdefault("success_mean", row.get("success_rate", 0.0))
        row.setdefault("success_std", 0.0)
        row.setdefault("success_ci95", 0.0)
        row.setdefault("latency_mean", row.get("mean_latency", 0.0))
        row.setdefault("latency_std", 0.0)
        row.setdefault("latency_ci95", 0.0)
        row.setdefault("cost_mean", 0.0)
        row.setdefault("tokens_mean", 0.0)
        report_lines.append(
            "| {system} | {runs} | {success_mean:.3f} | {success_std:.3f} | {success_ci95:.3f} | "
            "{recovery_rate:.3f} | {latency_mean:.6f} | {latency_std:.6f} | {latency_ci95:.6f} | "
            "{cost_mean:.6f} | {tokens_mean:.1f} |".format(**row)
        )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        for metric, plot_filename, ylabel in (
            ("success_rate", "success_rate_vs_fault_rate.png", "Success rate"),
            ("mean_latency", "latency_vs_fault_rate.png", "Mean latency (s)"),
        ):
            plt.figure(figsize=(6, 4))
            for adaptive in (False, True):
                points = [
                    record for record in records
                    if record.get("adaptive") == adaptive and record.get("system") == "amacs"
                    and "fault_rate" in record
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
            plt.savefig(output_dir / plot_filename)
            plt.close()
    except ImportError:
        (output_dir / "plots-unavailable.txt").write_text(
            "matplotlib is optional and was not installed; no plots were generated.\n",
            encoding="utf-8",
        )
    return {"records": records, "report": str(output_dir / "report.md")}
