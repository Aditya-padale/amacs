"""AMACS CLI — basic command-line interface.

Usage:
    amacs --version
    amacs run <script>
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``amacs`` CLI."""
    from amacs import __version__

    parser = argparse.ArgumentParser(
        prog="amacs",
        description="AMACS — Adaptive Multi-Agent Coordination System",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"amacs {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # amacs run <script>
    run_parser = subparsers.add_parser("run", help="Run a Python script with AMACS enabled.")
    run_parser.add_argument("script", help="Path to the Python script to execute.")
    run_parser.add_argument("--provider", help="Provider name used by undecorated configs.")
    run_parser.add_argument("--model", help="Model name used by undecorated configs.")
    run_parser.add_argument("--strategy", choices=["performance", "cost", "speed"])

    # amacs bench / benchmark
    subparsers.add_parser("bench", help="Run the offline AMACS benchmark suite.")
    subparsers.add_parser("benchmark", help="Run the AMACS performance benchmark suite.")
    trace_parser = subparsers.add_parser("trace", help="Render a saved AMACS trace JSON file.")
    trace_parser.add_argument("file", help="Path to a trace JSON file.")

    args = parser.parse_args(argv)

    if args.command == "run":
        if args.provider:
            os.environ["AMACS_LLM_PROVIDER"] = args.provider
        if args.model:
            os.environ["AMACS_LLM_MODEL"] = args.model
        if args.strategy:
            os.environ["AMACS_STRATEGY"] = args.strategy
        sys.argv = [args.script]
        namespace = runpy.run_path(args.script, run_name="__main__")
        result = namespace.get("amacs_result")
        if result is not None and hasattr(result, "trace"):
            Path(args.script + ".trace.json").write_text(result.trace.to_json(), encoding="utf-8")
    elif args.command in ("bench", "benchmark"):
        from amacs.benchmarks.benchmark_suite import run_benchmark
        print(json.dumps(run_benchmark(), sort_keys=True))
    elif args.command == "trace":
        data = json.loads(Path(args.file).read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
