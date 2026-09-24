"""AMACS CLI — basic command-line interface.

Usage:
    amacs --version
    amacs run <script>
"""

from __future__ import annotations

import argparse
import runpy
import sys


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

    args = parser.parse_args(argv)

    if args.command == "run":
        # Execute the script in __main__ namespace so @amacs decorators work
        sys.argv = [args.script]
        runpy.run_path(args.script, run_name="__main__")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
