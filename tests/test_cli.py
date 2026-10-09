"""Tests for the AMACS CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from amacs.benchmarks.benchmark_suite import run_benchmark
from amacs.cli import main


class TestCLI:
    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "amacs" in captured.out
        assert "0.1.0" in captured.out

    def test_run_command(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("x = 42\n")
            f.flush()
            script_path = f.name

        try:
            main(["run", script_path])
        finally:
            Path(script_path).unlink(missing_ok=True)

    def test_benchmark_command(self) -> None:
        metrics = run_benchmark()
        assert "speed_latency_seconds" in metrics

        main(["benchmark"])

    def test_no_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([])
        captured = capsys.readouterr()
        assert "usage: amacs" in captured.out or "AMACS" in captured.out
