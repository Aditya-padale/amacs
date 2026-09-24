"""Tests for the AMACS CLI."""

from __future__ import annotations

import pytest

from amacs.cli import main


class TestCLI:
    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "amacs" in captured.out
        assert "0.1.0" in captured.out
