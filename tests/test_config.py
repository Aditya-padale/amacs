"""Tests for config validation."""

from __future__ import annotations

import pytest

from amacs.config import AMACSConfig, Strategy, build_config
from amacs.exceptions import ConfigurationError


class TestConfig:
    def test_default_config(self) -> None:
        cfg = build_config()
        assert cfg.max_agents == 4
        assert cfg.strategy == Strategy.PERFORMANCE
        assert cfg.adaptive is False
        assert cfg.retry_limit == 3
        assert cfg.timeout == 120.0

    def test_custom_config(self) -> None:
        cfg = build_config(max_agents=8, strategy="speed", adaptive=True, retry_limit=5)
        assert cfg.max_agents == 8
        assert cfg.strategy == Strategy.SPEED
        assert cfg.adaptive is True
        assert cfg.retry_limit == 5

    def test_strategy_coercion(self) -> None:
        cfg = build_config(strategy="PERFORMANCE")
        assert cfg.strategy == Strategy.PERFORMANCE

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ConfigurationError, match="Invalid strategy"):
            build_config(strategy="invalid")

    def test_invalid_max_agents(self) -> None:
        with pytest.raises(ConfigurationError):
            build_config(max_agents=0)

    def test_invalid_max_agents_too_high(self) -> None:
        with pytest.raises(ConfigurationError):
            build_config(max_agents=100)

    def test_frozen_config(self) -> None:
        cfg = build_config()
        with pytest.raises(Exception):
            cfg.max_agents = 10  # type: ignore[misc]
