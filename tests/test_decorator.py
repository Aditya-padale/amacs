"""Tests for the @amacs decorator behavior — sync and async wrapping."""

from __future__ import annotations

import asyncio
import pytest

from amacs import amacs
from amacs.config import Strategy


class TestDecoratorSync:
    """Sync decorator tests."""

    def test_decorator_wraps_sync_function(self) -> None:
        @amacs(max_agents=2, strategy="performance")
        def my_task(topic: str) -> str:
            return f"Research on {topic}"

        result = my_task("AI")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decorator_preserves_function_name(self) -> None:
        @amacs()
        def named_function(x: int) -> str:
            return str(x)

        assert named_function.__name__ == "named_function"

    def test_decorator_with_all_strategies(self) -> None:
        for strategy in ["performance", "cost", "speed"]:

            @amacs(strategy=strategy, max_agents=2)
            def task(t: str) -> str:
                return t

            result = task("test")
            assert isinstance(result, str)

    def test_decorator_returns_non_empty_result(self) -> None:
        @amacs(max_agents=2)
        def hello() -> str:
            return "hello world"

        result = hello()
        assert result  # non-empty


class TestDecoratorAsync:
    """Async decorator tests."""

    @pytest.mark.asyncio
    async def test_decorator_wraps_async_function(self) -> None:
        @amacs(max_agents=2, strategy="speed")
        async def async_task(topic: str) -> str:
            return f"Research on {topic}"

        result = await async_task("AI")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_async_decorator_preserves_name(self) -> None:
        @amacs()
        async def async_named() -> str:
            return "async"

        assert async_named.__name__ == "async_named"
