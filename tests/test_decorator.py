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

    def test_decorator_last_result_and_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        @amacs(max_agents=2, verbose=True)
        def inspectable_task(topic: str) -> str:
            return f"Inspect {topic}"

        res = inspectable_task("Quantum Computing")
        assert isinstance(res, str)
        assert inspectable_task.last_result is not None
        assert len(inspectable_task.last_result.agent_results) > 0
        assert len(inspectable_task.last_result.communication_log) > 0

        captured = capsys.readouterr()
        assert "Agent Response" in captured.out or "AGENT RESPONSES" in captured.out
        assert "INTER-AGENT COMMUNICATION LOG" in captured.out

    def test_decorator_return_details(self) -> None:
        @amacs(max_agents=2, return_details=True)
        def detailed_task(topic: str) -> str:
            return f"Detailed {topic}"

        res = detailed_task("GenAI")
        assert hasattr(res, "agent_results")
        assert hasattr(res, "communication_log")
        assert len(res.agent_results) > 0
        assert len(res.communication_log) > 0
        assert str(res) == res.final_output


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

    @pytest.mark.asyncio
    async def test_async_decorator_return_details(self) -> None:
        @amacs(max_agents=2, return_details=True)
        async def async_detailed(topic: str) -> str:
            return f"Async {topic}"

        res = await async_detailed("Deep Learning")
        assert hasattr(res, "agent_results")
        assert hasattr(res, "communication_log")
        assert len(res.agent_results) > 0
        assert len(res.communication_log) > 0
        assert str(res) == res.final_output

