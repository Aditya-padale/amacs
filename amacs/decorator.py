"""The ``@amacs`` decorator — the single public entry point for the framework.

Wraps any sync or async function with the full AMACS orchestration pipeline:

    @amacs(max_agents=8, strategy="performance")
    def research_task(topic: str) -> str:
        return f"Research on {topic}"

    result = research_task("Renewable Energy")
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Protocol, cast

from amacs.config import build_config
from amacs.pipeline import Pipeline
from amacs.results import AMACSResult

logger = logging.getLogger("amacs.decorator")


class AMACSCallable(Protocol):
    """Protocol for functions wrapped by @amacs decorator."""
    last_result: AMACSResult | None
    def __call__(self, *args: Any, **kw: Any) -> Any: ...


def amacs(**kwargs: Any) -> Callable[[Callable[..., Any]], AMACSCallable]:
    """Decorator factory that wraps a function with multi-agent orchestration."""
    config = build_config(**kwargs)

    def decorator(func: Callable[..., Any]) -> AMACSCallable:
        pipeline = Pipeline(config)
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kw: Any) -> Any:
                res: AMACSResult = await pipeline.arun(func, args, kw)
                async_wrapper.last_result = res  # type: ignore[attr-defined]
                return res if config.return_details else res.final_output

            async_wrapper.last_result = None  # type: ignore[attr-defined]
            return cast(AMACSCallable, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kw: Any) -> Any:
                res: AMACSResult = pipeline.run(func, args, kw)
                sync_wrapper.last_result = res  # type: ignore[attr-defined]
                return res if config.return_details else res.final_output

            sync_wrapper.last_result = None  # type: ignore[attr-defined]
            return cast(AMACSCallable, sync_wrapper)

    return decorator


