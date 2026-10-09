"""Tests proving parity between sync (run) and async (arun) execution paths in Pipeline."""

from __future__ import annotations

from amacs.config import AMACSConfig
from amacs.integrations.llm_providers import FakeProvider
from amacs.pipeline import Pipeline


def sample_task(topic: str) -> str:
    return f"Research on {topic}"


async def async_sample_task(topic: str) -> str:
    return f"Research on {topic}"


def test_pipeline_sync_and_async_parity() -> None:
    config = AMACSConfig(max_agents=2, return_details=True)
    provider_sync = FakeProvider(default_response="Simulated research output for {user}")
    provider_async = FakeProvider(default_response="Simulated research output for {user}")

    pipeline_sync = Pipeline(config=config, provider=provider_sync)
    pipeline_async = Pipeline(config=config, provider=provider_async)

    res_sync = pipeline_sync.run(sample_task, ("AI Ethics",), {})

    import asyncio

    res_async = asyncio.run(
        pipeline_async.arun(async_sample_task, ("AI Ethics",), {})
    )

    # Assert equivalent outcomes
    assert res_sync.final_output == res_async.final_output
    assert len(res_sync.agent_results) == len(res_async.agent_results)
    assert len(res_sync.sub_tasks) == len(res_async.sub_tasks)
    assert len(res_sync.communication_log) == len(res_async.communication_log)

    for r_sync, r_async in zip(res_sync.agent_results, res_async.agent_results):
        assert r_sync.sub_task_id == r_async.sub_task_id
        assert r_sync.agent_name == r_async.agent_name
        assert r_sync.success == r_async.success
