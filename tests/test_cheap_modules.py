"""Unit tests for uncovered modules: cache, tracing, shims, vector_db."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from amacs.aggregation import Aggregator as DirectAggregator
from amacs.aggregation.aggregator import Aggregator as AggregatorShim
from amacs.cache import ResponseCache
from amacs.communication import CommunicationBus as DirectBus
from amacs.communication import LogEntry as DirectLogEntry
from amacs.communication.bus import CommunicationBus as BusShim
from amacs.communication.bus import LogEntry as LogEntryShim
from amacs.exceptions import AMACSError
from amacs.integrations.llm_providers import LLMResponse
from amacs.tracing import Tracer

# ── ResponseCache Tests ───────────────────────────────────────────────────────

def test_response_cache_enabled() -> None:
    cache = ResponseCache(enabled=True)
    assert cache.enabled is True

    class Msg:
        content = "hello"

    key = ResponseCache.compute_key([Msg(), "world"], model="gpt-4", temperature=0.5)
    assert isinstance(key, str)
    assert len(key) == 64  # sha256 hex string

    assert cache.get(key) is None

    resp = LLMResponse(content="answer", model="gpt-4", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    cache.set(key, resp)
    assert cache.get(key) == resp

    cache.clear()
    assert cache.get(key) is None


def test_response_cache_disabled() -> None:
    cache = ResponseCache(enabled=False)
    assert cache.enabled is False

    key = ResponseCache.compute_key(["hello"], model="gpt-4")
    resp = LLMResponse(content="answer", model="gpt-4", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    cache.set(key, resp)
    assert cache.get(key) is None


# ── Tracer Tests ─────────────────────────────────────────────────────────────

def test_tracer_spans_and_summary() -> None:
    tracer = Tracer()
    span1 = tracer.start_span("step1", "wave_execution", {"meta": 1})
    assert span1.duration_seconds == 0.0

    tracer.end_span(span1, {"extra": 2})
    assert span1.end_time is not None
    assert span1.duration_seconds >= 0.0
    assert span1.metadata["extra"] == 2

    span2 = tracer.start_span("step2", "agent_run")
    tracer.end_span(span2)

    summary = tracer.summary()
    assert summary["total_spans"] == 2
    assert "total_duration_seconds" in summary
    assert len(summary["spans"]) == 2
    assert summary["spans"][0]["name"] == "step1"
    assert summary["spans"][0]["type"] == "wave_execution"


def test_tracer_serialization_and_events() -> None:
    events = []
    tracer = Tracer(on_event=events.append)
    root = tracer.start_span("pipeline", "stage")
    child = tracer.start_span("task_0", "task", parent=root)
    tracer.end_span(child, {"success": True})
    tracer.end_span(root)

    trace_json = tracer.to_json()
    assert '"parent": "pipeline"' in trace_json
    assert "task_0 (task)" in tracer.to_mermaid()
    assert [event["event"] for event in events] == [
        "span_started", "span_started", "span_finished", "span_finished"
    ]


# ── Shim Tests ───────────────────────────────────────────────────────────────

def test_reexport_shims() -> None:
    assert AggregatorShim is DirectAggregator
    assert BusShim is DirectBus
    assert LogEntryShim is DirectLogEntry


# ── VectorStore Tests ─────────────────────────────────────────────────────────

def test_pinecone_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pinecone", None)
    from amacs.integrations.vector_db import PineconeStore

    with pytest.raises(AMACSError, match="pinecone-client not installed"):
        PineconeStore("test-index")


def test_pinecone_store_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_pinecone = MagicMock()
    mock_index = MagicMock()
    mock_pinecone.Pinecone.return_value.Index.return_value = mock_index

    # Mock response for query
    match1 = MagicMock(id="id1", score=0.9, metadata={"text": "a"})
    mock_index.query.return_value = MagicMock(matches=[match1])

    monkeypatch.setitem(sys.modules, "pinecone", mock_pinecone)
    from amacs.integrations.vector_db import PineconeStore

    store = PineconeStore("test-index", api_key="fake")
    store.upsert([{"id": "id1", "values": [0.1, 0.2]}])
    mock_index.upsert.assert_called_once_with(vectors=[{"id": "id1", "values": [0.1, 0.2]}])

    results = store.query([0.1, 0.2], top_k=1)
    assert len(results) == 1
    assert results[0] == {"id": "id1", "score": 0.9, "metadata": {"text": "a"}}


def test_faiss_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faiss", None)
    from amacs.integrations.vector_db import FAISSStore

    with pytest.raises(AMACSError, match="faiss-cpu not installed"):
        FAISSStore(dimension=128)


def test_faiss_store_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_faiss = MagicMock()
    mock_np = MagicMock()

    mock_index = MagicMock()
    mock_faiss.IndexFlatL2.return_value = mock_index

    # Mock faiss query output (distances, indices)
    mock_index.search.return_value = ([[0.05, 0.1]], [[0, -1]])
    mock_np.array.side_effect = lambda val, dtype=None: val

    monkeypatch.setitem(sys.modules, "faiss", mock_faiss)
    monkeypatch.setitem(sys.modules, "numpy", mock_np)
    from amacs.integrations.vector_db import FAISSStore

    store = FAISSStore(dimension=2)
    store.upsert([{"values": [0.1, 0.2], "metadata": {"tag": "test"}}])
    assert mock_index.add.called

    results = store.query([0.1, 0.2], top_k=2)
    assert len(results) == 1
    assert results[0]["id"] == 0
    assert results[0]["score"] == 0.05
    assert results[0]["metadata"] == {"tag": "test"}
