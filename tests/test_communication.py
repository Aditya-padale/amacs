"""Tests for the communication bus."""

from __future__ import annotations

import pytest

from amacs.communication import CommunicationBus


class TestCommunicationBus:
    def test_publish_and_get(self) -> None:
        bus = CommunicationBus()
        bus.publish("key1", "value1", writer="agent_a")
        assert bus.get("key1") == "value1"

    def test_get_default(self) -> None:
        bus = CommunicationBus()
        assert bus.get("missing") is None
        assert bus.get("missing", "fallback") == "fallback"

    def test_snapshot(self) -> None:
        bus = CommunicationBus()
        bus.publish("a", 1, writer="x")
        bus.publish("b", 2, writer="y")
        snap = bus.snapshot()
        assert snap == {"a": 1, "b": 2}

    def test_overwrite_logs_conflict(self) -> None:
        bus = CommunicationBus()
        bus.publish("key", "v1", writer="a")
        bus.publish("key", "v2", writer="b")
        assert bus.get("key") == "v2"
        conflicts = bus.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].writer == "b"

    def test_merge_strategy(self) -> None:
        def merge(old: str, new: str) -> str:
            return f"{old}; {new}"

        bus = CommunicationBus(merge_strategy=merge)
        bus.publish("k", "first", writer="a")
        bus.publish("k", "second", writer="b")
        assert bus.get("k") == "first; second"

    def test_subscribe(self) -> None:
        bus = CommunicationBus()
        received: list[tuple[str, object]] = []
        bus.subscribe("topic", lambda k, v: received.append((k, v)))
        bus.publish("topic", "data", writer="pub")
        assert len(received) == 1
        assert received[0] == ("topic", "data")

    def test_subscribe_wildcard(self) -> None:
        bus = CommunicationBus()
        received: list[tuple[str, object]] = []
        bus.subscribe("*", lambda k, v: received.append((k, v)))
        bus.publish("any_key", "val", writer="pub")
        assert len(received) == 1

    def test_clear(self) -> None:
        bus = CommunicationBus()
        bus.publish("k", "v", writer="a")
        bus.clear()
        assert bus.get("k") is None
        assert bus.get_log() == []

    def test_audit_log(self) -> None:
        bus = CommunicationBus()
        bus.publish("a", 1, writer="x")
        bus.publish("b", 2, writer="y")
        log = bus.get_log()
        assert len(log) == 2
        assert log[0].key == "a"
        assert log[1].key == "b"
