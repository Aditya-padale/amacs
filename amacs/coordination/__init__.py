"""AMACS Multi-Agent Coordination Protocols."""

from __future__ import annotations

from amacs.coordination.base import Coordinator
from amacs.coordination.debate import DebateCoordinator
from amacs.coordination.manager_worker import ManagerWorkerCoordinator

__all__ = [
    "Coordinator",
    "DebateCoordinator",
    "ManagerWorkerCoordinator",
]
