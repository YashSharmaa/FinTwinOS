"""Shared fixtures for the twin_sim test suite.

Provides a minimal in-memory :class:`TwinRuntime` so the simulators can be tested
without depending on the twin_core module (built in parallel). The fakes implement
just enough of the structural protocols in ``fintwinos.core.interfaces``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

import pytest

from fintwinos.core.audit import AuditTrail
from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import EntityRef, EventEnvelope


class FakeGraph:
    """Tiny in-memory graph store satisfying the GraphStore protocol."""

    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: list[tuple[str, str, str]] = []

    def upsert_entity(self, ref: EntityRef, attributes: dict[str, Any] | None = None) -> None:
        self.entities[ref.key()] = attributes or {}

    def get_entity(self, ref: EntityRef) -> dict[str, Any] | None:
        return self.entities.get(ref.key())

    def add_relationship(
        self, src: EntityRef, dst: EntityRef, kind: str, attributes: dict[str, Any] | None = None
    ) -> None:
        self.relationships.append((src.key(), dst.key(), kind))

    def neighbors(self, ref: EntityRef, kind: str | None = None, depth: int = 1) -> list[EntityRef]:
        return []

    def subgraph(self, refs: Iterable[EntityRef], depth: int = 1) -> Any:
        return None

    def stats(self) -> dict[str, Any]:
        return {"entities": len(self.entities), "relationships": len(self.relationships)}


class FakeTimeSeries:
    """In-memory time-series store satisfying the TimeSeriesStore protocol."""

    def __init__(self) -> None:
        self._series: dict[str, list[tuple[datetime, float]]] = {}

    def append(
        self, series_key: str, ts: datetime, value: float, tags: dict[str, Any] | None = None
    ) -> None:
        self._series.setdefault(series_key, []).append((ts, value))

    def window(
        self, series_key: str, start: datetime | None = None, end: datetime | None = None
    ) -> list[tuple[datetime, float]]:
        return list(self._series.get(series_key, []))

    def latest(self, series_key: str) -> tuple[datetime, float] | None:
        points = self._series.get(series_key)
        return points[-1] if points else None

    def keys(self) -> list[str]:
        return sorted(self._series)


class FakeDocuments:
    """In-memory document store satisfying the DocumentStore protocol."""

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}

    def add(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._docs[doc_id] = {"text": text, "metadata": metadata or {}}

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self._docs.get(doc_id)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        return []

    def count(self) -> int:
        return len(self._docs)


class FakeReplay:
    """In-memory replay engine satisfying the ReplayEngine protocol."""

    def __init__(self) -> None:
        self._episodes: dict[str, list[EventEnvelope]] = {}

    def record(self, envelope: EventEnvelope) -> None:
        self._episodes.setdefault(envelope.episode_id or "default", []).append(envelope)

    def episode(self, episode_id: str) -> list[EventEnvelope]:
        return list(self._episodes.get(episode_id, []))

    def episodes(self) -> list[str]:
        return sorted(self._episodes)

    def replay(self, episode_id: str, handler: Callable[[EventEnvelope], None]) -> int:
        events = self.episode(episode_id)
        for event in events:
            handler(event)
        return len(events)


@pytest.fixture()
def runtime() -> TwinRuntime:
    """A fresh minimal TwinRuntime backed by in-memory fakes."""
    return TwinRuntime(
        graph=FakeGraph(),
        timeseries=FakeTimeSeries(),
        documents=FakeDocuments(),
        replay=FakeReplay(),
        audit=AuditTrail(),
    )
