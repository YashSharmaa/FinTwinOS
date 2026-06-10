"""Structural contracts between FinTwinOS modules.

The twin is federated: ``twin_core`` implements the stores, ``twin_sim`` implements
``Simulator``s, ``connectors`` emit ``EventEnvelope`` streams, ``tools`` builds the
typed registry on top of a ``TwinRuntime``, and ``agents`` orchestrate over that.
Every module codes against these Protocols, never against each other's internals.

Canonical entry points (see CONTRACTS.md at the repo root):

- ``fintwinos.twin_core.runtime.build_runtime(seed=7, with_demo_data=True) -> TwinRuntime``
- ``fintwinos.twin_sim.register_all(runtime) -> None``
- ``fintwinos.tools.catalog.build_default_registry(runtime) -> ToolRegistry``
- ``fintwinos.agents.runtime.handle_case(case_id, objective, runtime, registry, llm) -> dict``
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from fintwinos.core.audit import AuditTrail
from fintwinos.core.types import EntityRef, EventEnvelope, Scenario, SimulationResult


@runtime_checkable
class GraphStore(Protocol):
    """Entities and relationships: customers, accounts, instruments, cases, rings."""

    def upsert_entity(self, ref: EntityRef, attributes: dict[str, Any] | None = None) -> None: ...

    def get_entity(self, ref: EntityRef) -> dict[str, Any] | None: ...

    def add_relationship(
        self, src: EntityRef, dst: EntityRef, kind: str, attributes: dict[str, Any] | None = None
    ) -> None: ...

    def neighbors(self, ref: EntityRef, kind: str | None = None, depth: int = 1) -> list[EntityRef]: ...

    def subgraph(self, refs: Iterable[EntityRef], depth: int = 1) -> Any: ...

    def stats(self) -> dict[str, Any]: ...


@runtime_checkable
class TimeSeriesStore(Protocol):
    """Market and operational dynamics keyed by series name."""

    def append(self, series_key: str, ts: datetime, value: float, tags: dict[str, Any] | None = None) -> None: ...

    def window(self, series_key: str, start: datetime | None = None, end: datetime | None = None) -> list[tuple[datetime, float]]: ...

    def latest(self, series_key: str) -> tuple[datetime, float] | None: ...

    def keys(self) -> list[str]: ...


@runtime_checkable
class DocumentStore(Protocol):
    """Filings, internal procedures, customer artefacts, supervisory rules."""

    def add(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None: ...

    def get(self, doc_id: str) -> dict[str, Any] | None: ...

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]: ...

    def count(self) -> int: ...


@runtime_checkable
class ReplayEngine(Protocol):
    """Records envelopes into episodes and replays them for counterfactual analysis."""

    def record(self, envelope: EventEnvelope) -> None: ...

    def episode(self, episode_id: str) -> list[EventEnvelope]: ...

    def episodes(self) -> list[str]: ...

    def replay(self, episode_id: str, handler: Callable[[EventEnvelope], None]) -> int: ...


@runtime_checkable
class Simulator(Protocol):
    """Every simulator exposes point estimates, confidence intervals and calibration."""

    name: str

    def run(self, scenario: Scenario, *, seed: int | None = None) -> SimulationResult: ...

    def calibration_report(self) -> dict[str, Any]: ...


@runtime_checkable
class EventIngestor(Protocol):
    def ingest(self, envelope: EventEnvelope) -> None: ...


@runtime_checkable
class Connector(Protocol):
    """Adapters from enterprise/external sources into EventEnvelope streams."""

    name: str

    def stream(self) -> AsyncIterator[EventEnvelope]: ...


@dataclass
class TwinRuntime:
    """The assembled federated twin handed to tools, agents, evals and demos."""

    graph: GraphStore
    timeseries: TimeSeriesStore
    documents: DocumentStore
    replay: ReplayEngine
    audit: AuditTrail
    simulators: dict[str, Simulator] = field(default_factory=dict)
    ingestor: EventIngestor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def register_simulator(self, simulator: Simulator) -> None:
        self.simulators[simulator.name] = simulator
