"""Protocol conformance: twin_core implementations satisfy the core contracts."""

from __future__ import annotations

from fintwinos.core.audit import AuditTrail
from fintwinos.core.interfaces import (
    DocumentStore,
    EventIngestor,
    GraphStore,
    ReplayEngine,
    TimeSeriesStore,
    TwinRuntime,
)
from fintwinos.twin_core import (
    JsonlReplayEngine,
    NetworkXGraphStore,
    PandasTimeSeriesStore,
    TfidfDocumentStore,
    TwinIngestor,
)
from fintwinos.twin_core.runtime import build_runtime


def test_stores_satisfy_runtime_checkable_protocols():
    graph = NetworkXGraphStore()
    timeseries = PandasTimeSeriesStore()
    documents = TfidfDocumentStore()
    replay = JsonlReplayEngine()
    ingestor = TwinIngestor(
        graph=graph,
        timeseries=timeseries,
        documents=documents,
        replay=replay,
        audit=AuditTrail(),
    )
    assert isinstance(graph, GraphStore)
    assert isinstance(timeseries, TimeSeriesStore)
    assert isinstance(documents, DocumentStore)
    assert isinstance(replay, ReplayEngine)
    assert isinstance(ingestor, EventIngestor)


def test_build_runtime_returns_wired_twin_runtime(empty_runtime):
    runtime = empty_runtime
    assert isinstance(runtime, TwinRuntime)
    assert isinstance(runtime.graph, GraphStore)
    assert isinstance(runtime.timeseries, TimeSeriesStore)
    assert isinstance(runtime.documents, DocumentStore)
    assert isinstance(runtime.replay, ReplayEngine)
    assert isinstance(runtime.ingestor, EventIngestor)
    assert isinstance(runtime.audit, AuditTrail)
    assert runtime.metadata == {"seed": 7, "demo": False}
    assert runtime.simulators == {}


def test_build_runtime_metadata_reflects_arguments(demo_runtime):
    assert demo_runtime.metadata == {"seed": 7, "demo": True}


def test_build_runtime_default_signature():
    """The contracted entry point accepts (seed, with_demo_data) with defaults."""
    runtime = build_runtime(seed=3, with_demo_data=False)
    assert runtime.metadata == {"seed": 3, "demo": False}
    assert runtime.graph.stats()["entities"] == 0
