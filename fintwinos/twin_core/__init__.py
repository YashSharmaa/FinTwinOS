"""twin_core — the federated twin substrate of FinTwinOS.

Provides the concrete stores behind the :mod:`fintwinos.core.interfaces`
protocols (graph, time series, documents, replay), deterministic entity
resolution, the single envelope ingestion path, content-hashed state
snapshots, a seeded synthetic demo book, and the contracted
:func:`build_runtime` entry point that assembles them all.

FinTwinOS is MIT-licensed, created by Yash Sharma
(https://www.linkedin.com/in/yashsharmaa/).
"""

from fintwinos.twin_core.demo_data import build_demo_book, load_demo_data
from fintwinos.twin_core.documents import TfidfDocumentStore
from fintwinos.twin_core.graph import NetworkXGraphStore
from fintwinos.twin_core.ingestion import TwinIngestor
from fintwinos.twin_core.replay import JsonlReplayEngine
from fintwinos.twin_core.resolution import EntityResolver, ResolutionResult
from fintwinos.twin_core.runtime import build_runtime
from fintwinos.twin_core.snapshot import diff_snapshots, snapshot_runtime, take_snapshot
from fintwinos.twin_core.timeseries import PandasTimeSeriesStore

__all__ = [
    "EntityResolver",
    "JsonlReplayEngine",
    "NetworkXGraphStore",
    "PandasTimeSeriesStore",
    "ResolutionResult",
    "TfidfDocumentStore",
    "TwinIngestor",
    "build_demo_book",
    "build_runtime",
    "diff_snapshots",
    "load_demo_data",
    "snapshot_runtime",
    "take_snapshot",
]
