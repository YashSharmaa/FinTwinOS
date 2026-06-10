"""Assembly of the federated twin: the contracted ``build_runtime`` entry point.

This is the canonical way every other FinTwinOS module obtains a twin
(CONTRACTS.md): tools build their registries on top of it, agents reason over
it, simulators register into it, evals and demos rebuild it deterministically
from a seed.

>>> from fintwinos.twin_core.runtime import build_runtime
>>> runtime = build_runtime(seed=7, with_demo_data=True)
>>> runtime.graph.stats()["entities"] > 0
True
"""

from __future__ import annotations

from pathlib import Path

from fintwinos.core.audit import AuditTrail
from fintwinos.core.interfaces import TwinRuntime
from fintwinos.twin_core.demo_data import load_demo_data
from fintwinos.twin_core.documents import TfidfDocumentStore
from fintwinos.twin_core.graph import NetworkXGraphStore
from fintwinos.twin_core.ingestion import TwinIngestor
from fintwinos.twin_core.replay import JsonlReplayEngine
from fintwinos.twin_core.resolution import EntityResolver
from fintwinos.twin_core.timeseries import PandasTimeSeriesStore


def build_runtime(
    seed: int = 7,
    with_demo_data: bool = True,
    data_dir: Path | str | None = None,
) -> TwinRuntime:
    """Construct a fully wired :class:`TwinRuntime`.

    Parameters
    ----------
    seed:
        Drives the demo-data generator (``numpy.random.default_rng(seed)``);
        the same seed always yields a twin with an identical snapshot hash.
    with_demo_data:
        When True (the default), the synthetic demo book is generated and
        ingested through the normal ingestion path, populating the graph,
        time-series, document stores and the ``"demo"`` replay episode.
    data_dir:
        Optional directory for persistence. When given, replay episodes are
        written as JSONL under ``<data_dir>/episodes`` and the audit trail
        under ``<data_dir>/audit.jsonl``; when omitted (the default)
        everything stays in memory.

    Returns
    -------
    TwinRuntime
        With all stores constructed, ``ingestor`` set, and
        ``metadata == {"seed": seed, "demo": with_demo_data}``.
    """
    graph = NetworkXGraphStore()
    timeseries = PandasTimeSeriesStore()
    documents = TfidfDocumentStore()

    if data_dir is not None:
        base = Path(data_dir)
        base.mkdir(parents=True, exist_ok=True)
        replay = JsonlReplayEngine(base / "episodes")
        audit = AuditTrail(base / "audit.jsonl")
    else:
        replay = JsonlReplayEngine()
        audit = AuditTrail()

    resolver = EntityResolver()
    ingestor = TwinIngestor(
        graph=graph,
        timeseries=timeseries,
        documents=documents,
        replay=replay,
        audit=audit,
        resolver=resolver,
    )
    runtime = TwinRuntime(
        graph=graph,
        timeseries=timeseries,
        documents=documents,
        replay=replay,
        audit=audit,
        ingestor=ingestor,
        metadata={"seed": seed, "demo": with_demo_data},
    )
    audit.append(
        "twin_core",
        "twin.runtime.built",
        {"seed": seed, "demo": with_demo_data, "persistent": data_dir is not None},
    )
    if with_demo_data:
        summary = load_demo_data(ingestor, seed=seed)
        audit.append("twin_core", "twin.demo_loaded", summary)
    return runtime
