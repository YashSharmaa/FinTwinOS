"""Twin state snapshots: content-hashed serialisation and structural diffs.

A snapshot captures the *state* of the twin — every entity with its
attributes, every typed relationship, per-series metadata for the time series
store, and every document's text + metadata — as a plain dict with a SHA-256
content hash over its canonical JSON form. Two twins built from the same seed
produce byte-identical hashes, which is the backbone of FinTwinOS's determinism
guarantees: demo data generation, episode replay and counterfactual rebuilds
are all verified by comparing snapshot hashes.

:func:`diff_snapshots` compares two snapshots structurally, reporting added,
removed and changed entities, relationships and series.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from fintwinos.core.interfaces import DocumentStore, GraphStore, TimeSeriesStore, TwinRuntime

SNAPSHOT_VERSION = 2


def _canonical_json(value: Any) -> str:
    """Deterministic JSON used both for hashing and change comparison."""
    return json.dumps(value, sort_keys=True, default=str)


def take_snapshot(
    graph: GraphStore,
    timeseries: TimeSeriesStore,
    documents: DocumentStore | None = None,
) -> dict[str, Any]:
    """Serialise graph + time-series + document state into a content-hashed snapshot.

    The graph store must expose ``entities()`` and ``relationships()``
    enumerators (as :class:`fintwinos.twin_core.graph.NetworkXGraphStore`
    does) in addition to the ``GraphStore`` protocol. The document store is
    optional and is included only when it exposes ``doc_ids()`` (as
    :class:`fintwinos.twin_core.documents.TfidfDocumentStore` does).

    Returns a dict with:

    - ``version`` — snapshot schema version;
    - ``entities`` — ``{entity_key: attributes}``;
    - ``relationships`` — ``{"src|kind|dst": attributes}``;
    - ``timeseries`` — ``{series_key: {points, start, end, first, last}}``;
    - ``documents`` — ``{doc_id: {text, metadata}}`` (empty if no store);
    - ``hash`` — SHA-256 over the canonical JSON of everything above.
    """
    entities_fn = getattr(graph, "entities", None)
    relationships_fn = getattr(graph, "relationships", None)
    if entities_fn is None or relationships_fn is None:
        raise TypeError(
            "take_snapshot requires a graph store exposing entities() and "
            f"relationships(); got {type(graph).__name__}"
        )

    entities: dict[str, Any] = {}
    for ref, attributes in entities_fn():
        entities[ref.key()] = json.loads(_canonical_json(attributes))

    relationships: dict[str, Any] = {}
    for src, dst, kind, attributes in relationships_fn():
        edge_key = f"{src.key()}|{kind}|{dst.key()}"
        relationships[edge_key] = json.loads(_canonical_json(attributes))

    series_meta: dict[str, Any] = {}
    for series_key in timeseries.keys():
        points = timeseries.window(series_key)
        if not points:
            series_meta[series_key] = {"points": 0}
            continue
        first_ts, first_value = points[0]
        last_ts, last_value = points[-1]
        series_meta[series_key] = {
            "points": len(points),
            "start": first_ts.isoformat(),
            "end": last_ts.isoformat(),
            "first": first_value,
            "last": last_value,
        }

    documents_meta: dict[str, Any] = {}
    doc_ids_fn = getattr(documents, "doc_ids", None) if documents is not None else None
    if doc_ids_fn is not None:
        for doc_id in doc_ids_fn():
            doc = documents.get(doc_id)  # type: ignore[union-attr]
            if doc is None:
                continue
            documents_meta[str(doc_id)] = {
                "text": doc.get("text", ""),
                "metadata": json.loads(_canonical_json(doc.get("metadata", {}))),
            }

    body = {
        "version": SNAPSHOT_VERSION,
        "entities": entities,
        "relationships": relationships,
        "timeseries": series_meta,
        "documents": documents_meta,
    }
    content_hash = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    return {**body, "hash": content_hash}


def snapshot_runtime(runtime: TwinRuntime) -> dict[str, Any]:
    """Convenience wrapper: snapshot a :class:`TwinRuntime`'s graph, series and docs."""
    return take_snapshot(runtime.graph, runtime.timeseries, runtime.documents)


def _diff_keyed(old: dict[str, Any], new: dict[str, Any]) -> dict[str, list[str]]:
    """Added/removed/changed keys between two ``{key: value}`` sections."""
    old_keys, new_keys = set(old), set(new)
    return {
        "added": sorted(new_keys - old_keys),
        "removed": sorted(old_keys - new_keys),
        "changed": sorted(
            key
            for key in old_keys & new_keys
            if _canonical_json(old[key]) != _canonical_json(new[key])
        ),
    }


def diff_snapshots(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Structural difference between two snapshots produced by :func:`take_snapshot`.

    Returns ``{"entities": {...}, "relationships": {...}, "timeseries": {...},
    "identical": bool}`` where each section holds sorted ``added`` /
    ``removed`` / ``changed`` key lists, and ``identical`` is True iff the two
    content hashes match.
    """
    return {
        "entities": _diff_keyed(old.get("entities", {}), new.get("entities", {})),
        "relationships": _diff_keyed(old.get("relationships", {}), new.get("relationships", {})),
        "timeseries": _diff_keyed(old.get("timeseries", {}), new.get("timeseries", {})),
        "documents": _diff_keyed(old.get("documents", {}), new.get("documents", {})),
        "identical": bool(old.get("hash")) and old.get("hash") == new.get("hash"),
    }
