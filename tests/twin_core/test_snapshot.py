"""Snapshots: content hashing, structural diffs, demo determinism."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fintwinos.core.types import EntityRef
from fintwinos.twin_core.graph import NetworkXGraphStore
from fintwinos.twin_core.runtime import build_runtime
from fintwinos.twin_core.snapshot import diff_snapshots, snapshot_runtime, take_snapshot
from fintwinos.twin_core.timeseries import PandasTimeSeriesStore

T0 = datetime(2026, 4, 1, tzinfo=UTC)


def _ref(entity_type: str, entity_id: str) -> EntityRef:
    return EntityRef(entity_type=entity_type, entity_id=entity_id)


def _small_twin(extra: bool = False, balance: float = 100.0):
    graph = NetworkXGraphStore()
    timeseries = PandasTimeSeriesStore()
    graph.upsert_entity(_ref("customer", "c1"), {"name": "Acme Corp Ltd"})
    graph.upsert_entity(_ref("account", "a1"), {"balance": balance})
    graph.add_relationship(_ref("customer", "c1"), _ref("account", "a1"), "owns")
    timeseries.append("balance:a1", T0, balance)
    if extra:
        graph.upsert_entity(_ref("customer", "c2"), {"name": "Zephyr Marine"})
        graph.add_relationship(_ref("customer", "c2"), _ref("account", "a1"), "advises")
        timeseries.append("balance:a2", T0, 5.0)
    return graph, timeseries


def test_snapshot_shape_and_hash_stability():
    graph, timeseries = _small_twin()
    snap1 = take_snapshot(graph, timeseries)
    snap2 = take_snapshot(graph, timeseries)
    assert snap1["hash"] == snap2["hash"]
    assert set(snap1) == {"version", "entities", "relationships", "timeseries", "hash"}
    assert "customer:c1" in snap1["entities"]
    assert "customer:c1|owns|account:a1" in snap1["relationships"]
    assert snap1["timeseries"]["balance:a1"]["points"] == 1


def test_identical_content_same_hash_even_for_distinct_stores():
    snap_a = take_snapshot(*_small_twin())
    snap_b = take_snapshot(*_small_twin())
    assert snap_a["hash"] == snap_b["hash"]
    assert diff_snapshots(snap_a, snap_b)["identical"] is True


def test_diff_reports_added_removed_changed():
    base = take_snapshot(*_small_twin())
    grown = take_snapshot(*_small_twin(extra=True))
    changed = take_snapshot(*_small_twin(balance=250.0))

    diff_grow = diff_snapshots(base, grown)
    assert diff_grow["entities"]["added"] == ["customer:c2"]
    assert diff_grow["entities"]["removed"] == []
    assert diff_grow["relationships"]["added"] == ["customer:c2|advises|account:a1"]
    assert diff_grow["timeseries"]["added"] == ["balance:a2"]
    assert diff_grow["identical"] is False

    diff_shrink = diff_snapshots(grown, base)
    assert diff_shrink["entities"]["removed"] == ["customer:c2"]
    assert diff_shrink["relationships"]["removed"] == ["customer:c2|advises|account:a1"]

    diff_change = diff_snapshots(base, changed)
    assert diff_change["entities"]["changed"] == ["account:a1"]
    assert diff_change["timeseries"]["changed"] == ["balance:a1"]
    assert diff_change["entities"]["added"] == []


def test_snapshot_requires_enumerable_graph():
    class Opaque:
        pass

    with pytest.raises(TypeError):
        take_snapshot(Opaque(), PandasTimeSeriesStore())


def test_demo_data_deterministic_same_seed_same_hash(demo_runtime):
    """The contracted determinism guarantee: same seed => identical snapshot hash."""
    rebuilt = build_runtime(seed=7, with_demo_data=True)
    assert snapshot_runtime(rebuilt)["hash"] == snapshot_runtime(demo_runtime)["hash"]


def test_demo_data_seed_changes_hash(demo_runtime):
    other = build_runtime(seed=11, with_demo_data=True)
    snap_demo = snapshot_runtime(demo_runtime)
    snap_other = snapshot_runtime(other)
    assert snap_other["hash"] != snap_demo["hash"]
    diff = diff_snapshots(snap_demo, snap_other)
    assert diff["identical"] is False
    # same universe of ids, different attribute draws
    assert diff["entities"]["changed"]
