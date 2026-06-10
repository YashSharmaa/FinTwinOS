"""Demo book: coherence, counts, the laundering ring, determinism, persistence."""

from __future__ import annotations

from fintwinos.core.types import EntityRef
from fintwinos.twin_core.demo_data import build_demo_book, load_demo_data
from fintwinos.twin_core.runtime import build_runtime


def _ref(entity_type: str, entity_id: str) -> EntityRef:
    return EntityRef(entity_type=entity_type, entity_id=entity_id)


def test_demo_book_counts(demo_runtime):
    stats = demo_runtime.graph.stats()
    assert stats["entity_types"]["customer"] == 40
    assert stats["entity_types"]["account"] == 60
    assert stats["entity_types"]["instrument"] == 8
    assert stats["entity_types"]["trade"] == 200
    assert stats["entity_types"]["case"] == 6
    assert stats["entity_types"]["alert"] == 2
    assert stats["entity_types"]["policy"] == 5
    # 5 policies + 6 case narratives live in the document store
    assert demo_runtime.documents.count() == 11


def test_demo_price_series(demo_runtime):
    price_keys = [k for k in demo_runtime.timeseries.keys() if k.startswith("price:")]
    assert len(price_keys) == 8
    for key in price_keys:
        points = demo_runtime.timeseries.window(key)
        assert len(points) == 120
        assert all(value > 0 for _, value in points)


def test_demo_cash_ladders(demo_runtime):
    for currency in ("USD", "EUR", "GBP"):
        points = demo_runtime.timeseries.window(f"cash_ladder:{currency}")
        assert len(points) == 30
        tagged = demo_runtime.timeseries.points(f"cash_ladder:{currency}")
        assert tagged[0][2]["currency"] == currency


def test_demo_laundering_ring_is_a_cycle():
    runtime = build_runtime(seed=7, with_demo_data=False)
    summary = load_demo_data(runtime.ingestor, seed=7)
    ring = summary["ring_accounts"]
    assert len(ring) == 6
    assert len(summary["ring_customers"]) == 6
    for i, src in enumerate(ring):
        dst = ring[(i + 1) % len(ring)]
        edge = runtime.graph.get_relationship(
            _ref("account", src), _ref("account", dst), "transfers_to"
        )
        assert edge is not None, f"missing ring leg {src} -> {dst}"
        assert edge["transfer_count"] == 4
        # structuring: every average transfer sits below the 10k reporting threshold
        assert edge["total_amount"] / edge["transfer_count"] < 10_000
        assert edge["total_amount"] / edge["transfer_count"] > 8_000


def test_demo_ring_case_links_ring_entities(demo_runtime):
    case = demo_runtime.graph.get_entity(_ref("case", "case_0001"))
    assert case["kind"] == "aml_alert"
    involved = demo_runtime.graph.neighbors(_ref("case", "case_0001"), kind="involves")
    keys = {ref.key() for ref in involved}
    assert sum(k.startswith("account:") for k in keys) == 6
    assert sum(k.startswith("customer:") for k in keys) == 6


def test_demo_positions_consistent_with_trades(demo_runtime):
    holds = [
        (src, dst, attrs)
        for src, dst, kind, attrs in demo_runtime.graph.relationships()
        if kind == "holds"
    ]
    assert holds, "demo book must produce positions"
    for _, _, attrs in holds:
        assert attrs["quantity"] != 0
        assert "market_value" in attrs


def test_demo_envelopes_fully_deterministic():
    first, summary_a = build_demo_book(seed=7)
    second, summary_b = build_demo_book(seed=7)
    assert summary_a == summary_b
    assert len(first) == len(second) == summary_a["envelopes"]
    assert [e.event_id for e in first] == [e.event_id for e in second]
    assert [e.content_hash() for e in first] == [e.content_hash() for e in second]


def test_demo_envelopes_seed_sensitive():
    base, _ = build_demo_book(seed=7)
    other, _ = build_demo_book(seed=8)
    assert [e.content_hash() for e in base] != [e.content_hash() for e in other]


def test_demo_no_accidental_entity_merges(demo_runtime):
    """All 40 demo customer names are distinct enough to survive fuzzy resolution."""
    assert demo_runtime.graph.stats()["entity_types"]["customer"] == 40
    assert demo_runtime.audit.records(action="twin.entity_resolved") == []


def test_demo_replay_episode_recorded(demo_runtime):
    assert "demo" in demo_runtime.replay.episodes()
    episode = demo_runtime.replay.episode("demo")
    assert len(episode) > 1000
    assert all(e.episode_id == "demo" for e in episode)
    assert all(e.provenance is not None for e in episode)


def test_demo_audit_trail_intact(demo_runtime):
    assert demo_runtime.audit.verify()
    ingested = demo_runtime.audit.records(action="twin.ingested")
    assert len(ingested) == len(demo_runtime.replay.episode("demo"))
    assert demo_runtime.audit.records(action="twin.demo_loaded")


def test_build_runtime_with_data_dir_persists(tmp_path):
    runtime = build_runtime(seed=7, with_demo_data=True, data_dir=tmp_path / "twin")
    episodes_dir = tmp_path / "twin" / "episodes"
    audit_file = tmp_path / "twin" / "audit.jsonl"
    assert (episodes_dir / "demo.jsonl").exists()
    assert audit_file.exists()
    assert runtime.audit.verify()
    # a second engine pointed at the same directory sees the persisted episode
    from fintwinos.twin_core.replay import JsonlReplayEngine

    reloaded = JsonlReplayEngine(episodes_dir)
    assert len(reloaded.episode("demo")) == len(runtime.replay.episode("demo"))
