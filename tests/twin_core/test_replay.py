"""JsonlReplayEngine: episode recording, JSONL persistence, replay round-trips."""

from __future__ import annotations

from datetime import UTC, datetime

from fintwinos.core.types import EventEnvelope
from fintwinos.twin_core.replay import JsonlReplayEngine
from fintwinos.twin_core.runtime import build_runtime
from fintwinos.twin_core.snapshot import snapshot_runtime

T0 = datetime(2026, 4, 1, 9, 0, tzinfo=UTC)


def _env(kind: str, episode_id: str | None, payload: dict | None = None) -> EventEnvelope:
    return EventEnvelope(
        kind=kind,
        occurred_at=T0,
        source="test",
        payload=payload or {},
        episode_id=episode_id,
    )


def test_record_and_read_in_memory():
    engine = JsonlReplayEngine()
    engine.record(_env("customer.created", "ep1", {"customer_id": "c1"}))
    engine.record(_env("account.created", "ep1", {"account_id": "a1"}))
    engine.record(_env("market.price", "ep2", {"mid": 10.0}))
    assert engine.episodes() == ["ep1", "ep2"]
    assert [e.kind for e in engine.episode("ep1")] == ["customer.created", "account.created"]
    assert len(engine) == 3
    assert engine.episode("missing") == []


def test_default_episode_for_unset_id():
    engine = JsonlReplayEngine()
    engine.record(_env("market.price", None, {"mid": 1.0}))
    assert engine.episodes() == ["default"]
    assert engine.episode("default")[0].episode_id == "default"


def test_replay_invokes_handler_in_order():
    engine = JsonlReplayEngine()
    for i in range(5):
        engine.record(_env("market.price", "ep", {"mid": float(i)}))
    seen: list[float] = []
    count = engine.replay("ep", lambda e: seen.append(e.payload["mid"]))
    assert count == 5
    assert seen == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert engine.replay("missing", lambda e: seen.append(-1.0)) == 0
    assert len(seen) == 5


def test_jsonl_persistence_round_trip(tmp_path):
    path = tmp_path / "episodes"
    engine = JsonlReplayEngine(path)
    original = [_env("customer.created", "ep1", {"customer_id": f"c{i}"}) for i in range(3)]
    for envelope in original:
        engine.record(envelope)
    engine.record(_env("market.price", None, {"mid": 4.2}))

    assert (path / "ep1.jsonl").exists()
    assert (path / "default.jsonl").exists()

    reloaded = JsonlReplayEngine(path)
    assert reloaded.episodes() == ["default", "ep1"]
    restored = reloaded.episode("ep1")
    assert [e.event_id for e in restored] == [e.event_id for e in original]
    assert [e.payload for e in restored] == [e.payload for e in original]
    assert restored[0].occurred_at == original[0].occurred_at


def test_episode_ids_sanitised_for_filenames(tmp_path):
    engine = JsonlReplayEngine(tmp_path / "episodes")
    engine.record(_env("market.price", "weird/episode id!", {"mid": 1.0}))
    files = list((tmp_path / "episodes").glob("*.jsonl"))
    assert len(files) == 1
    assert "/" not in files[0].name
    # the authoritative id survives the round trip
    reloaded = JsonlReplayEngine(tmp_path / "episodes")
    assert reloaded.episodes() == ["weird/episode id!"]


def test_replaying_demo_episode_rebuilds_identical_twin(demo_runtime):
    """Episodes are the source of truth: replay into an empty twin == original twin."""
    fresh = build_runtime(seed=7, with_demo_data=False)
    count = demo_runtime.replay.replay("demo", fresh.ingestor.ingest)
    assert count == len(demo_runtime.replay.episode("demo"))
    assert count > 1000
    assert snapshot_runtime(fresh)["hash"] == snapshot_runtime(demo_runtime)["hash"]
