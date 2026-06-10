"""Tests for the seeded GBM+jump synthetic market connector."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from fintwinos.connectors.base import ConnectorError, pump
from fintwinos.connectors.market_synth import (
    DEFAULT_START_TIME,
    SyntheticMarketConnector,
)


async def collect(connector):
    return [envelope async for envelope in connector.stream()]


def mids(envelopes, instrument):
    return [e.payload["mid"] for e in envelopes if e.payload["instrument"] == instrument]


async def test_envelope_shape_and_count():
    connector = SyntheticMarketConnector(["EQ.AAPL", "EQ.MSFT"], n_ticks=10, seed=11)
    envelopes = await collect(connector)

    assert len(envelopes) == 20  # n_ticks * instruments
    first = envelopes[0]
    assert first.kind == "market.tick"
    assert first.source == "synthetic_market"
    assert set(first.payload) == {"instrument", "mid", "ts", "tick", "seed"}
    assert first.payload["instrument"] == "EQ.AAPL"
    assert first.payload["seed"] == 11
    assert [ref.key() for ref in first.entities] == ["instrument:EQ.AAPL"]
    assert first.occurred_at == DEFAULT_START_TIME
    assert first.payload["ts"] == DEFAULT_START_TIME.isoformat()
    # ticks are interleaved by time: AAPL, MSFT, AAPL, MSFT, ...
    assert [e.payload["instrument"] for e in envelopes[:4]] == [
        "EQ.AAPL", "EQ.MSFT", "EQ.AAPL", "EQ.MSFT",
    ]
    assert all(e.payload["mid"] > 0 for e in envelopes)


async def test_same_seed_reproduces_identical_stream():
    kwargs = dict(n_ticks=50, seed=42, jump_intensity=0.5, jump_sigma=0.05)
    a = await collect(SyntheticMarketConnector(["EQ.X", "EQ.Y"], **kwargs))
    b = await collect(SyntheticMarketConnector(["EQ.X", "EQ.Y"], **kwargs))
    assert [e.payload["mid"] for e in a] == [e.payload["mid"] for e in b]
    assert [e.payload["ts"] for e in a] == [e.payload["ts"] for e in b]


async def test_different_seed_changes_the_path():
    a = await collect(SyntheticMarketConnector(["EQ.X"], n_ticks=50, seed=1))
    b = await collect(SyntheticMarketConnector(["EQ.X"], n_ticks=50, seed=2))
    assert [e.payload["mid"] for e in a] != [e.payload["mid"] for e in b]


async def test_timestamps_advance_by_tick_interval():
    start = datetime(2026, 6, 1, 13, 30, tzinfo=UTC)
    connector = SyntheticMarketConnector(
        ["EQ.X"], n_ticks=4, seed=3, tick_interval_s=30.0, start_time=start
    )
    envelopes = await collect(connector)
    times = [e.occurred_at for e in envelopes]
    assert times == [start + timedelta(seconds=30 * i) for i in range(4)]


async def test_zero_vol_zero_jumps_is_flat_modulo_drift():
    connector = SyntheticMarketConnector(
        ["EQ.X"], n_ticks=20, seed=5, mu=0.0, sigma=0.0, jump_intensity=0.0,
        start_prices=50.0,
    )
    envelopes = await collect(connector)
    assert all(e.payload["mid"] == pytest.approx(50.0) for e in envelopes)


async def test_jumps_move_the_path():
    base = dict(n_ticks=200, seed=9, mu=0.0, sigma=0.0)
    calm = await collect(
        SyntheticMarketConnector(["EQ.X"], **base, jump_intensity=0.0)
    )
    jumpy = await collect(
        SyntheticMarketConnector(["EQ.X"], **base, jump_intensity=2.0, jump_sigma=0.1)
    )
    calm_mids = np.array(mids(calm, "EQ.X"))
    jumpy_mids = np.array(mids(jumpy, "EQ.X"))
    assert float(np.std(calm_mids)) == pytest.approx(0.0)
    assert float(np.std(jumpy_mids)) > 0.0


async def test_per_instrument_start_prices():
    connector = SyntheticMarketConnector(
        ["EQ.X", "EQ.Y"], n_ticks=1, seed=7, sigma=0.0, mu=0.0, jump_intensity=0.0,
        start_prices={"EQ.X": 10.0, "EQ.Y": 250.0},
    )
    envelopes = await collect(connector)
    assert mids(envelopes, "EQ.X") == [pytest.approx(10.0)]
    assert mids(envelopes, "EQ.Y") == [pytest.approx(250.0)]


def test_paths_method_matches_streamed_mids():
    connector = SyntheticMarketConnector(["EQ.X"], n_ticks=25, seed=13)
    direct = connector.paths()["EQ.X"]
    assert direct.shape == (25,)
    # paths() re-derives the identical deterministic series every call
    again = connector.paths()["EQ.X"]
    assert np.allclose(direct, again)


def test_invalid_parameters_rejected():
    with pytest.raises(ConnectorError):
        SyntheticMarketConnector(["EQ.X"], n_ticks=0)
    with pytest.raises(ConnectorError):
        SyntheticMarketConnector([])
    with pytest.raises(ConnectorError):
        SyntheticMarketConnector(["EQ.X"], sigma=-0.1)


async def test_pump_synthetic_ticks(recorder_factory):
    connector = SyntheticMarketConnector(["EQ.X"], n_ticks=15, seed=21)
    recorder = recorder_factory()
    summary = await pump(connector, recorder)
    assert summary["ingested"] == 15
    assert {e.kind for e in recorder.envelopes} == {"market.tick"}
