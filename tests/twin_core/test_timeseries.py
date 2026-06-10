"""PandasTimeSeriesStore: ordering, windows, latest, tags, resampling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fintwinos.twin_core.timeseries import PandasTimeSeriesStore

T0 = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)


@pytest.fixture()
def store() -> PandasTimeSeriesStore:
    s = PandasTimeSeriesStore()
    # appended deliberately out of order
    s.append("price:ALPH", T0 + timedelta(hours=2), 102.0, tags={"venue": "X"})
    s.append("price:ALPH", T0, 100.0, tags={"venue": "X"})
    s.append("price:ALPH", T0 + timedelta(hours=1), 101.0, tags={"venue": "Y"})
    s.append("balance:a1", T0, 5000.0)
    return s


def test_points_kept_sorted(store):
    values = [v for _, v in store.window("price:ALPH")]
    assert values == [100.0, 101.0, 102.0]


def test_window_bounds_inclusive(store):
    window = store.window("price:ALPH", start=T0, end=T0 + timedelta(hours=1))
    assert [v for _, v in window] == [100.0, 101.0]
    half_open = store.window("price:ALPH", start=T0 + timedelta(minutes=1))
    assert [v for _, v in half_open] == [101.0, 102.0]


def test_window_unknown_series_is_empty(store):
    assert store.window("missing") == []


def test_latest(store):
    latest = store.latest("price:ALPH")
    assert latest is not None
    ts, value = latest
    assert value == 102.0
    assert ts == T0 + timedelta(hours=2)
    assert store.latest("missing") is None


def test_keys_sorted(store):
    assert store.keys() == ["balance:a1", "price:ALPH"]


def test_naive_timestamps_coerced_to_utc(store):
    store.append("price:ALPH", datetime(2026, 3, 2, 12, 0), 103.0)  # naive
    ts, value = store.latest("price:ALPH")
    assert value == 103.0
    assert ts.tzinfo is not None


def test_tags_stored_alongside(store):
    points = store.points("price:ALPH")
    assert points[0][2] == {"venue": "X"}
    assert points[1][2] == {"venue": "Y"}
    # tags are copies — mutating the returned dict must not corrupt the store
    points[0][2]["venue"] = "MUTATED"
    assert store.points("price:ALPH")[0][2] == {"venue": "X"}


def test_resample_daily_mean():
    s = PandasTimeSeriesStore()
    for hour, value in [(0, 10.0), (6, 20.0), (30, 40.0)]:
        s.append("series", T0 + timedelta(hours=hour), value)
    daily = s.resample("series", "1D", how="mean")
    assert len(daily) == 2
    assert daily[0][1] == 15.0
    assert daily[1][1] == 40.0


def test_resample_last_and_validation(store):
    last = store.resample("price:ALPH", "1D", how="last")
    assert last[-1][1] == 102.0
    assert store.resample("missing", "1D") == []
    with pytest.raises(ValueError):
        store.resample("price:ALPH", "1D", how="bogus")


def test_to_frame(store):
    frame = store.to_frame("price:ALPH")
    assert list(frame["value"]) == [100.0, 101.0, 102.0]
    assert list(frame["venue"]) == ["X", "Y", "X"]
    empty = store.to_frame("missing")
    assert empty.empty
