"""Pandas-flavoured in-memory time-series store for the twin.

Market mid-prices, account balances, cash ladders, alert scores and any other
dynamic quantity in the twin live here as named series of ``(timestamp, value,
tags)`` points kept sorted by timestamp. Reads are pure Python/bisect for
speed; :meth:`PandasTimeSeriesStore.resample` and
:meth:`PandasTimeSeriesStore.to_frame` bridge into pandas for analytics.

Implements the :class:`fintwinos.core.interfaces.TimeSeriesStore` protocol.
"""

from __future__ import annotations

import bisect
from datetime import UTC, datetime
from operator import itemgetter
from typing import Any

import pandas as pd

_ALLOWED_AGGREGATIONS = {"mean", "sum", "last", "first", "min", "max", "median", "count"}


def _to_utc(ts: datetime) -> datetime:
    """Coerce a timestamp to timezone-aware UTC (naive values are assumed UTC)."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


class PandasTimeSeriesStore:
    """Per-key sorted point lists with tags stored alongside each observation.

    Points are kept sorted by timestamp on insert (binary insertion), so
    windows and latest-value lookups are O(log n) seeks plus a slice. Equal
    timestamps preserve append order.
    """

    def __init__(self) -> None:
        self._series: dict[str, list[tuple[datetime, float, dict[str, Any]]]] = {}

    # -- write ---------------------------------------------------------------------

    def append(
        self,
        series_key: str,
        ts: datetime,
        value: float,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """Insert one observation, keeping the series sorted by timestamp."""
        row = (_to_utc(ts), float(value), dict(tags or {}))
        rows = self._series.setdefault(series_key, [])
        if not rows or row[0] >= rows[-1][0]:
            rows.append(row)  # fast path: appends are usually chronological
        else:
            bisect.insort_right(rows, row, key=itemgetter(0))

    # -- read ----------------------------------------------------------------------

    def window(
        self,
        series_key: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[tuple[datetime, float]]:
        """Points with ``start <= ts <= end`` (inclusive bounds, None = open)."""
        rows = self._series.get(series_key, [])
        lo = 0 if start is None else bisect.bisect_left(rows, _to_utc(start), key=itemgetter(0))
        hi = len(rows) if end is None else bisect.bisect_right(rows, _to_utc(end), key=itemgetter(0))
        return [(ts, value) for ts, value, _ in rows[lo:hi]]

    def points(
        self,
        series_key: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[tuple[datetime, float, dict[str, Any]]]:
        """Like :meth:`window` but including a copy of each point's tags."""
        rows = self._series.get(series_key, [])
        lo = 0 if start is None else bisect.bisect_left(rows, _to_utc(start), key=itemgetter(0))
        hi = len(rows) if end is None else bisect.bisect_right(rows, _to_utc(end), key=itemgetter(0))
        return [(ts, value, dict(tags)) for ts, value, tags in rows[lo:hi]]

    def latest(self, series_key: str) -> tuple[datetime, float] | None:
        """The most recent ``(timestamp, value)`` of a series, or None if empty."""
        rows = self._series.get(series_key)
        if not rows:
            return None
        ts, value, _ = rows[-1]
        return (ts, value)

    def keys(self) -> list[str]:
        """All series keys, sorted."""
        return sorted(self._series)

    def __len__(self) -> int:
        return len(self._series)

    # -- pandas bridge ---------------------------------------------------------------

    def to_frame(self, series_key: str) -> pd.DataFrame:
        """The series as a DataFrame indexed by UTC timestamp with a ``value`` column.

        Tag keys appearing on any point become additional columns.
        """
        rows = self._series.get(series_key, [])
        if not rows:
            return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], tz="UTC"))
        index = pd.DatetimeIndex([ts for ts, _, _ in rows], name="ts")
        frame = pd.DataFrame({"value": [v for _, v, _ in rows]}, index=index)
        tag_keys = sorted({k for _, _, tags in rows for k in tags})
        for key in tag_keys:
            frame[key] = [tags.get(key) for _, _, tags in rows]
        return frame

    def resample(
        self,
        series_key: str,
        freq: str,
        how: str = "mean",
    ) -> list[tuple[datetime, float]]:
        """Aggregate a series onto a regular grid (e.g. ``freq="1D", how="last"``).

        ``how`` must be one of mean/sum/last/first/min/max/median/count. Empty
        buckets are dropped. Returns ``(timestamp, value)`` tuples with
        timezone-aware UTC timestamps.
        """
        if how not in _ALLOWED_AGGREGATIONS:
            raise ValueError(
                f"unsupported aggregation {how!r}; choose one of {sorted(_ALLOWED_AGGREGATIONS)}"
            )
        rows = self._series.get(series_key, [])
        if not rows:
            return []
        series = pd.Series(
            [value for _, value, _ in rows],
            index=pd.DatetimeIndex([ts for ts, _, _ in rows]),
        )
        aggregated = series.resample(freq).agg(how).dropna()
        return [(ts.to_pydatetime(), float(value)) for ts, value in aggregated.items()]
