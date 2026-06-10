"""Regime detection via rolling z-scores with two-state hysteresis.

A simple, fully auditable change detector for the twin's market and operational
series: each point is z-scored against the trailing ``window`` observations
(excluding the point itself), and a two-state machine flips from *calm* (0) to
*stressed* (1) when the score breaches ``enter_z``, only returning to calm once it
falls back below ``exit_z``. The gap between the two thresholds is the hysteresis
band that prevents label chatter when the signal hovers near a single threshold.

The detector is level-based by design. To detect *volatility* regimes, feed it a
volatility proxy series — e.g. ``numpy.abs(returns)`` or the ``vol_`` series from
:class:`fintwinos.models.time_series.forecast.EwmaVol` — rather than raw returns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: z-score magnitude substituted when the trailing window has ~zero variance but the
#: current point deviates from its mean (a genuine jump off a flat line).
_FLATLINE_Z = 1e6


@dataclass
class RegimeResult:
    """Output of :meth:`RegimeDetector.detect`.

    Attributes:
        labels: Integer array (0 = calm, 1 = stressed), one label per input point.
        change_points: Indices ``t`` where ``labels[t] != labels[t-1]`` — i.e. the
            first index of each new regime segment after the initial one.
        zscores: Rolling z-score of each point against its trailing window
            (0.0 wherever there was insufficient history).
        window: Trailing window length used.
        enter_z: Threshold that switches calm -> stressed.
        exit_z: Threshold that switches stressed -> calm.
    """

    labels: np.ndarray
    change_points: list[int]
    zscores: np.ndarray
    window: int
    enter_z: float
    exit_z: float

    def segments(self) -> list[tuple[int, int, int]]:
        """Contiguous regime segments as ``(start, end_exclusive, label)`` triples."""
        if self.labels.size == 0:
            return []
        out: list[tuple[int, int, int]] = []
        start = 0
        for cp in self.change_points:
            out.append((start, cp, int(self.labels[start])))
            start = cp
        out.append((start, int(self.labels.size), int(self.labels[start])))
        return out


class RegimeDetector:
    """Two-state (calm/stressed) regime detector with hysteresis.

    State machine, evaluated left to right over the series:

    - In state 0 (calm): switch to 1 when the trigger score reaches ``enter_z``.
    - In state 1 (stressed): switch back to 0 only when it drops to ``exit_z`` or
      below. Because ``exit_z < enter_z``, scores inside the band keep the current
      state — that asymmetry is the hysteresis.

    The trigger score is ``|z|`` when ``two_sided`` (default) so both up- and
    down-shocks count, or the signed ``z`` for one-sided (upward-only) detection.
    """

    def __init__(
        self,
        window: int = 50,
        *,
        enter_z: float = 3.0,
        exit_z: float = 1.0,
        two_sided: bool = True,
    ) -> None:
        """Configure the detector.

        Args:
            window: Trailing window length used for the rolling mean/std. Points
                with fewer than ``window`` predecessors are labelled calm with a
                z-score of 0.
            enter_z: Calm -> stressed threshold (must exceed ``exit_z``).
            exit_z: Stressed -> calm threshold (>= 0).
            two_sided: Trigger on ``|z|`` (True) or signed ``z`` (False).

        Raises:
            ValueError: If ``window < 2`` or ``enter_z <= exit_z`` or ``exit_z < 0``.
        """
        if window < 2:
            raise ValueError("window must be >= 2")
        if exit_z < 0:
            raise ValueError("exit_z must be non-negative")
        if enter_z <= exit_z:
            raise ValueError("enter_z must be strictly greater than exit_z (hysteresis band)")
        self.window = int(window)
        self.enter_z = float(enter_z)
        self.exit_z = float(exit_z)
        self.two_sided = bool(two_sided)

    def detect(self, series: Sequence[float] | np.ndarray) -> RegimeResult:
        """Label every point of a series and locate the regime change points.

        Args:
            series: Observations ordered oldest to newest.

        Returns:
            :class:`RegimeResult` with per-point labels, z-scores and change points.

        Raises:
            ValueError: If the series contains non-finite values.
        """
        y = np.asarray(series, dtype=np.float64).ravel()
        n = y.size
        if n > 0 and not np.all(np.isfinite(y)):
            raise ValueError("series contains non-finite values")

        labels = np.zeros(n, dtype=np.int64)
        zscores = np.zeros(n, dtype=np.float64)
        state = 0
        for t in range(n):
            if t < self.window:
                labels[t] = state  # insufficient history: keep the initial calm state
                continue
            hist = y[t - self.window : t]
            mu = float(hist.mean())
            sd = float(hist.std(ddof=1))
            diff = float(y[t]) - mu
            if sd < 1e-12:
                z = 0.0 if abs(diff) < 1e-12 else math.copysign(_FLATLINE_Z, diff)
            else:
                z = diff / sd
            zscores[t] = z
            trigger = abs(z) if self.two_sided else z
            if state == 0 and trigger >= self.enter_z:
                state = 1
            elif state == 1 and trigger <= self.exit_z:
                state = 0
            labels[t] = state

        change_points = [t for t in range(1, n) if labels[t] != labels[t - 1]]
        return RegimeResult(
            labels=labels,
            change_points=change_points,
            zscores=zscores,
            window=self.window,
            enter_z=self.enter_z,
            exit_z=self.exit_z,
        )
