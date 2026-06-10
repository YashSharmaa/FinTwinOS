"""Post-hoc calibration utilities for FinTwinOS simulators.

Every simulator in ``fintwinos.twin_sim`` is required (CONTRACTS.md, hard rule 3) to
return uncertainty alongside point estimates. This module supplies the shared numerics:

- :func:`ks_statistic` — two-sample Kolmogorov–Smirnov distance, pure numpy.
- :func:`bootstrap_ci` — percentile-bootstrap confidence interval for an arbitrary
  statistic, used by every simulator (>= 200 resampling draws).
- :func:`coverage_check` — empirical-coverage diagnostic for produced confidence
  intervals versus realised outcomes.
- :class:`PostHocCalibrator` — quantile-mapping (probability integral transform)
  calibration of a simulated distribution onto a reference sample, so simulator output
  distributions can be corrected against observed history without retraining the
  underlying dynamics.

All randomness flows through ``numpy.random.Generator`` instances supplied by the
caller, so results are reproducible end to end.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

#: Minimum number of bootstrap resampling draws used anywhere in twin_sim.
DEFAULT_BOOTSTRAP_DRAWS = 256

#: Default two-sided confidence level for bootstrap intervals (90% interval).
DEFAULT_ALPHA = 0.10

#: Number of quantile knots stored by the calibrator's monotone mapping.
_QUANTILE_KNOTS = 199


def ks_statistic(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic, implemented in numpy.

    The statistic is the supremum distance between the two empirical CDFs:
    ``D = sup_x |F_a(x) - F_b(x)|``. It lies in ``[0, 1]``; 0 means the empirical
    distributions coincide, 1 means their supports are disjoint.

    Args:
        sample_a: First sample (any 1-d sequence of floats).
        sample_b: Second sample.

    Returns:
        The KS distance as a float in ``[0, 1]``.

    Raises:
        ValueError: If either sample is empty.
    """
    a = np.sort(np.asarray(sample_a, dtype=float).ravel())
    b = np.sort(np.asarray(sample_b, dtype=float).ravel())
    if a.size == 0 or b.size == 0:
        raise ValueError("ks_statistic requires two non-empty samples")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def bootstrap_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    *,
    n_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    alpha: float = DEFAULT_ALPHA,
    statistic: Callable[[np.ndarray], float] | None = None,
) -> list[float]:
    """Percentile-bootstrap confidence interval ``[lo, hi]`` for a statistic.

    Resamples ``values`` with replacement ``n_draws`` times (>= 200 everywhere in
    twin_sim), applies ``statistic`` to each resample, and returns the
    ``alpha/2`` and ``1 - alpha/2`` percentiles of the resampled statistics.

    Args:
        values: Observed values (e.g. one metric per simulation replication).
        rng: Generator used for resampling — pass a seeded generator for determinism.
        n_draws: Number of bootstrap resamples (clamped to a minimum of 200).
        alpha: Two-sided tail mass; ``alpha=0.10`` yields a 90% interval.
        statistic: Statistic applied to each resample; defaults to the mean.

    Returns:
        ``[lo, hi]`` as a two-element list of floats. For a single observation the
        interval degenerates to ``[v, v]``.

    Raises:
        ValueError: If ``values`` is empty.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("bootstrap_ci requires at least one value")
    stat = statistic or (lambda x: float(np.mean(x)))
    if arr.size == 1:
        v = float(arr[0])
        return [v, v]
    n_draws = max(int(n_draws), 200)
    idx = rng.integers(0, arr.size, size=(n_draws, arr.size))
    draws = np.array([stat(arr[row]) for row in idx], dtype=float)
    lo = float(np.percentile(draws, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return [lo, hi]


def coverage_check(
    intervals: Sequence[Sequence[float]],
    realised: Sequence[float],
    *,
    nominal: float = 1.0 - DEFAULT_ALPHA,
    tolerance: float = 0.10,
) -> dict[str, Any]:
    """Empirical-coverage diagnostic for confidence intervals.

    Compares how often realised outcomes actually fall inside the produced
    ``[lo, hi]`` intervals against the nominal coverage level. A well-calibrated
    90% interval should cover roughly 90% of realised values.

    Args:
        intervals: Iterable of ``[lo, hi]`` pairs.
        realised: Realised outcome per interval (same length as ``intervals``).
        nominal: The nominal coverage the intervals claim (default matches
            :data:`DEFAULT_ALPHA`).
        tolerance: Maximum |empirical - nominal| gap considered calibrated.

    Returns:
        A dict with ``n``, ``covered``, ``coverage``, ``nominal``, ``gap`` and
        ``calibrated`` (bool).

    Raises:
        ValueError: If lengths differ or no intervals are supplied.
    """
    pairs = [(float(lo), float(hi)) for lo, hi in intervals]
    outcomes = [float(x) for x in realised]
    if len(pairs) != len(outcomes):
        raise ValueError("intervals and realised must have the same length")
    if not pairs:
        raise ValueError("coverage_check requires at least one interval")
    covered = sum(1 for (lo, hi), x in zip(pairs, outcomes, strict=True) if lo <= x <= hi)
    coverage = covered / len(pairs)
    gap = coverage - nominal
    return {
        "n": len(pairs),
        "covered": covered,
        "coverage": round(coverage, 6),
        "nominal": nominal,
        "gap": round(gap, 6),
        "calibrated": abs(gap) <= tolerance,
    }


class PostHocCalibrator:
    """Quantile-mapping calibration of simulated draws onto a reference sample.

    Fit on a pair ``(sim_sample, ref_sample)``: the calibrator learns a monotone map
    that sends each quantile of the simulated distribution to the corresponding
    quantile of the reference distribution. Transforming fresh simulator draws through
    that map yields output whose distribution matches the reference far more closely —
    the classic "quantile mapping" / distribution-correction technique used for model
    output statistics, implemented entirely in numpy.

    Attributes:
        fitted: Whether :meth:`fit` has been called.
        ks_before: KS distance between the fit-time sim sample and the reference.
        ks_after: KS distance between the transformed fit-time sim sample and the
            reference (the in-sample improvement achieved by the mapping).
    """

    def __init__(self, n_knots: int = _QUANTILE_KNOTS):
        if n_knots < 3:
            raise ValueError("PostHocCalibrator needs at least 3 quantile knots")
        self.n_knots = int(n_knots)
        self.fitted: bool = False
        self.ks_before: float | None = None
        self.ks_after: float | None = None
        self._sim_knots: np.ndarray | None = None
        self._ref_knots: np.ndarray | None = None

    def fit(self, sim_sample: Sequence[float], ref_sample: Sequence[float]) -> PostHocCalibrator:
        """Learn the quantile map from a simulated sample to a reference sample.

        Args:
            sim_sample: Draws from the (mis-calibrated) simulator distribution.
            ref_sample: Reference draws the simulator should match (e.g. history).

        Returns:
            ``self``, to allow ``PostHocCalibrator().fit(a, b)`` chaining.

        Raises:
            ValueError: If either sample has fewer than 8 observations.
        """
        sim = np.asarray(sim_sample, dtype=float).ravel()
        ref = np.asarray(ref_sample, dtype=float).ravel()
        if sim.size < 8 or ref.size < 8:
            raise ValueError("fit requires at least 8 observations per sample")
        probs = np.linspace(0.0, 1.0, self.n_knots + 2)[1:-1]
        sim_knots = np.quantile(sim, probs)
        ref_knots = np.quantile(ref, probs)
        # np.interp needs a strictly increasing abscissa: break ties with a tiny,
        # scale-aware epsilon so flat quantile stretches stay invertible.
        span = max(float(sim_knots[-1] - sim_knots[0]), 1e-12)
        eps = span * 1e-9
        sim_knots = np.maximum.accumulate(sim_knots + eps * np.arange(sim_knots.size))
        ref_knots = np.maximum.accumulate(ref_knots)
        self._sim_knots = sim_knots
        self._ref_knots = ref_knots
        self.fitted = True
        self.ks_before = ks_statistic(sim, ref)
        self.ks_after = ks_statistic(self.transform(sim), ref)
        return self

    def transform(self, draws: Sequence[float]) -> np.ndarray:
        """Map fresh simulator draws through the learned quantile correction.

        Values outside the fitted simulated range are clamped to the outermost
        reference quantiles (constant extrapolation), which keeps the map monotone
        and avoids inventing tail mass the reference never exhibited.

        Args:
            draws: New draws from the simulator distribution.

        Returns:
            A numpy array of calibrated draws, same shape as the input (flattened).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.fitted or self._sim_knots is None or self._ref_knots is None:
            raise RuntimeError("PostHocCalibrator.transform called before fit()")
        x = np.asarray(draws, dtype=float).ravel()
        return np.interp(x, self._sim_knots, self._ref_knots)

    def report(self) -> dict[str, Any]:
        """Summarise the calibrator state for inclusion in calibration reports."""
        return {
            "fitted": self.fitted,
            "n_knots": self.n_knots,
            "ks_before": None if self.ks_before is None else round(self.ks_before, 6),
            "ks_after": None if self.ks_after is None else round(self.ks_after, 6),
            "method": "quantile_mapping",
        }
