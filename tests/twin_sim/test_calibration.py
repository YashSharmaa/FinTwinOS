"""Tests for the calibration utilities: KS, bootstrap, coverage, quantile mapping."""

from __future__ import annotations

import numpy as np
import pytest

from fintwinos.twin_sim.calibration import (
    PostHocCalibrator,
    bootstrap_ci,
    coverage_check,
    ks_statistic,
)

# --- ks_statistic ---------------------------------------------------------------


def test_ks_identical_samples_is_zero():
    x = np.linspace(-3, 3, 500)
    assert ks_statistic(x, x) == pytest.approx(0.0)


def test_ks_disjoint_supports_is_one():
    a = np.linspace(0.0, 1.0, 200)
    b = np.linspace(10.0, 11.0, 200)
    assert ks_statistic(a, b) == pytest.approx(1.0)


def test_ks_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 300)
    b = rng.normal(0.5, 1.4, 400)
    d_ab = ks_statistic(a, b)
    d_ba = ks_statistic(b, a)
    assert d_ab == pytest.approx(d_ba)
    assert 0.0 < d_ab < 1.0


def test_ks_rejects_empty_sample():
    with pytest.raises(ValueError):
        ks_statistic([], [1.0, 2.0])


# --- bootstrap_ci ------------------------------------------------------------------


def test_bootstrap_ci_is_deterministic_and_ordered():
    values = np.random.default_rng(1).normal(10.0, 2.0, 40)
    ci1 = bootstrap_ci(values, np.random.default_rng(42))
    ci2 = bootstrap_ci(values, np.random.default_rng(42))
    assert ci1 == ci2
    lo, hi = ci1
    assert lo <= hi
    assert lo <= float(np.mean(values)) <= hi


def test_bootstrap_ci_single_value_degenerates():
    assert bootstrap_ci([3.5], np.random.default_rng(0)) == [3.5, 3.5]


def test_bootstrap_ci_enforces_minimum_draws():
    # Even when asked for fewer draws, the implementation clamps to >= 200 and
    # still produces a valid interval.
    values = np.arange(20.0)
    lo, hi = bootstrap_ci(values, np.random.default_rng(7), n_draws=10)
    assert lo <= hi


def test_bootstrap_ci_custom_statistic():
    values = np.random.default_rng(3).exponential(1.0, 60)
    lo, hi = bootstrap_ci(
        values, np.random.default_rng(5), statistic=lambda x: float(np.percentile(x, 90))
    )
    assert lo <= hi
    assert hi > float(np.mean(values))  # 90th percentile of an exponential > mean


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([], np.random.default_rng(0))


# --- coverage_check ------------------------------------------------------------------


def test_coverage_check_counts_correctly():
    intervals = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    realised = [0.5, 0.99, 1.5, -0.2]  # two inside, two outside
    out = coverage_check(intervals, realised, nominal=0.5, tolerance=0.05)
    assert out["n"] == 4
    assert out["covered"] == 2
    assert out["coverage"] == pytest.approx(0.5)
    assert out["calibrated"] is True


def test_coverage_check_flags_miscalibration():
    intervals = [[0.0, 0.1]] * 10
    realised = [5.0] * 10
    out = coverage_check(intervals, realised, nominal=0.9)
    assert out["coverage"] == 0.0
    assert out["calibrated"] is False


def test_coverage_check_validates_inputs():
    with pytest.raises(ValueError):
        coverage_check([[0, 1]], [0.5, 0.6])
    with pytest.raises(ValueError):
        coverage_check([], [])


# --- PostHocCalibrator ------------------------------------------------------------------


def test_calibrator_reduces_ks_statistic():
    rng = np.random.default_rng(11)
    sim_fit = rng.normal(0.0, 1.0, 2000)
    ref = rng.normal(2.0, 2.0, 2000)
    cal = PostHocCalibrator().fit(sim_fit, ref)

    fresh = np.random.default_rng(99).normal(0.0, 1.0, 1500)
    ks_raw = ks_statistic(fresh, ref)
    ks_cal = ks_statistic(cal.transform(fresh), ref)
    assert ks_cal < ks_raw
    assert ks_cal < 0.1  # quantile mapping should nearly match the reference

    assert cal.fitted is True
    assert cal.ks_after is not None and cal.ks_before is not None
    assert cal.ks_after < cal.ks_before


def test_calibrator_transform_is_monotone():
    rng = np.random.default_rng(2)
    cal = PostHocCalibrator().fit(rng.normal(0, 1, 500), rng.exponential(2.0, 500))
    x = np.linspace(-3, 3, 101)
    y = cal.transform(x)
    assert np.all(np.diff(y) >= 0.0)


def test_calibrator_requires_fit_before_transform():
    with pytest.raises(RuntimeError):
        PostHocCalibrator().transform([0.0])


def test_calibrator_rejects_tiny_samples():
    with pytest.raises(ValueError):
        PostHocCalibrator().fit([1.0, 2.0], np.arange(100.0))


def test_calibrator_report_shape():
    rng = np.random.default_rng(4)
    cal = PostHocCalibrator().fit(rng.normal(0, 1, 100), rng.normal(1, 1, 100))
    report = cal.report()
    assert report["fitted"] is True
    assert report["method"] == "quantile_mapping"
    assert report["ks_after"] <= report["ks_before"]
