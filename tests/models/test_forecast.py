"""Tests for AR forecasting, EWMA volatility and rolling-origin backtesting."""

from __future__ import annotations

import numpy as np
import pytest

from fintwinos.models.time_series.forecast import (
    ARForecaster,
    EwmaVol,
    NaiveForecaster,
    _norm_ppf,
    backtest,
)


def simulate_ar2(
    n: int = 500, phi1: float = 0.65, phi2: float = -0.3, sigma: float = 1.0, seed: int = 42
) -> np.ndarray:
    """Stationary AR(2) sample path with standard-normal innovations."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]
    return y


class TestNormPpf:
    def test_known_quantiles(self):
        assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
        assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)
        assert _norm_ppf(0.025) == pytest.approx(-1.959964, abs=1e-4)
        assert _norm_ppf(0.995) == pytest.approx(2.575829, abs=1e-4)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            _norm_ppf(0.0)
        with pytest.raises(ValueError):
            _norm_ppf(1.0)


class TestARForecaster:
    def test_recovers_ar2_coefficients(self):
        y = simulate_ar2(n=4000, seed=1)
        model = ARForecaster(p=2).fit(y)
        assert model.coef_[0] == pytest.approx(0.65, abs=0.06)
        assert model.coef_[1] == pytest.approx(-0.30, abs=0.06)
        assert model.sigma_ == pytest.approx(1.0, abs=0.07)
        assert not model.used_ridge_

    def test_beats_naive_on_ar2_backtest(self):
        y = simulate_ar2(n=500, seed=42)
        ar_result = backtest(y, ARForecaster(p=2), window=120)
        naive_result = backtest(y, NaiveForecaster(), window=120)
        assert ar_result["mae"] < naive_result["mae"]
        assert ar_result["n"] == naive_result["n"]

    def test_backtest_interval_coverage_near_nominal(self):
        y = simulate_ar2(n=500, seed=42)
        result = backtest(y, ARForecaster(p=2), window=120, level=0.95)
        assert 0.85 <= result["coverage"] <= 1.0
        assert result["level"] == 0.95
        assert result["mape"] > 0.0

    def test_forecast_shapes_and_band_ordering(self):
        y = simulate_ar2(n=300, seed=7)
        fc = ARForecaster(p=2).fit(y).forecast(10)
        assert fc.mean.shape == fc.lower.shape == fc.upper.shape == (10,)
        assert np.all(fc.lower < fc.mean) and np.all(fc.mean < fc.upper)
        assert fc.horizon == 10 and fc.level == 0.95

    def test_interval_width_grows_with_horizon(self):
        y = simulate_ar2(n=300, seed=7)
        fc = ARForecaster(p=2).fit(y).forecast(8)
        widths = fc.upper - fc.lower
        assert np.all(np.diff(widths) >= -1e-12)
        assert widths[-1] > widths[0]

    def test_ridge_fallback_on_degenerate_series(self):
        # A constant series makes the lag columns collinear with the intercept.
        model = ARForecaster(p=2).fit(np.full(50, 3.0))
        assert model.used_ridge_
        fc = model.forecast(3)
        assert fc.mean == pytest.approx(np.full(3, 3.0), abs=1e-6)

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError):
            ARForecaster(p=3).fit([1.0, 2.0, 3.0])

    def test_forecast_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            ARForecaster().forecast(5)

    def test_non_finite_series_raises(self):
        with pytest.raises(ValueError):
            ARForecaster().fit([1.0, np.nan, 2.0, 3.0, 4.0])

    def test_as_dict_is_json_friendly(self):
        fc = ARForecaster(p=2).fit(simulate_ar2(n=100, seed=3)).forecast(4)
        d = fc.as_dict()
        assert isinstance(d["mean"], list) and len(d["mean"]) == 4
        assert d["horizon"] == 4


class TestNaiveForecaster:
    def test_flat_mean_path_at_last_value(self):
        fc = NaiveForecaster().fit([1.0, 2.0, 3.0, 4.0]).forecast(5)
        assert np.all(fc.mean == 4.0)

    def test_interval_grows_like_sqrt_h(self):
        walk = np.cumsum(np.random.default_rng(4).normal(0.0, 1.0, 50))
        fc = NaiveForecaster().fit(walk).forecast(4)
        widths = fc.upper - fc.lower
        assert widths[3] / widths[0] == pytest.approx(2.0, rel=1e-9)


class TestEwmaVol:
    def test_responds_to_variance_break(self):
        rng = np.random.default_rng(13)
        calm = rng.normal(0.0, 0.01, 300)
        stressed = rng.normal(0.0, 0.05, 150)
        returns = np.concatenate([calm, stressed])
        ev = EwmaVol(lam=0.94).fit(returns)
        vol_before = float(ev.vol_[299])
        vol_after = ev.latest_vol
        assert vol_before == pytest.approx(0.01, rel=0.5)
        assert vol_after == pytest.approx(0.05, rel=0.5)
        assert vol_after > 2.5 * vol_before
        assert ev.vol_.shape == returns.shape

    def test_streaming_update_extends_series(self):
        ev = EwmaVol().fit(np.full(30, 0.01))
        n_before = ev.vol_.size
        new_vol = ev.update(0.10)
        assert ev.vol_.size == n_before + 1
        assert new_vol > ev.vol_[n_before - 1]

    def test_forecast_is_flat_at_latest_vol(self):
        ev = EwmaVol().fit(np.random.default_rng(0).normal(0, 0.02, 100))
        fc = ev.forecast(5)
        assert np.all(fc == ev.latest_vol)

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            EwmaVol(lam=1.0)

    def test_unfitted_access_raises(self):
        with pytest.raises(RuntimeError):
            _ = EwmaVol().latest_vol


class TestBacktest:
    def test_too_short_series_raises(self):
        with pytest.raises(ValueError):
            backtest(np.arange(10.0), NaiveForecaster(), window=10)

    def test_step_reduces_evaluations(self):
        y = simulate_ar2(n=300, seed=5)
        dense = backtest(y, NaiveForecaster(), window=100, step=1)
        sparse = backtest(y, NaiveForecaster(), window=100, step=5)
        assert sparse["n"] < dense["n"]

    def test_multi_step_horizon(self):
        y = simulate_ar2(n=300, seed=5)
        result = backtest(y, ARForecaster(p=2), window=100, horizon=3)
        assert result["n"] > 0
        assert result["mae"] > 0
