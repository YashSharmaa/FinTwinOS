"""Time-series forecasting in pure numpy: AR(p), EWMA volatility, and backtesting.

These are the workhorse classical models behind the twin's market and operational
dynamics. Per the FinTwinOS contract every model returns *intervals*, never bare
point estimates: ``forecast(h)`` produces a mean path plus lower/upper bands derived
from the residual sigma and the model's impulse-response weights, and ``backtest``
reports how often reality actually landed inside those bands (interval coverage)
alongside MAE/MAPE.

Contents:

- :class:`ARForecaster` — autoregressive AR(p) model fitted by ordinary least
  squares, with an automatic ridge fallback when the design matrix is
  rank-deficient or numerically unstable.
- :class:`NaiveForecaster` — random-walk ("last value") baseline implementing the
  same protocol, so model lift can always be measured against it.
- :class:`EwmaVol` — RiskMetrics-style exponentially weighted volatility
  (``lambda = 0.94`` daily by convention).
- :func:`backtest` — rolling-origin one-/multi-step evaluation of any model with
  the ``fit``/``forecast`` protocol.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

DEFAULT_LEVEL = 0.95


def _norm_ppf(q: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9 accurate).

    Implemented locally because scipy is intentionally not a dependency.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must be strictly between 0 and 1")
    a = (
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low
    if q < p_low:
        u = math.sqrt(-2.0 * math.log(q))
        return (((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / (
            (((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0
        )
    if q > p_high:
        u = math.sqrt(-2.0 * math.log(1.0 - q))
        return -(((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / (
            (((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0
        )
    u = q - 0.5
    r = u * u
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * u / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


@dataclass
class ForecastResult:
    """A mean forecast path with symmetric Gaussian prediction intervals.

    Attributes:
        mean: Point forecasts for horizons ``1..h``.
        lower / upper: Prediction-interval bounds at ``level`` for each horizon.
        sigma: One-step-ahead residual standard deviation of the fitted model.
        level: Two-sided interval coverage level (e.g. 0.95).
        horizon: Number of steps forecast.
    """

    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sigma: float
    level: float
    horizon: int

    def as_dict(self) -> dict[str, list[float] | float | int]:
        """JSON-friendly representation (e.g. for ``SimulationResult.series``)."""
        return {
            "mean": self.mean.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "sigma": self.sigma,
            "level": self.level,
            "horizon": self.horizon,
        }


class ARForecaster:
    """Autoregressive AR(p) forecaster fitted by least squares with ridge fallback.

    The model is ``y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p} + eps_t``. Fitting
    first attempts ordinary least squares (``numpy.linalg.lstsq``); if the design
    matrix is rank-deficient or the solution is not finite, it falls back to a ridge
    solve of the normal equations (the intercept is never penalised). Multi-step
    prediction intervals widen with horizon according to the AR impulse-response
    (psi) weights: ``Var(e_{t+h}) = sigma^2 * sum_{j<h} psi_j^2``.

    Attributes:
        coef_: Fitted AR coefficients ``phi_1..phi_p`` (after ``fit``).
        intercept_: Fitted constant term.
        sigma_: Residual standard deviation (one-step-ahead).
        used_ridge_: True when the ridge fallback was taken.
    """

    def __init__(self, p: int = 2, *, ridge: float = 1e-4, level: float = DEFAULT_LEVEL) -> None:
        """Configure the model.

        Args:
            p: Autoregressive order (number of lags).
            ridge: Regularisation strength used only by the fallback solver.
            level: Default two-sided prediction-interval level for ``forecast``.
        """
        if p < 1:
            raise ValueError("AR order p must be >= 1")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        self.p = int(p)
        self.ridge = float(ridge)
        self.level = float(level)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.sigma_: float = 0.0
        self.used_ridge_: bool = False
        self._last_values: np.ndarray | None = None

    def fit(self, series: Sequence[float] | np.ndarray) -> ARForecaster:
        """Fit the AR(p) coefficients on a 1-D series.

        Args:
            series: Observations ordered oldest to newest; needs at least ``p + 2``
                points so the residual variance has a degree of freedom.

        Returns:
            ``self`` (fluent style).

        Raises:
            ValueError: If the series is too short or contains non-finite values.
        """
        y = np.asarray(series, dtype=np.float64).ravel()
        if y.size < self.p + 2:
            raise ValueError(f"need at least p + 2 = {self.p + 2} observations, got {y.size}")
        if not np.all(np.isfinite(y)):
            raise ValueError("series contains non-finite values")

        n = y.size
        target = y[self.p :]
        lags = [y[self.p - k : n - k] for k in range(1, self.p + 1)]
        X = np.column_stack([np.ones(n - self.p), *lags])

        beta: np.ndarray | None = None
        self.used_ridge_ = False
        try:
            solution, _, rank, _ = np.linalg.lstsq(X, target, rcond=None)
            if rank == X.shape[1] and np.all(np.isfinite(solution)):
                beta = solution
        except np.linalg.LinAlgError:
            beta = None
        if beta is None:
            self.used_ridge_ = True
            penalty = self.ridge * np.eye(X.shape[1])
            penalty[0, 0] = 0.0  # never shrink the intercept
            gram = X.T @ X + penalty
            try:
                beta = np.linalg.solve(gram, X.T @ target)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(gram) @ (X.T @ target)

        residuals = target - X @ beta
        dof = max(1, target.size - X.shape[1])
        self.sigma_ = float(np.sqrt(residuals @ residuals / dof))
        self.intercept_ = float(beta[0])
        self.coef_ = np.asarray(beta[1:], dtype=np.float64)
        self._last_values = y[-self.p :].copy()
        return self

    def forecast(self, h: int, *, level: float | None = None) -> ForecastResult:
        """Forecast ``h`` steps ahead with prediction intervals.

        Args:
            h: Forecast horizon (>= 1).
            level: Two-sided interval level; defaults to the level set at init.

        Returns:
            :class:`ForecastResult` with mean path and ``level`` bands whose width
            grows with horizon via the AR psi-weight recursion.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.coef_ is None or self._last_values is None:
            raise RuntimeError("ARForecaster is not fitted; call fit() first")
        if h < 1:
            raise ValueError("horizon h must be >= 1")
        lvl = self.level if level is None else float(level)
        if not 0.0 < lvl < 1.0:
            raise ValueError("level must be in (0, 1)")

        phi = self.coef_
        history = list(self._last_values)  # most recent value last
        means = np.empty(h, dtype=np.float64)
        for step in range(h):
            pred = self.intercept_ + sum(
                phi[k] * history[-1 - k] for k in range(self.p)
            )
            means[step] = pred
            history.append(pred)

        # MA(infinity) impulse-response weights: psi_0 = 1, psi_j = sum phi_k psi_{j-k}.
        psi = np.zeros(h, dtype=np.float64)
        psi[0] = 1.0
        for j in range(1, h):
            psi[j] = sum(phi[k] * psi[j - 1 - k] for k in range(min(j, self.p)))
        std_err = self.sigma_ * np.sqrt(np.cumsum(psi**2))
        z = _norm_ppf(0.5 + lvl / 2.0)
        return ForecastResult(
            mean=means,
            lower=means - z * std_err,
            upper=means + z * std_err,
            sigma=self.sigma_,
            level=lvl,
            horizon=h,
        )


class NaiveForecaster:
    """Random-walk baseline: forecast equals the last observed value.

    Implements the same ``fit``/``forecast`` protocol as :class:`ARForecaster` so it
    can run through :func:`backtest` and serve as the lift baseline every richer
    model must beat. Interval widths grow as ``sigma * sqrt(h)``, the random-walk
    forecast-error law, with ``sigma`` estimated from first differences.
    """

    def __init__(self, *, level: float = DEFAULT_LEVEL) -> None:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        self.level = float(level)
        self.last_: float | None = None
        self.sigma_: float = 0.0

    def fit(self, series: Sequence[float] | np.ndarray) -> NaiveForecaster:
        """Record the last value and the standard deviation of first differences."""
        y = np.asarray(series, dtype=np.float64).ravel()
        if y.size < 2:
            raise ValueError("need at least 2 observations")
        if not np.all(np.isfinite(y)):
            raise ValueError("series contains non-finite values")
        self.last_ = float(y[-1])
        diffs = np.diff(y)
        self.sigma_ = float(diffs.std(ddof=1)) if diffs.size >= 2 else float(abs(diffs[0]))
        return self

    def forecast(self, h: int, *, level: float | None = None) -> ForecastResult:
        """Flat mean path at the last value with sqrt-of-horizon interval growth."""
        if self.last_ is None:
            raise RuntimeError("NaiveForecaster is not fitted; call fit() first")
        if h < 1:
            raise ValueError("horizon h must be >= 1")
        lvl = self.level if level is None else float(level)
        if not 0.0 < lvl < 1.0:
            raise ValueError("level must be in (0, 1)")
        means = np.full(h, self.last_, dtype=np.float64)
        std_err = self.sigma_ * np.sqrt(np.arange(1, h + 1, dtype=np.float64))
        z = _norm_ppf(0.5 + lvl / 2.0)
        return ForecastResult(
            mean=means,
            lower=means - z * std_err,
            upper=means + z * std_err,
            sigma=self.sigma_,
            level=lvl,
            horizon=h,
        )


@dataclass
class EwmaVol:
    """RiskMetrics-style exponentially weighted moving-average volatility.

    The conditional variance recursion is
    ``var_t = lam * var_{t-1} + (1 - lam) * r_t^2`` with the classic RiskMetrics
    decay ``lam = 0.94`` for daily returns. The recursion is seeded with the mean
    squared return over the first ``init_window`` observations (documented
    look-ahead limited to that warm-up only).

    Attributes:
        lam: Decay factor in ``(0, 1)``; higher = slower adaptation.
        init_window: Number of initial observations used to seed the variance.
        variance_: Conditional variance series aligned with the fitted returns
            (``variance_[t]`` uses returns up to and including ``t``).
        vol_: ``sqrt(variance_)``.
    """

    lam: float = 0.94
    init_window: int = 20
    variance_: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    vol_: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.lam < 1.0:
            raise ValueError("lam must be in (0, 1)")
        if self.init_window < 1:
            raise ValueError("init_window must be >= 1")

    def fit(self, returns: Sequence[float] | np.ndarray) -> EwmaVol:
        """Run the EWMA recursion over a 1-D return series.

        Args:
            returns: Period returns ordered oldest to newest (at least 2 points).

        Returns:
            ``self`` with ``variance_`` and ``vol_`` populated.
        """
        r = np.asarray(returns, dtype=np.float64).ravel()
        if r.size < 2:
            raise ValueError("need at least 2 returns")
        if not np.all(np.isfinite(r)):
            raise ValueError("returns contain non-finite values")
        warm = min(self.init_window, r.size)
        prev = float(np.mean(r[:warm] ** 2))
        if prev <= 0.0:
            prev = 1e-12
        variance = np.empty(r.size, dtype=np.float64)
        one_minus = 1.0 - self.lam
        for t in range(r.size):
            prev = self.lam * prev + one_minus * r[t] * r[t]
            variance[t] = prev
        self.variance_ = variance
        self.vol_ = np.sqrt(variance)
        return self

    def update(self, r: float) -> float:
        """Streaming update with one new return; returns the new volatility."""
        if self.variance_.size == 0:
            raise RuntimeError("EwmaVol is not fitted; call fit() first")
        new_var = self.lam * float(self.variance_[-1]) + (1.0 - self.lam) * float(r) ** 2
        self.variance_ = np.append(self.variance_, new_var)
        self.vol_ = np.append(self.vol_, math.sqrt(new_var))
        return float(self.vol_[-1])

    @property
    def latest_vol(self) -> float:
        """Most recent conditional volatility estimate."""
        if self.vol_.size == 0:
            raise RuntimeError("EwmaVol is not fitted; call fit() first")
        return float(self.vol_[-1])

    def forecast(self, h: int = 1) -> np.ndarray:
        """h-step volatility forecast (flat: EWMA's variance forecast is constant)."""
        if h < 1:
            raise ValueError("horizon h must be >= 1")
        return np.full(h, self.latest_vol, dtype=np.float64)


def backtest(
    series: Sequence[float] | np.ndarray,
    model: ARForecaster | NaiveForecaster,
    window: int,
    *,
    horizon: int = 1,
    step: int = 1,
    level: float = DEFAULT_LEVEL,
) -> dict[str, float]:
    """Rolling-origin backtest of any ``fit``/``forecast`` model.

    At each origin ``t`` the model is refitted (in place) on the trailing
    ``window`` observations and asked for an ``horizon``-step forecast; the
    ``horizon``-th prediction is compared with the realised value.

    Args:
        series: Full history, oldest to newest.
        model: Object with ``fit(series)`` and ``forecast(h, level=...) ->
            ForecastResult``. The model is mutated by repeated refits.
        window: Length of the rolling training window.
        horizon: Steps ahead to evaluate (the last step of each forecast).
        step: Stride between consecutive origins (1 = every point).
        level: Interval level used for the coverage statistic.

    Returns:
        Dict with ``mae``, ``mape``, ``coverage`` (fraction of realised values
        inside the ``level`` interval), ``n`` (number of evaluations) and
        ``level``.

    Raises:
        ValueError: If the series is too short for a single evaluation.
    """
    y = np.asarray(series, dtype=np.float64).ravel()
    if window < 2:
        raise ValueError("window must be >= 2")
    if horizon < 1 or step < 1:
        raise ValueError("horizon and step must be >= 1")
    origins = range(window, y.size - horizon + 1, step)
    if len(origins) == 0:
        raise ValueError(
            f"series of length {y.size} is too short for window={window}, horizon={horizon}"
        )

    preds: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    actuals: list[float] = []
    for origin in origins:
        model.fit(y[origin - window : origin])
        fc = model.forecast(horizon, level=level)
        preds.append(float(fc.mean[-1]))
        lowers.append(float(fc.lower[-1]))
        uppers.append(float(fc.upper[-1]))
        actuals.append(float(y[origin + horizon - 1]))

    pred_arr = np.asarray(preds)
    actual_arr = np.asarray(actuals)
    lower_arr = np.asarray(lowers)
    upper_arr = np.asarray(uppers)
    errors = np.abs(actual_arr - pred_arr)
    eps = 1e-12
    mape = float(np.mean(errors / np.maximum(np.abs(actual_arr), eps)))
    coverage = float(np.mean((actual_arr >= lower_arr) & (actual_arr <= upper_arr)))
    return {
        "mae": float(errors.mean()),
        "mape": mape,
        "coverage": coverage,
        "n": float(len(preds)),
        "level": float(level),
    }
