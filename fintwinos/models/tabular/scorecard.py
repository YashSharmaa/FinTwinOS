"""Classical credit-risk scorecard: quantile bins, weight of evidence, reason codes.

This is the industry-standard, regulator-friendly tabular model: each feature is
quantile-binned, each bin is replaced by its weight of evidence
``WOE = ln(share_of_goods / share_of_bads)`` (Laplace-smoothed), and a numpy
logistic regression is fitted on the WOE-transformed features. The result is a
model whose every prediction decomposes exactly into per-feature contributions —
the *reason codes* lenders are required to return with adverse decisions — plus a
points scale (PDO calibration) and a decile calibration table for monitoring.

Sign conventions used throughout:

- ``y = 1`` is the *bad* (default / fraud) outcome.
- WOE is ``ln(good share / bad share)``, so higher WOE = safer bin and the fitted
  weights are typically negative.
- A reason-code ``contribution`` is ``weight * WOE`` — its share of the bad
  log-odds relative to the population average (a bin with WOE 0 behaves exactly
  like the overall population and contributes nothing).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _quantile_edges(column: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior quantile cut points for one feature (deduplicated, may be empty)."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    if quantiles.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.unique(np.quantile(column, quantiles))


def _bin_indices(column: np.ndarray, interior_edges: np.ndarray) -> np.ndarray:
    """Map values to bin ids ``0..len(interior_edges)`` (right-closed at each edge)."""
    return np.searchsorted(interior_edges, column, side="right")


def _bin_label(interior_edges: np.ndarray, bin_id: int) -> str:
    """Human-readable half-open interval label for a bin id."""
    lo = -math.inf if bin_id == 0 else float(interior_edges[bin_id - 1])
    hi = math.inf if bin_id >= interior_edges.size else float(interior_edges[bin_id])
    return f"({lo:.6g}, {hi:.6g}]"


class Scorecard:
    """Quantile-binned WOE scorecard with a numpy logistic layer on top.

    Lifecycle: ``fit(X, y)`` learns bins, WOE tables and logistic weights; then
    ``predict_proba`` scores new records, ``reason_codes`` explains a single
    record, ``points`` maps probabilities to a familiar score scale and
    ``calibration_table`` summarises observed vs predicted risk by decile.

    Attributes:
        feature_names_: Feature names captured at fit time.
        edges_: Per-feature interior quantile edges.
        woe_: Per-feature WOE value for each bin.
        iv_: Per-feature information value (a standard predictive-power summary).
        weights_ / bias_: Fitted logistic coefficients over the WOE features.
        event_rate_: Overall bad rate of the training sample.
    """

    def __init__(
        self,
        n_bins: int = 5,
        *,
        learning_rate: float = 0.5,
        epochs: int = 2000,
        l2: float = 1e-3,
        smoothing: float = 0.5,
        seed: int = 7,
    ) -> None:
        """Configure the scorecard.

        Args:
            n_bins: Target number of quantile bins per feature (ties can reduce it).
            learning_rate: Gradient-descent step size for the logistic layer.
            epochs: Number of full-batch gradient steps.
            l2: L2 penalty on the logistic weights (bias unpenalised).
            smoothing: Laplace count added to each bin's good/bad totals so empty
                bins get a finite, shrunk-to-zero WOE.
            seed: Seed for the ``numpy.random.default_rng`` weight initialisation.
        """
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if learning_rate <= 0 or epochs < 1:
            raise ValueError("learning_rate must be positive and epochs >= 1")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.n_bins = int(n_bins)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.smoothing = float(smoothing)
        self.seed = int(seed)

        self.feature_names_: list[str] | None = None
        self.edges_: list[np.ndarray] = []
        self.woe_: list[np.ndarray] = []
        self.iv_: list[float] = []
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.event_rate_: float = 0.0

    # -- fitting ---------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: Sequence[float] | np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> Scorecard:
        """Learn bins, WOE tables and the logistic layer.

        Args:
            X: Raw feature matrix of shape ``(n, d)``.
            y: Binary outcomes, 1 = bad.
            feature_names: Optional names for the ``d`` columns; defaults to
                ``feature_0..feature_{d-1}``.

        Returns:
            ``self`` (fluent style).

        Raises:
            ValueError: On shape mismatches or single-class labels.
        """
        X = np.asarray(X, dtype=np.float64)
        yv = (np.asarray(y, dtype=np.float64).ravel() > 0).astype(np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        n, d = X.shape
        if n != yv.size:
            raise ValueError("X and y must have the same number of rows")
        n_bad = float(yv.sum())
        n_good = float(n - n_bad)
        if n_bad == 0 or n_good == 0:
            raise ValueError("fit requires both good (0) and bad (1) outcomes")
        names = list(feature_names) if feature_names is not None else [
            f"feature_{j}" for j in range(d)
        ]
        if len(names) != d:
            raise ValueError(f"expected {d} feature names, got {len(names)}")

        self.feature_names_ = names
        self.event_rate_ = n_bad / n
        self.edges_ = []
        self.woe_ = []
        self.iv_ = []
        for j in range(d):
            column = X[:, j]
            interior = _quantile_edges(column, self.n_bins)
            bins = _bin_indices(column, interior)
            n_bin = interior.size + 1
            woe = np.zeros(n_bin, dtype=np.float64)
            iv = 0.0
            for b in range(n_bin):
                mask = bins == b
                bad_b = float(yv[mask].sum())
                good_b = float(mask.sum()) - bad_b
                dist_good = (good_b + self.smoothing) / (n_good + self.smoothing * n_bin)
                dist_bad = (bad_b + self.smoothing) / (n_bad + self.smoothing * n_bin)
                woe[b] = math.log(dist_good / dist_bad)
                iv += (dist_good - dist_bad) * woe[b]
            self.edges_.append(interior)
            self.woe_.append(woe)
            self.iv_.append(iv)

        W = self.transform(X)
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0.0, 0.01, size=d)
        b = math.log(self.event_rate_ / (1.0 - self.event_rate_))  # warm-start at base rate
        for _ in range(self.epochs):
            p = _sigmoid(W @ w + b)
            grad_w = W.T @ (p - yv) / n + self.l2 * w
            grad_b = float(np.mean(p - yv))
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
        self.weights_ = w
        self.bias_ = float(b)
        return self

    def _check_fitted(self) -> None:
        if self.weights_ is None or self.feature_names_ is None:
            raise RuntimeError("Scorecard is not fitted; call fit() first")

    # -- transforms and predictions ----------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace raw feature values with their fitted per-bin WOE values."""
        if not self.edges_:
            raise RuntimeError("Scorecard is not fitted; call fit() first")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != len(self.edges_):
            raise ValueError(
                f"X has {X.shape[1]} columns but the scorecard was fitted on {len(self.edges_)}"
            )
        W = np.empty_like(X)
        for j, (interior, woe) in enumerate(zip(self.edges_, self.woe_, strict=True)):
            W[:, j] = woe[_bin_indices(X[:, j], interior)]
        return W

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the bad outcome (``y = 1``) for each row."""
        self._check_fitted()
        W = self.transform(X)
        assert self.weights_ is not None
        return _sigmoid(W @ self.weights_ + self.bias_)

    def points(
        self,
        X: np.ndarray,
        *,
        base_points: float = 600.0,
        pdo: float = 50.0,
        base_odds: float = 50.0,
    ) -> np.ndarray:
        """Map model log-odds to a conventional points scale.

        Uses the standard PDO ("points to double the odds") calibration:
        a record with good:bad odds of ``base_odds`` scores ``base_points``, and
        every doubling of the good:bad odds adds ``pdo`` points.

        Args:
            X: Raw feature matrix.
            base_points: Score anchored at ``base_odds``.
            pdo: Points required to double the good:bad odds.
            base_odds: Good:bad odds at the anchor score.

        Returns:
            Array of scores (higher = safer).
        """
        self._check_fitted()
        if pdo <= 0 or base_odds <= 0:
            raise ValueError("pdo and base_odds must be positive")
        W = self.transform(X)
        assert self.weights_ is not None
        log_odds_bad = W @ self.weights_ + self.bias_
        factor = pdo / math.log(2.0)
        offset = base_points - factor * math.log(base_odds)
        return offset + factor * (-log_odds_bad)

    # -- explainability ------------------------------------------------------------------

    def reason_codes(self, x: Sequence[float] | np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
        """Explain one record: top feature contributions to its bad log-odds.

        Args:
            x: A single raw feature vector of length ``d``.
            top_k: Number of reasons to return.

        Returns:
            Up to ``top_k`` dicts ordered by descending ``contribution`` (the most
            risk-increasing reason first). Each dict carries ``feature``, ``bin``
            (interval label), ``woe``, ``weight`` and ``contribution``
            (``weight * woe``, in bad-log-odds units relative to the population
            average).
        """
        self._check_fitted()
        xv = np.asarray(x, dtype=np.float64).ravel()
        assert self.weights_ is not None and self.feature_names_ is not None
        if xv.size != len(self.feature_names_):
            raise ValueError(
                f"x has {xv.size} values but the scorecard was fitted on "
                f"{len(self.feature_names_)} features"
            )
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        entries: list[dict[str, Any]] = []
        for j, name in enumerate(self.feature_names_):
            interior = self.edges_[j]
            bin_id = int(_bin_indices(np.array([xv[j]]), interior)[0])
            woe = float(self.woe_[j][bin_id])
            weight = float(self.weights_[j])
            entries.append(
                {
                    "feature": name,
                    "bin": _bin_label(interior, bin_id),
                    "woe": woe,
                    "weight": weight,
                    "contribution": weight * woe,
                }
            )
        entries.sort(key=lambda e: e["contribution"], reverse=True)
        return entries[:top_k]

    # -- monitoring ----------------------------------------------------------------------

    def calibration_table(
        self,
        X: np.ndarray,
        y: Sequence[float] | np.ndarray,
        deciles: int = 10,
    ) -> pd.DataFrame:
        """Observed vs predicted risk by predicted-probability decile.

        Args:
            X: Raw feature matrix.
            y: Realised binary outcomes (1 = bad).
            deciles: Number of probability bins (quantile-based; ties may merge).

        Returns:
            DataFrame ordered by increasing predicted risk with columns
            ``decile`` (1-based), ``n``, ``mean_pred`` (mean predicted bad
            probability), ``bad_rate`` (observed) and ``lift`` (observed rate over
            the overall rate). ``n`` sums to ``len(y)``.
        """
        self._check_fitted()
        if deciles < 2:
            raise ValueError("deciles must be >= 2")
        yv = (np.asarray(y, dtype=np.float64).ravel() > 0).astype(np.float64)
        p = self.predict_proba(X)
        if p.size != yv.size:
            raise ValueError("X and y must have the same number of rows")
        overall = float(yv.mean()) if yv.size else 0.0
        interior = _quantile_edges(p, deciles)
        bins = _bin_indices(p, interior)
        rows: list[dict[str, Any]] = []
        for b in range(interior.size + 1):
            mask = bins == b
            count = int(mask.sum())
            if count == 0:
                continue
            bad_rate = float(yv[mask].mean())
            rows.append(
                {
                    "decile": len(rows) + 1,
                    "n": count,
                    "mean_pred": float(p[mask].mean()),
                    "bad_rate": bad_rate,
                    "lift": bad_rate / overall if overall > 0 else float("nan"),
                }
            )
        return pd.DataFrame(rows, columns=["decile", "n", "mean_pred", "bad_rate", "lift"])
