"""Deterministic, numpy-only analytical fallbacks used by the packaged demos.

Every demo must run end-to-end with ``FINTWIN_OFFLINE=1``, no OpenAI key and no
network. When the federated twin exposes the corresponding simulator or model the
demos prefer it; when a capability is missing (for example in a partial install)
these fallbacks keep the demo honest: they produce the same shapes the platform
contracts require, ``SimulationResult`` with confidence intervals and a
calibration block, using stylised but defensible financial models.

All randomness flows through ``numpy.random.default_rng(seed)`` so every run is
exactly reproducible.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from fintwinos.core.types import SimulationResult

#: Canonical feature vector used by the AML triage demo. The compliance simulator
#: stand-ins and the fallback alert generator both emit exactly these names.
FEATURES: list[str] = [
    "amount_zscore",
    "structuring_score",
    "fan_in",
    "fan_out",
    "cycle_participation",
    "velocity",
    "dormancy_break",
    "cross_border",
]


# ---------------------------------------------------------------------------
# Treasury: cash ladder and liquidity stress
# ---------------------------------------------------------------------------


def cash_ladder(
    seed: int = 7,
    horizon_days: int = 14,
    opening_cash: float = 120.0,
    currency: str = "USD",
) -> list[dict[str, Any]]:
    """Build a deterministic daily cash ladder (all figures in millions).

    Inflows and outflows are drawn from gamma distributions calibrated so the
    book runs a mild structural outflow, which is what makes the squeeze
    rehearsal interesting.
    """
    rng = np.random.default_rng(seed)
    inflows = rng.gamma(shape=9.0, scale=1.15, size=horizon_days)
    outflows = rng.gamma(shape=10.0, scale=1.25, size=horizon_days)
    net = inflows - outflows
    closing = opening_cash + np.cumsum(net)
    return [
        {
            "day": day + 1,
            "currency": currency,
            "inflows": round(float(inflows[day]), 2),
            "outflows": round(float(outflows[day]), 2),
            "net": round(float(net[day]), 2),
            "closing_cash": round(float(closing[day]), 2),
        }
        for day in range(horizon_days)
    ]


def liquidity_stress_model(
    severity: float,
    seed: int = 7,
    scenario_name: str = "liquidity_stress",
    horizon_days: int = 30,
    n_paths: int = 500,
) -> dict[str, Any]:
    """Monte Carlo survival-days model for a USD liquidity squeeze.

    The counterbalancing capacity is a stylised contingency-funding-plan stack:
    unencumbered cash (day 1), repo-able HQLA at a haircut (day 2+) and a
    committed credit line (day 4+, after drawdown notice). Daily net outflows
    follow a lognormal whose location scales with ``severity``, with an early
    "run" multiplier that decays over the first week.

    Returns ``SimulationResult.model_dump()`` so callers can treat it exactly
    like the output of a registered platform simulator.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    days = np.arange(1, horizon_days + 1)

    # Counterbalancing capacity (USD mm), unlocking by day.
    available = 140.0 + (days >= 2) * (90.0 * 0.9) + (days >= 4) * 60.0

    outflow_scale = 3.0 + 24.0 * severity
    run_multiplier = 1.0 + 1.5 * severity * np.exp(-(days - 1) / 6.0)
    outflows = rng.lognormal(
        mean=np.log(outflow_scale), sigma=0.35, size=(n_paths, horizon_days)
    ) * run_multiplier[None, :]
    inflows = rng.gamma(shape=6.0, scale=1.0 - 0.6 * severity + 1e-9, size=(n_paths, horizon_days))
    cumulative_net = np.cumsum(outflows - inflows, axis=1)

    breached = cumulative_net > available[None, :]
    any_breach = breached.any(axis=1)
    first_breach = np.where(any_breach, breached.argmax(axis=1) + 1, horizon_days)
    survival = first_breach.astype(float)

    result = SimulationResult(
        simulator="liquidity_stress_fallback",
        scenario_name=scenario_name,
        metrics={
            "survival_days": float(np.median(survival)),
            "survival_days_mean": round(float(survival.mean()), 3),
            "breach_probability": round(float(any_breach.mean()), 4),
            "peak_cumulative_outflow": round(float(np.median(cumulative_net[:, -1])), 2),
            "counterbalancing_capacity": float(available[-1]),
        },
        series={
            "cumulative_net_outflow_p50": [
                round(float(v), 2) for v in np.median(cumulative_net, axis=0)
            ],
        },
        confidence={
            "survival_days": [
                float(np.percentile(survival, 5)),
                float(np.percentile(survival, 95)),
            ],
            "breach_probability": _binomial_ci(float(any_breach.mean()), n_paths),
        },
        calibration={
            "method": "monte_carlo",
            "n_paths": n_paths,
            "assumptions": "stylised CFP stack: cash day-1, repo HQLA day-2 (10% haircut), "
            "committed line day-4; lognormal outflows with decaying run multiplier",
            "severity": severity,
        },
        warnings=[] if severity < 0.9 else ["severity at model boundary; tails understated"],
        seed=seed,
    )
    return result.model_dump()


def _binomial_ci(p: float, n: int) -> list[float]:
    """Normal-approximation 95% confidence interval for a proportion."""
    half = 1.96 * float(np.sqrt(max(p * (1.0 - p), 1e-12) / n))
    return [round(max(0.0, p - half), 4), round(min(1.0, p + half), 4)]


# ---------------------------------------------------------------------------
# Customer operations: complaints queue
# ---------------------------------------------------------------------------


def queue_simulation(
    staff: int,
    arrival_multiplier: float = 1.6,
    horizon_days: int = 10,
    sla_hours: float = 24.0,
    seed: int = 7,
    n_reps: int = 200,
    base_arrivals_per_day: float = 90.0,
    cases_per_agent_per_day: float = 12.0,
    initial_backlog: float = 35.0,
) -> dict[str, Any]:
    """Fluid-approximation Monte Carlo of a complaints queue under a spike.

    Uses common random numbers: with the same ``seed`` the arrival and capacity
    noise draws are identical regardless of ``staff``, so staffing
    counterfactuals are an apples-to-apples comparison (more staff can never
    look worse purely through sampling luck).

    Waiting time is estimated via Little's law (``backlog / capacity`` days) and
    an arriving case breaches the SLA when the queue ahead of it implies a wait
    beyond ``sla_hours``.
    """
    rng = np.random.default_rng(seed)
    arrivals = rng.poisson(
        base_arrivals_per_day * arrival_multiplier, size=(n_reps, horizon_days)
    ).astype(float)
    capacity_noise = rng.lognormal(mean=0.0, sigma=0.08, size=(n_reps, horizon_days))
    capacity = staff * cases_per_agent_per_day * capacity_noise

    backlog = np.full(n_reps, initial_backlog)
    backlog_path = np.empty((n_reps, horizon_days))
    wait_hours = np.empty((n_reps, horizon_days))
    for day in range(horizon_days):
        backlog = np.maximum(backlog + arrivals[:, day] - capacity[:, day], 0.0)
        backlog_path[:, day] = backlog
        wait_hours[:, day] = 24.0 * backlog / capacity[:, day]

    breach = (wait_hours > sla_hours).astype(float)
    breach_rate_per_rep = (arrivals * breach).sum(axis=1) / arrivals.sum(axis=1)
    mean_wait_per_rep = wait_hours.mean(axis=1)
    backlog_end = backlog_path[:, -1]

    result = SimulationResult(
        simulator="ops_queue_fallback",
        scenario_name=f"complaints_spike_staff_{staff}",
        metrics={
            "avg_wait_hours": round(float(mean_wait_per_rep.mean()), 3),
            "p95_wait_hours": round(float(np.percentile(wait_hours, 95)), 3),
            "sla_breach_rate": round(float(breach_rate_per_rep.mean()), 4),
            "backlog_end": round(float(backlog_end.mean()), 2),
            "staff": float(staff),
            "arrivals_per_day": round(float(arrivals.mean()), 2),
        },
        series={
            "backlog_mean": [round(float(v), 2) for v in backlog_path.mean(axis=0)],
        },
        confidence={
            "avg_wait_hours": [
                round(float(np.percentile(mean_wait_per_rep, 2.5)), 3),
                round(float(np.percentile(mean_wait_per_rep, 97.5)), 3),
            ],
            "sla_breach_rate": [
                round(float(np.percentile(breach_rate_per_rep, 2.5)), 4),
                round(float(np.percentile(breach_rate_per_rep, 97.5)), 4),
            ],
            "backlog_end": [
                round(float(np.percentile(backlog_end, 2.5)), 2),
                round(float(np.percentile(backlog_end, 97.5)), 2),
            ],
        },
        calibration={
            "method": "fluid_queue_monte_carlo",
            "n_reps": n_reps,
            "common_random_numbers": True,
            "sla_hours": sla_hours,
        },
        seed=seed,
    )
    return result.model_dump()


# ---------------------------------------------------------------------------
# AML: synthetic alert population, scorer, ranking metrics
# ---------------------------------------------------------------------------


def make_alert_population(
    n_alerts: int,
    ring_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Generate a labelled synthetic AML alert population.

    Ring-member alerts are drawn from shifted feature distributions (higher
    fan-in/out, structuring just below reporting thresholds, cycle
    participation), which is what gives a trained scorer a real precision /
    recall trade-off to expose.

    Returns ``(alerts, X, y)`` where ``X`` has columns in :data:`FEATURES`
    order and ``y`` is the ground-truth ring membership.
    """
    n_ring = max(1, int(round(n_alerts * ring_fraction)))
    labels = np.zeros(n_alerts, dtype=int)
    ring_idx = rng.choice(n_alerts, size=n_ring, replace=False)
    labels[ring_idx] = 1

    X = np.empty((n_alerts, len(FEATURES)))
    for i in range(n_alerts):
        if labels[i] == 1:
            row = [
                rng.normal(1.2, 0.8),                # amount_zscore
                rng.beta(5.0, 2.5),                  # structuring_score
                float(rng.poisson(6)),               # fan_in
                float(rng.poisson(5)),               # fan_out
                float(rng.random() < 0.7),           # cycle_participation
                rng.lognormal(0.8, 0.4),             # velocity
                float(rng.random() < 0.35),          # dormancy_break
                float(rng.random() < 0.55),          # cross_border
            ]
        else:
            row = [
                rng.normal(0.0, 1.0),
                rng.beta(1.2, 6.0),
                float(rng.poisson(2)),
                float(rng.poisson(2)),
                float(rng.random() < 0.03),
                rng.lognormal(0.0, 0.4),
                float(rng.random() < 0.05),
                float(rng.random() < 0.15),
            ]
        X[i] = row

    alerts = [
        {
            "alert_id": f"alr_demo_{i:04d}",
            "kind": "aml_transaction_pattern",
            "customer_id": f"cust_{1000 + i}",
            "features": {name: round(float(X[i, j]), 4) for j, name in enumerate(FEATURES)},
            "is_ring": int(labels[i]),
        }
        for i in range(n_alerts)
    ]
    return alerts, X, labels


def coerce_alert_queue(
    data: Any,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray] | None:
    """Adapt a simulator payload into ``(alerts, X, y)`` if, and only if, it
    carries the full canonical feature set *and* ground-truth labels.

    Anything less and the demo falls back to its own labelled population, so
    the precision/recall table is always computed against known truth.
    """
    items: list[Any] | None = None
    if isinstance(data, dict):
        for key in ("alerts", "alert_queue", "queue"):
            candidate = data.get(key)
            if isinstance(candidate, list) and candidate:
                items = candidate
                break
    elif isinstance(data, list) and data:
        items = data
    if items is None:
        return None

    rows: list[list[float]] = []
    labels: list[int] = []
    alerts: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return None
        feats = item.get("features") if isinstance(item.get("features"), dict) else item
        vector: list[float] = []
        for name in FEATURES:
            value = feats.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                return None
            vector.append(float(value))
        label = item.get("is_ring", item.get("label", item.get("ground_truth")))
        if isinstance(label, bool):
            label = int(label)
        if not isinstance(label, int | float):
            return None
        rows.append(vector)
        labels.append(int(label))
        alerts.append(
            {
                "alert_id": str(item.get("alert_id", f"alr_sim_{i:04d}")),
                "kind": str(item.get("kind", "aml_transaction_pattern")),
                "customer_id": str(item.get("customer_id", f"cust_{i:04d}")),
                "features": {name: vector[j] for j, name in enumerate(FEATURES)},
                "is_ring": int(label),
            }
        )
    if len(rows) < 20:
        return None
    return alerts, np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


class NumpyLogisticScorer:
    """L2-regularised logistic regression implemented in plain numpy.

    Full-batch gradient descent on standardised features with a zero
    initialisation, so ``fit`` + ``predict_proba`` are fully deterministic.
    Serves as the AML alert scorer whenever ``fintwinos.models.graph`` is
    unavailable or exposes an incompatible interface.
    """

    def __init__(self, learning_rate: float = 0.4, n_iter: int = 600, l2: float = 1e-3):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NumpyLogisticScorer:
        """Fit on a feature matrix ``X`` and binary labels ``y``."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._mean = X.mean(axis=0)
        self._std = np.maximum(X.std(axis=0), 1e-6)
        Z = self._design(X)
        w = np.zeros(Z.shape[1])
        n = len(y)
        for _ in range(self.n_iter):
            p = _sigmoid(Z @ w)
            grad = Z.T @ (p - y) / n + self.l2 * w
            w -= self.learning_rate * grad
        self._weights = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``P(ring | features)`` for each row of ``X``."""
        if self._weights is None:
            raise RuntimeError("scorer is not fitted; call fit(X, y) first")
        return _sigmoid(self._design(np.asarray(X, dtype=float)) @ self._weights)

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self._mean) / self._std
        return np.hstack([np.ones((len(Z), 1)), Z])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


#: Class names probed, in order, on ``fintwinos.models.graph``.
_SCORER_CANDIDATES = (
    "AmlSubgraphScorer",
    "AMLScorer",
    "AMLAlertScorer",
    "AlertScorer",
    "GraphAlertScorer",
    "RingScorer",
)

_FALLBACK_SCORER_SOURCE = "fintwinos.demos._fallbacks.NumpyLogisticScorer"


def resolve_aml_scorer() -> tuple[Any, str]:
    """Locate an AML alert scorer, preferring ``fintwinos.models.graph``.

    Probes a small set of conventional class names for a constructable object
    exposing ``fit`` plus a scoring method; classes accepting a
    ``feature_names`` keyword are configured with the demo's canonical
    :data:`FEATURES` columns. Falls back to :class:`NumpyLogisticScorer` so
    the demo never depends on the sibling module being installed.
    """
    try:
        module = importlib.import_module("fintwinos.models.graph")
    except Exception:
        return NumpyLogisticScorer(), _FALLBACK_SCORER_SOURCE
    for attr in _SCORER_CANDIDATES:
        cls = getattr(module, attr, None)
        if cls is None:
            continue
        instance: Any = None
        for kwargs in ({"feature_names": list(FEATURES)}, {}):
            try:
                instance = cls(**kwargs)
                break
            except Exception:
                continue
        if instance is None:
            continue
        has_fit = callable(getattr(instance, "fit", None))
        has_score = any(
            callable(getattr(instance, method, None))
            for method in ("predict_proba", "decision_function", "predict")
        )
        if has_fit and has_score:
            return instance, f"fintwinos.models.graph.{attr}"
    return NumpyLogisticScorer(), _FALLBACK_SCORER_SOURCE


def train_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_queue: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Train the resolved scorer on synthetic data and score the live queue.

    If the resolved third-party scorer fails at any point the deterministic
    numpy scorer takes over, so the demo always produces a ranked queue.
    Scores are normalised into ``[0, 1]``.
    """
    scorer, source = resolve_aml_scorer()
    try:
        scorer.fit(X_train, y_train)
        scores = _extract_scores(scorer, X_queue)
    except Exception:
        scorer, source = NumpyLogisticScorer(), _FALLBACK_SCORER_SOURCE
        scorer.fit(X_train, y_train)
        scores = scorer.predict_proba(X_queue)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.min() < 0.0 or scores.max() > 1.0:
        span = scores.max() - scores.min()
        scores = (scores - scores.min()) / (span if span > 0 else 1.0)
    return scores, source


def _extract_scores(scorer: Any, X: np.ndarray) -> np.ndarray:
    """Pull a per-row score out of whichever scoring method the object offers."""
    for method in ("predict_proba", "decision_function", "predict"):
        fn = getattr(scorer, method, None)
        if not callable(fn):
            continue
        try:
            out = np.asarray(fn(X), dtype=float)
        except Exception:
            continue
        if out.ndim == 2:
            out = out[:, -1]
        if out.shape[0] == len(X):
            return out
    raise RuntimeError("scorer exposes no usable scoring method")


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the pairwise Mann–Whitney statistic."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def precision_recall_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    ks: list[int],
) -> list[dict[str, float]]:
    """Precision and recall of the top-``k`` triage cut for each ``k``.

    Sorting is stable on descending score so equal-scored alerts keep their
    queue order and the table is reproducible.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    total_pos = int(ranked.sum())
    rows: list[dict[str, float]] = []
    for k in ks:
        kk = min(int(k), len(ranked))
        if kk <= 0:
            continue
        tp = int(ranked[:kk].sum())
        rows.append(
            {
                "k": float(kk),
                "precision": round(tp / kk, 4),
                "recall": round(tp / total_pos, 4) if total_pos else 0.0,
            }
        )
    return rows
