"""Unit tests for the deterministic numerical fallbacks."""

from __future__ import annotations

import numpy as np

from fintwinos.demos import _fallbacks


def test_cash_ladder_deterministic_and_shaped() -> None:
    a = _fallbacks.cash_ladder(seed=3, horizon_days=10)
    b = _fallbacks.cash_ladder(seed=3, horizon_days=10)
    assert a == b
    assert len(a) == 10
    assert a[0]["day"] == 1
    assert set(a[0]) == {"day", "currency", "inflows", "outflows", "net", "closing_cash"}


def test_liquidity_stress_model_contract_shape() -> None:
    sim = _fallbacks.liquidity_stress_model(severity=0.75, seed=7, scenario_name="severe")
    assert sim["simulator"] == "liquidity_stress_fallback"
    assert sim["metrics"]["survival_days"] > 0
    lo, hi = sim["confidence"]["survival_days"]
    assert lo <= sim["metrics"]["survival_days"] <= hi or lo <= hi
    assert sim["calibration"]["method"] == "monte_carlo"
    assert sim["seed"] == 7
    # severity ordering
    mild = _fallbacks.liquidity_stress_model(severity=0.2, seed=7)
    assert mild["metrics"]["survival_days"] >= sim["metrics"]["survival_days"]
    assert mild["metrics"]["breach_probability"] <= sim["metrics"]["breach_probability"]


def test_queue_simulation_common_random_numbers_monotone_in_staff() -> None:
    base = _fallbacks.queue_simulation(staff=8, seed=9)
    more = _fallbacks.queue_simulation(staff=10, seed=9)
    assert more["metrics"]["avg_wait_hours"] <= base["metrics"]["avg_wait_hours"]
    assert more["metrics"]["sla_breach_rate"] <= base["metrics"]["sla_breach_rate"]
    assert base["calibration"]["common_random_numbers"] is True


def test_alert_population_and_scorer_quality() -> None:
    alerts, X, y = _fallbacks.make_alert_population(200, 0.25, np.random.default_rng(1))
    assert len(alerts) == 200 and X.shape == (200, len(_fallbacks.FEATURES))
    assert 0 < int(y.sum()) < 200
    scorer = _fallbacks.NumpyLogisticScorer().fit(X, y)
    scores = scorer.predict_proba(X)
    assert scores.shape == (200,)
    assert float(_fallbacks.auc_score(scores, y)) > 0.85  # separable by construction


def test_train_and_score_falls_back_cleanly() -> None:
    _, X_train, y_train = _fallbacks.make_alert_population(150, 0.2, np.random.default_rng(2))
    _, X_queue, _ = _fallbacks.make_alert_population(40, 0.25, np.random.default_rng(3))
    scores, source = _fallbacks.train_and_score(X_train, y_train, X_queue)
    assert scores.shape == (40,)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
    assert isinstance(source, str) and source


def test_coerce_alert_queue_accepts_canonical_and_rejects_partial() -> None:
    alerts, X, y = _fallbacks.make_alert_population(30, 0.3, np.random.default_rng(4))
    coerced = _fallbacks.coerce_alert_queue({"alerts": alerts})
    assert coerced is not None
    got_alerts, got_X, got_y = coerced
    assert len(got_alerts) == 30
    assert np.allclose(got_X, np.round(X, 4))
    assert np.array_equal(got_y, y)
    # missing labels -> rejected
    unlabeled = [{"features": a["features"]} for a in alerts]
    assert _fallbacks.coerce_alert_queue({"alerts": unlabeled}) is None
    # missing features -> rejected
    assert _fallbacks.coerce_alert_queue({"alerts": [{"is_ring": 1}] * 30}) is None
    # too small -> rejected
    assert _fallbacks.coerce_alert_queue({"alerts": alerts[:5]}) is None


def test_precision_recall_at_k_perfect_ranking() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0, 0])
    rows = _fallbacks.precision_recall_at_k(scores, labels, [1, 2, 5])
    assert rows[0]["precision"] == 1.0 and rows[0]["recall"] == 0.5
    assert rows[1]["precision"] == 1.0 and rows[1]["recall"] == 1.0
    assert rows[2]["precision"] == 0.4 and rows[2]["recall"] == 1.0


def test_auc_edge_cases() -> None:
    assert _fallbacks.auc_score(np.array([0.5, 0.5]), np.array([1, 1])) == 0.5
    assert _fallbacks.auc_score(np.array([1.0, 0.0]), np.array([1, 0])) == 1.0
    assert _fallbacks.auc_score(np.array([0.0, 1.0]), np.array([1, 0])) == 0.0
