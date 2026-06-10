"""Tests for the WOE scorecard: discrimination, reason codes, calibration, points."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fintwinos.models.graph.aml_scorer import roc_auc
from fintwinos.models.tabular.scorecard import Scorecard


def make_credit_data(n: int = 3000, seed: int = 5) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic credit sample: utilisation drives risk up, income drives it down."""
    rng = np.random.default_rng(seed)
    utilisation = rng.normal(0.0, 1.0, n)
    income = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)
    logit = -1.0 + 1.4 * utilisation - 1.1 * income
    p = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, p).astype(float)
    X = np.column_stack([utilisation, income, noise])
    return X, y, ["utilisation", "income", "noise"]


@pytest.fixture(scope="module")
def fitted() -> tuple[Scorecard, np.ndarray, np.ndarray, list[str]]:
    X, y, names = make_credit_data()
    card = Scorecard(n_bins=6, seed=7).fit(X, y, feature_names=names)
    return card, X, y, names


class TestScorecardFit:
    def test_discriminates(self, fitted):
        card, X, y, _ = fitted
        assert roc_auc(y, card.predict_proba(X)) > 0.75

    def test_informative_features_have_higher_iv_than_noise(self, fitted):
        card, _, _, names = fitted
        iv = dict(zip(names, card.iv_, strict=True))
        assert iv["utilisation"] > iv["noise"]
        assert iv["income"] > iv["noise"]

    def test_fit_is_deterministic(self):
        X, y, names = make_credit_data(n=800, seed=2)
        a = Scorecard(n_bins=5, seed=7).fit(X, y, feature_names=names)
        b = Scorecard(n_bins=5, seed=7).fit(X, y, feature_names=names)
        np.testing.assert_array_equal(a.weights_, b.weights_)
        assert a.bias_ == b.bias_

    def test_single_class_raises(self):
        X = np.random.default_rng(0).normal(size=(50, 2))
        with pytest.raises(ValueError):
            Scorecard().fit(X, np.zeros(50))

    def test_wrong_name_count_raises(self):
        X, y, _ = make_credit_data(n=200, seed=1)
        with pytest.raises(ValueError):
            Scorecard().fit(X, y, feature_names=["only_one"])

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            Scorecard().predict_proba(np.zeros((2, 3)))


class TestReasonCodes:
    def test_ordered_by_descending_contribution(self, fitted):
        card, X, _, _ = fitted
        codes = card.reason_codes(X[0], top_k=3)
        contributions = [c["contribution"] for c in codes]
        assert contributions == sorted(contributions, reverse=True)

    def test_full_ranking_covers_all_features_in_order(self, fitted):
        card, X, _, names = fitted
        codes = card.reason_codes(X[1], top_k=len(names))
        assert len(codes) == len(names)
        contributions = [c["contribution"] for c in codes]
        assert contributions == sorted(contributions, reverse=True)
        assert {c["feature"] for c in codes} == set(names)

    def test_high_risk_record_blames_informative_feature(self, fitted):
        card, _, _, _ = fitted
        # extreme utilisation, very negative income, neutral noise -> risky record
        risky = np.array([3.0, -3.0, 0.0])
        codes = card.reason_codes(risky, top_k=3)
        assert codes[0]["feature"] in {"utilisation", "income"}
        assert codes[0]["contribution"] > 0  # pushes towards bad

    def test_contribution_is_weight_times_woe(self, fitted):
        card, X, _, _ = fitted
        for code in card.reason_codes(X[2], top_k=3):
            assert code["contribution"] == pytest.approx(code["weight"] * code["woe"])

    def test_bin_labels_are_intervals(self, fitted):
        card, X, _, _ = fitted
        for code in card.reason_codes(X[3], top_k=3):
            assert code["bin"].startswith("(") and code["bin"].endswith("]")

    def test_top_k_limits_output(self, fitted):
        card, X, _, _ = fitted
        assert len(card.reason_codes(X[0], top_k=2)) == 2

    def test_wrong_length_raises(self, fitted):
        card, _, _, _ = fitted
        with pytest.raises(ValueError):
            card.reason_codes([1.0, 2.0])


class TestCalibrationTable:
    def test_structure_and_totals(self, fitted):
        card, X, y, _ = fitted
        table = card.calibration_table(X, y, deciles=10)
        assert isinstance(table, pd.DataFrame)
        assert list(table.columns) == ["decile", "n", "mean_pred", "bad_rate", "lift"]
        assert table["n"].sum() == len(y)
        assert len(table) <= 10
        assert list(table["decile"]) == list(range(1, len(table) + 1))

    def test_risk_orders_across_deciles(self, fitted):
        card, X, y, _ = fitted
        table = card.calibration_table(X, y, deciles=10)
        assert table["mean_pred"].is_monotonic_increasing
        assert table["bad_rate"].iloc[-1] > table["bad_rate"].iloc[0]

    def test_lift_anchored_to_overall_rate(self, fitted):
        card, X, y, _ = fitted
        table = card.calibration_table(X, y, deciles=10)
        weighted = (table["bad_rate"] * table["n"]).sum() / table["n"].sum()
        assert weighted == pytest.approx(y.mean())

    def test_predictions_roughly_calibrated(self, fitted):
        card, X, y, _ = fitted
        table = card.calibration_table(X, y, deciles=10)
        for _, row in table.iterrows():
            assert row["bad_rate"] == pytest.approx(row["mean_pred"], abs=0.12)


class TestPoints:
    def test_higher_risk_means_fewer_points(self, fitted):
        card, _, _, _ = fitted
        risky = np.array([[3.0, -3.0, 0.0]])
        safe = np.array([[-3.0, 3.0, 0.0]])
        assert card.points(risky)[0] < card.points(safe)[0]

    def test_pdo_doubles_odds(self, fitted):
        card, X, _, _ = fitted
        # +pdo points <=> good:bad odds doubled <=> bad log-odds reduced by ln 2
        pts = card.points(X[:50], base_points=600, pdo=50, base_odds=50)
        W = card.transform(X[:50])
        log_odds_bad = W @ card.weights_ + card.bias_
        order = np.argsort(pts)
        delta_pts = pts[order[-1]] - pts[order[0]]
        delta_log_odds = log_odds_bad[order[0]] - log_odds_bad[order[-1]]
        assert delta_pts == pytest.approx(50.0 / np.log(2.0) * delta_log_odds)
