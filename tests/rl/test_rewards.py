"""Hand-checked tests for the founding-brief reward and ES mathematics."""

from __future__ import annotations

import numpy as np
import pytest

from fintwinos.rl.rewards import BriefWeights, expected_shortfall, risk_sensitive_reward


class TestBriefWeights:
    def test_founding_brief_defaults(self):
        w = BriefWeights()
        assert w.expected_shortfall == 2.0
        assert w.policy_violation == 5.0
        assert w.human_minutes == 1.0
        assert w.latency == 0.5
        assert w.business_gain == 1.0

    def test_immutable(self):
        with pytest.raises(AttributeError):
            BriefWeights().latency = 9.0  # type: ignore[misc]


class TestRiskSensitiveReward:
    def test_hand_computed_default_weights(self):
        # 10 - 2*2 - 5*1 - 1*3 - 0.5*4 = 10 - 4 - 5 - 3 - 2 = -4
        reward = risk_sensitive_reward(
            business_gain=10.0,
            expected_shortfall_penalty=2.0,
            policy_violation_penalty=1.0,
            human_minutes_used=3.0,
            latency_penalty=4.0,
        )
        assert reward == pytest.approx(-4.0)

    def test_zero_penalties_pass_gain_through(self):
        assert risk_sensitive_reward(7.5, 0.0, 0.0, 0.0, 0.0) == pytest.approx(7.5)

    def test_custom_weights(self):
        weights = BriefWeights(
            business_gain=2.0,
            expected_shortfall=1.0,
            policy_violation=1.0,
            human_minutes=0.0,
            latency=0.0,
        )
        # 2*3 - 1*1 - 1*1 - 0 - 0 = 4
        assert risk_sensitive_reward(3.0, 1.0, 1.0, 60.0, 9.0, weights) == pytest.approx(4.0)

    def test_policy_violations_dominate(self):
        clean = risk_sensitive_reward(1.0, 0.0, 0.0, 0.0, 0.0)
        violating = risk_sensitive_reward(1.0, 0.0, 1.0, 0.0, 0.0)
        assert clean - violating == pytest.approx(5.0)


class TestExpectedShortfall:
    def test_alpha_95_of_20_samples_is_the_max(self):
        samples = np.arange(1.0, 21.0)  # 1..20
        assert expected_shortfall(samples, alpha=0.95) == pytest.approx(20.0)

    def test_alpha_90_of_20_samples_is_mean_of_top_two(self):
        samples = np.arange(1.0, 21.0)
        assert expected_shortfall(samples, alpha=0.90) == pytest.approx((19.0 + 20.0) / 2.0)

    def test_alpha_50_of_20_samples_is_mean_of_top_ten(self):
        samples = np.arange(1.0, 21.0)
        assert expected_shortfall(samples, alpha=0.50) == pytest.approx(np.mean(np.arange(11.0, 21.0)))

    def test_alpha_zero_is_the_mean(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0])
        assert expected_shortfall(samples, alpha=0.0) == pytest.approx(2.5)

    def test_order_invariant_and_accepts_lists(self):
        shuffled = [3.0, 20.0, 1.0, 7.0]
        assert expected_shortfall(shuffled, alpha=0.75) == pytest.approx(20.0)

    def test_single_sample(self):
        assert expected_shortfall([4.2], alpha=0.99) == pytest.approx(4.2)

    def test_es_never_below_var_level_mean(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(size=1000)
        assert expected_shortfall(samples, alpha=0.95) >= float(np.quantile(samples, 0.95)) - 1e-9

    def test_validation(self):
        with pytest.raises(ValueError, match="at least one"):
            expected_shortfall([])
        with pytest.raises(ValueError, match="alpha"):
            expected_shortfall([1.0], alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            expected_shortfall([1.0], alpha=-0.1)
