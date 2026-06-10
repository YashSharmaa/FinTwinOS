"""Off-policy estimator accuracy tests on a synthetic contextual bandit."""

from __future__ import annotations

import numpy as np
import pytest

from fintwinos.rl.episodes import ReplayBuffer, Transition
from fintwinos.rl.ope import (
    OPEResult,
    doubly_robust,
    effective_sample_size,
    fit_reward_model,
    ips,
    snips,
)

N_ACTIONS = 2
TARGET_P1 = 0.95  # target plays action 1 with prob 0.95
MEAN_REWARD = {0: 0.2, 1: 1.0}
TRUE_VALUE = TARGET_P1 * MEAN_REWARD[1] + (1 - TARGET_P1) * MEAN_REWARD[0]  # = 0.96


def _bandit_buffer(n: int = 4000, seed: int = 0) -> ReplayBuffer:
    """Logged bandit: uniform behaviour, reward = mean(action) + ctx effect + noise.

    The context effect ``0.5 * (x - 0.5)`` is zero-mean over the uniform
    context distribution, so the true target value stays ``TRUE_VALUE`` while
    giving the doubly-robust reward model real signal to fit.
    """
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(n_actions=N_ACTIONS)
    for _ in range(n):
        x = float(rng.uniform())
        action = int(rng.integers(0, N_ACTIONS))
        reward = MEAN_REWARD[action] + 0.5 * (x - 0.5) + float(rng.normal(0.0, 0.1))
        buffer.add(
            Transition(
                state=np.array([x]),
                action=action,
                reward=reward,
                next_state=np.array([x]),
                done=True,
                behaviour_prob=1.0 / N_ACTIONS,
            )
        )
    return buffer


def _target_policy(state: np.ndarray) -> np.ndarray:
    return np.array([1.0 - TARGET_P1, TARGET_P1])


@pytest.fixture(scope="module")
def buffer() -> ReplayBuffer:
    return _bandit_buffer()


class TestEstimatorAccuracy:
    @pytest.mark.parametrize("estimator", [ips, snips, doubly_robust])
    def test_recovers_true_value(self, buffer, estimator):
        result = estimator(buffer, _target_policy, seed=1)
        assert result.estimate == pytest.approx(TRUE_VALUE, abs=0.05)
        assert result.lo_ci <= TRUE_VALUE <= result.hi_ci
        assert result.lo_ci < result.estimate < result.hi_ci
        assert 0 < result.ess <= len(buffer)
        assert result.n == len(buffer)

    def test_result_unpacks_as_three_tuple(self, buffer):
        estimate, lo, hi = ips(buffer, _target_policy, seed=1)
        assert lo < estimate < hi

    def test_matrix_target_policy_matches_callable(self, buffer):
        matrix = np.tile(np.array([1.0 - TARGET_P1, TARGET_P1]), (len(buffer), 1))
        from_callable = snips(buffer, _target_policy, seed=3)
        from_matrix = snips(buffer, matrix, seed=3)
        assert from_matrix.estimate == pytest.approx(from_callable.estimate)
        assert from_matrix.lo_ci == pytest.approx(from_callable.lo_ci)

    def test_deterministic_given_seed(self, buffer):
        a = doubly_robust(buffer, _target_policy, seed=9)
        b = doubly_robust(buffer, _target_policy, seed=9)
        assert (a.estimate, a.lo_ci, a.hi_ci) == (b.estimate, b.lo_ci, b.hi_ci)

    def test_dr_with_perfect_model_is_tight(self, buffer):
        def true_q(state: np.ndarray) -> np.ndarray:
            x = float(state[0])
            return np.array([MEAN_REWARD[0] + 0.5 * (x - 0.5), MEAN_REWARD[1] + 0.5 * (x - 0.5)])

        exact = doubly_robust(buffer, _target_policy, q_model=true_q, seed=2)
        assert exact.estimate == pytest.approx(TRUE_VALUE, abs=0.02)
        # The DR interval with the true model is tighter than plain IPS's.
        plain = ips(buffer, _target_policy, seed=2)
        assert (exact.hi_ci - exact.lo_ci) < (plain.hi_ci - plain.lo_ci)

    def test_weight_clipping_caps_weights(self, buffer):
        clipped = ips(buffer, _target_policy, clip_weight=1.0, seed=4)
        assert clipped.mean_weight <= 1.0
        # Clipping a policy that upweights the good action biases downwards.
        assert clipped.estimate <= ips(buffer, _target_policy, seed=4).estimate


class TestRewardModel:
    def test_fits_per_action_means(self, buffer):
        model = fit_reward_model(buffer)
        predictions = model(np.array([0.5]))
        assert predictions[0] == pytest.approx(MEAN_REWARD[0], abs=0.05)
        assert predictions[1] == pytest.approx(MEAN_REWARD[1], abs=0.05)

    def test_unobserved_action_falls_back_to_global_mean(self):
        buffer = ReplayBuffer(n_actions=3)
        for i in range(50):
            buffer.add(
                Transition(
                    state=np.array([float(i)]),
                    action=i % 2,  # action 2 never logged
                    reward=1.0,
                    next_state=np.array([0.0]),
                    done=True,
                )
            )
        model = fit_reward_model(buffer)
        assert model(np.array([1.0]))[2] == pytest.approx(1.0)


class TestEffectiveSampleSize:
    def test_uniform_weights_give_n(self):
        assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)

    def test_concentrated_weights_collapse_to_one(self):
        weights = np.zeros(100)
        weights[0] = 5.0
        assert effective_sample_size(weights) == pytest.approx(1.0)

    def test_hand_computed_value(self):
        # w = [1, 3]: (1+3)^2 / (1+9) = 1.6
        assert effective_sample_size([1.0, 3.0]) == pytest.approx(1.6)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            effective_sample_size([])


class TestValidation:
    def test_empty_buffer_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            ips(ReplayBuffer(), _target_policy)

    def test_bad_matrix_shape_rejected(self, buffer):
        with pytest.raises(ValueError, match="shape"):
            ips(buffer, np.ones((3, 2)))

    def test_negative_probabilities_rejected(self, buffer):
        with pytest.raises(ValueError, match="non-negative"):
            ips(buffer, lambda s: np.array([-0.5, 1.5]))

    def test_behaviour_override_length_checked(self, buffer):
        with pytest.raises(ValueError, match="length"):
            ips(buffer, _target_policy, behaviour_probs=np.array([0.5]))

    def test_result_as_dict(self, buffer):
        result = snips(buffer, _target_policy, seed=1)
        assert isinstance(result, OPEResult)
        payload = result.as_dict()
        assert payload["method"] == "snips"
        assert set(payload) >= {"estimate", "lo_ci", "hi_ci", "ess", "n"}
