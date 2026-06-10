"""Contextual bandit tests: regret versus random, posteriors, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from fintwinos.rl.bandits import LinUCB, ThompsonGaussian

DIM = 3
N_ARMS = 3
THETAS = np.array(
    [
        [1.0, 0.0, -0.5],
        [-0.5, 1.0, 0.5],
        [0.2, -0.8, 1.0],
    ]
)


def _simulate(select_fn, update_fn, horizon: int, seed: int) -> float:
    """Run a bandit loop on the synthetic linear problem; returns total regret."""
    rng = np.random.default_rng(seed)
    regret = 0.0
    for _ in range(horizon):
        x = rng.uniform(-1.0, 1.0, size=DIM)
        expected = THETAS @ x
        arm = select_fn(x, rng)
        reward = float(expected[arm] + rng.normal(0.0, 0.1))
        update_fn(arm, x, reward)
        regret += float(expected.max() - expected[arm])
    return regret


def _random_regret(horizon: int, seed: int) -> float:
    chosen: dict[str, int] = {}

    def select(x, rng):
        chosen["arm"] = int(rng.integers(0, N_ARMS))
        return chosen["arm"]

    return _simulate(select, lambda *_: None, horizon, seed)


class TestLinUCB:
    def test_beats_random_on_linear_problem(self):
        bandit = LinUCB(n_arms=N_ARMS, dim=DIM, alpha=0.8)
        regret = _simulate(
            lambda x, rng: bandit.select(x), bandit.update, horizon=600, seed=0
        )
        random_regret = _random_regret(600, seed=0)
        assert regret < 0.5 * random_regret

    def test_theta_converges_to_truth(self):
        bandit = LinUCB(n_arms=N_ARMS, dim=DIM, alpha=0.5)
        _simulate(lambda x, rng: bandit.select(x), bandit.update, horizon=900, seed=1)
        # Every arm gets pulled enough for its estimate to approach the truth.
        for arm in range(N_ARMS):
            if bandit.arm_state(arm)["pulls"] >= 50:
                assert np.linalg.norm(bandit.theta(arm) - THETAS[arm]) < 0.3

    def test_deterministic_without_rng(self):
        results = []
        for _ in range(2):
            bandit = LinUCB(n_arms=N_ARMS, dim=DIM, alpha=1.0)
            actions = []
            rng = np.random.default_rng(7)
            for _ in range(50):
                x = rng.uniform(-1, 1, size=DIM)
                arm = bandit.select(x)
                actions.append(arm)
                bandit.update(arm, x, float(THETAS[arm] @ x))
            results.append(actions)
        assert results[0] == results[1]

    def test_scores_shape_and_optimism(self):
        bandit = LinUCB(n_arms=N_ARMS, dim=DIM, alpha=2.0)
        x = np.array([0.5, -0.5, 0.2])
        scores = bandit.scores(x)
        assert scores.shape == (N_ARMS,)
        # With zero data, theta = 0 so the score is the pure exploration bonus.
        assert np.all(scores > 0)

    def test_tie_break_with_rng_stays_in_range(self):
        bandit = LinUCB(n_arms=N_ARMS, dim=DIM, alpha=1.0)
        x = np.zeros(DIM)  # all scores identical -> full tie
        rng = np.random.default_rng(0)
        arms = {bandit.select(x, rng) for _ in range(20)}
        assert arms <= set(range(N_ARMS))
        assert len(arms) > 1  # the rng actually breaks ties

    def test_validation(self):
        with pytest.raises(ValueError, match="arms"):
            LinUCB(n_arms=1, dim=2)
        with pytest.raises(ValueError, match="alpha"):
            LinUCB(n_arms=2, dim=2, alpha=-1.0)
        bandit = LinUCB(n_arms=2, dim=2)
        with pytest.raises(ValueError, match="context"):
            bandit.select(np.ones(3))
        with pytest.raises(ValueError, match="arm"):
            bandit.update(5, np.ones(2), 1.0)


class TestThompsonGaussian:
    def test_beats_random_on_linear_problem(self):
        bandit = ThompsonGaussian(n_arms=N_ARMS, dim=DIM, noise_variance=0.05, seed=0)
        regret = _simulate(bandit.select, bandit.update, horizon=600, seed=0)
        random_regret = _random_regret(600, seed=0)
        assert regret < 0.5 * random_regret

    def test_posterior_concentrates_with_data(self):
        bandit = ThompsonGaussian(n_arms=N_ARMS, dim=DIM, noise_variance=0.05, seed=2)
        _, prior_cov = bandit.posterior(0)
        rng = np.random.default_rng(3)
        for _ in range(300):
            x = rng.uniform(-1, 1, size=DIM)
            bandit.update(0, x, float(THETAS[0] @ x + rng.normal(0, 0.1)))
        mean, cov = bandit.posterior(0)
        assert np.trace(cov) < 0.05 * np.trace(prior_cov)
        assert np.linalg.norm(mean - THETAS[0]) < 0.3

    def test_prior_posterior_is_prior_variance(self):
        bandit = ThompsonGaussian(n_arms=2, dim=2, noise_variance=0.25, prior_variance=2.0)
        mean, cov = bandit.posterior(0)
        assert np.allclose(mean, 0.0)
        assert np.allclose(cov, 2.0 * np.eye(2))

    def test_deterministic_given_seed(self):
        actions = []
        for _ in range(2):
            bandit = ThompsonGaussian(n_arms=N_ARMS, dim=DIM, seed=11)
            rng = np.random.default_rng(5)
            run = []
            for _ in range(40):
                x = rng.uniform(-1, 1, size=DIM)
                arm = bandit.select(x, np.random.default_rng(99))
                run.append(arm)
                bandit.update(arm, x, float(THETAS[arm] @ x))
            actions.append(run)
        assert actions[0] == actions[1]

    def test_sample_thetas_shape(self):
        bandit = ThompsonGaussian(n_arms=N_ARMS, dim=DIM, seed=0)
        draws = bandit.sample_thetas(np.random.default_rng(1))
        assert draws.shape == (N_ARMS, DIM)

    def test_validation(self):
        with pytest.raises(ValueError, match="positive"):
            ThompsonGaussian(n_arms=2, dim=2, noise_variance=0.0)
        bandit = ThompsonGaussian(n_arms=2, dim=2)
        state = bandit.arm_state(1)
        assert state["pulls"] == 0
        assert len(state["posterior_mean"]) == 2
