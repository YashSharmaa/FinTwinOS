"""Tests for the state discretiser and the conservative Q-learner."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fintwinos.rl.episodes import ReplayBuffer, Transition
from fintwinos.rl.offline import ConservativeQLearner, StateDiscretiser

S0 = np.array([0.0])
S1 = np.array([1.0])


def _toy_mdp_buffer() -> ReplayBuffer:
    """A 2-state MDP where the optimal policy is a0 in s0 and a1 in s1.

    Logged coverage:
      - (s0, a0) -> r=+1, s1      (good, frequently logged)
      - (s1, a1) -> r=+2, s0 done (good, frequently logged)
      - (s1, a0) -> r=0,  s0 done (bad alternative, logged for contrast)
      - (s0, a1) NEVER appears — the unsupported action.
    """
    buffer = ReplayBuffer(n_actions=2)
    for _ in range(30):
        buffer.add(Transition(state=S0, action=0, reward=1.0, next_state=S1, behaviour_prob=0.9))
        buffer.add(
            Transition(state=S1, action=1, reward=2.0, next_state=S0, done=True, behaviour_prob=0.8)
        )
    for _ in range(10):
        buffer.add(
            Transition(state=S1, action=0, reward=0.0, next_state=S0, done=True, behaviour_prob=0.2)
        )
    return buffer


class TestStateDiscretiser:
    def test_quantile_bins_separate_low_and_high(self):
        rng = np.random.default_rng(0)
        states = rng.uniform(0.0, 1.0, size=(500, 2))
        disc = StateDiscretiser(bins_per_dim=4).fit(states)
        assert disc.fitted
        low = disc.transform(np.array([0.01, 0.01]))
        high = disc.transform(np.array([0.99, 0.99]))
        assert low != high
        assert 0 <= low < disc.n_states
        assert 0 <= high < disc.n_states

    def test_constant_dimension_costs_one_bin(self):
        states = np.column_stack([np.full(100, 3.14), np.linspace(0, 1, 100)])
        disc = StateDiscretiser(bins_per_dim=5).fit(states)
        # dim 0 collapses to a single bin; total states = 1 * (<=5)
        assert disc.n_states <= 5

    def test_batch_matches_single(self):
        rng = np.random.default_rng(1)
        states = rng.normal(size=(200, 3))
        disc = StateDiscretiser(bins_per_dim=6).fit(states)
        batch = disc.transform_batch(states[:10])
        singles = [disc.transform(s) for s in states[:10]]
        assert batch.tolist() == singles

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="fitted"):
            StateDiscretiser().transform(np.array([0.0]))
        with pytest.raises(RuntimeError, match="fitted"):
            _ = StateDiscretiser().n_states

    def test_export_is_json_serialisable(self):
        disc = StateDiscretiser(bins_per_dim=3).fit(np.random.default_rng(2).normal(size=(50, 2)))
        payload = disc.export()
        assert json.dumps(payload)
        assert payload["n_states"] == disc.n_states

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="bins_per_dim"):
            StateDiscretiser(bins_per_dim=0)
        with pytest.raises(ValueError, match="matrix"):
            StateDiscretiser().fit(np.empty((0, 2)))


class TestConservativeQLearner:
    def test_recovers_optimal_actions_on_toy_mdp(self):
        learner = ConservativeQLearner(n_actions=2, alpha=0.5, gamma=0.9, seed=0)
        diagnostics = learner.fit(_toy_mdp_buffer(), epochs=60)
        policy = learner.greedy_policy()
        assert policy(S0) == 0
        assert policy(S1) == 1
        assert diagnostics["states_visited"] == 2
        # TD error should have shrunk over training.
        assert (
            learner.history[-1]["mean_abs_td_error"] < learner.history[0]["mean_abs_td_error"]
        )

    def test_stays_pessimistic_on_unsupported_action(self):
        learner = ConservativeQLearner(n_actions=2, alpha=0.5, gamma=0.9, seed=0)
        learner.fit(_toy_mdp_buffer(), epochs=60)
        q_s0 = learner.q_values(S0)
        # (s0, a1) never appears in the log: the CQL penalty must have pushed
        # it strictly below the supported action AND below zero (its init).
        assert q_s0[1] < q_s0[0]
        assert q_s0[1] < 0.0
        visits = learner.visit_counts()
        code_s0 = learner.discretiser.transform(S0)
        assert visits[code_s0][1] == 0.0

    def test_more_conservatism_means_bigger_gap(self):
        def gap(alpha: float) -> float:
            learner = ConservativeQLearner(n_actions=2, alpha=alpha, gamma=0.9, seed=0)
            learner.fit(_toy_mdp_buffer(), epochs=40)
            q = learner.q_values(S0)
            return float(q[0] - q[1])

        assert gap(2.0) > gap(0.1)

    def test_alpha_zero_recovers_plain_q_learning_values(self):
        learner = ConservativeQLearner(
            n_actions=2, alpha=0.0, gamma=0.9, learning_rate=0.5, seed=0
        )
        learner.fit(_toy_mdp_buffer(), epochs=200)
        # Bellman fixed point: Q(s1,a1)=2 (terminal), Q(s0,a0)=1+0.9*max(2,0)=2.8.
        assert learner.q_values(S1)[1] == pytest.approx(2.0, abs=0.05)
        assert learner.q_values(S0)[0] == pytest.approx(2.8, abs=0.1)
        # Unsupported action keeps its zero init without the CQL penalty.
        assert learner.q_values(S0)[1] == 0.0

    def test_deterministic_given_seed(self):
        runs = []
        for _ in range(2):
            learner = ConservativeQLearner(n_actions=2, alpha=0.5, seed=42)
            learner.fit(_toy_mdp_buffer(), epochs=20)
            runs.append((learner.q_values(S0), learner.q_values(S1)))
        assert np.array_equal(runs[0][0], runs[1][0])
        assert np.array_equal(runs[0][1], runs[1][1])

    def test_policy_probabilities_epsilon_soft(self):
        learner = ConservativeQLearner(n_actions=2, alpha=0.5, seed=0)
        learner.fit(_toy_mdp_buffer(), epochs=30)
        probs = learner.policy_probabilities(S0, epsilon=0.1)
        assert probs.sum() == pytest.approx(1.0)
        assert probs[0] == pytest.approx(0.95)
        assert probs[1] == pytest.approx(0.05)
        with pytest.raises(ValueError, match="epsilon"):
            learner.policy_probabilities(S0, epsilon=1.5)

    def test_unseen_state_defaults_to_zero_values(self):
        learner = ConservativeQLearner(n_actions=3, alpha=0.5, seed=0)
        learner.discretiser.fit(np.linspace(0.0, 1.0, 50).reshape(-1, 1))
        assert np.array_equal(learner.q_values(np.array([0.5])), np.zeros(3))

    def test_q_table_export_is_json_serialisable(self):
        learner = ConservativeQLearner(n_actions=2, alpha=0.5, seed=0)
        learner.fit(_toy_mdp_buffer(), epochs=10)
        exported = learner.export_q_table()
        assert json.dumps(exported)
        assert exported["n_actions"] == 2
        assert len(exported["q"]) == 2
        assert exported["discretiser"] is not None
        table = learner.q_table()
        assert all(v.shape == (2,) for v in table.values())

    def test_input_validation(self):
        with pytest.raises(ValueError, match="n_actions"):
            ConservativeQLearner(n_actions=0)
        learner = ConservativeQLearner(n_actions=2)
        with pytest.raises(ValueError, match="empty"):
            learner.fit(ReplayBuffer(), epochs=1)
        with pytest.raises(ValueError, match="epochs"):
            learner.fit(_toy_mdp_buffer(), epochs=0)
