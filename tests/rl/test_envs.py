"""Determinism, boundedness and contract tests for the decision envs."""

from __future__ import annotations

import numpy as np
import pytest

from fintwinos.rl.envs import (
    ENV_NAMES,
    AlertTriageEnv,
    LiquidityActionEnv,
    QueueRoutingEnv,
    make_env,
)


def _rollout(env, seed: int, actions: list[int]):
    """Run a fixed action sequence; returns (observations, rewards, dones)."""
    observations = [env.reset(seed=seed)]
    rewards: list[float] = []
    dones: list[bool] = []
    for action in actions:
        obs, reward, done, _ = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        dones.append(done)
        if done:
            break
    return observations, rewards, dones


@pytest.mark.parametrize("name", ENV_NAMES)
class TestAllEnvs:
    def test_reset_returns_declared_state_dim(self, name):
        env = make_env(name, seed=5)
        state = env.reset(seed=5)
        assert state.shape == (env.state_dim,)
        assert np.all(np.isfinite(state))

    def test_deterministic_given_seed_and_actions(self, name):
        env_a = make_env(name, seed=1)
        env_b = make_env(name, seed=999)  # constructor seed overridden by reset
        actions = [i % env_a.n_actions for i in range(env_a.max_steps)]
        obs_a, rew_a, done_a = _rollout(env_a, seed=123, actions=actions)
        obs_b, rew_b, done_b = _rollout(env_b, seed=123, actions=actions)
        assert rew_a == rew_b
        assert done_a == done_b
        for oa, ob in zip(obs_a, obs_b, strict=True):
            assert np.array_equal(oa, ob)

    def test_different_seeds_diverge(self, name):
        env = make_env(name, seed=1)
        actions = [i % env.n_actions for i in range(env.max_steps)]
        obs_a, _, _ = _rollout(env, seed=1, actions=actions)
        obs_b, _, _ = _rollout(env, seed=2, actions=actions)
        # State trajectories must differ under different seeds (rewards may
        # coincide, e.g. deterministic funding costs without a breach).
        assert any(
            not np.array_equal(oa, ob) for oa, ob in zip(obs_a, obs_b, strict=True)
        )

    def test_rewards_bounded_and_states_finite(self, name):
        env = make_env(name, seed=42)
        lo, hi = env.reward_range
        rng = np.random.default_rng(0)
        for episode_seed in (7, 8, 9):
            state = env.reset(seed=episode_seed)
            done = False
            while not done:
                action = int(rng.integers(0, env.n_actions))
                state, reward, done, info = env.step(action)
                assert lo <= reward <= hi
                assert np.all(np.isfinite(state))
                assert "raw_reward" in info and "action_name" in info

    def test_episode_terminates_at_max_steps(self, name):
        env = make_env(name, seed=3)
        env.reset(seed=3)
        steps = 0
        done = False
        while not done:
            _, _, done, _ = env.step(0)
            steps += 1
            assert steps <= env.max_steps
        assert steps == env.max_steps

    def test_invalid_action_rejected(self, name):
        env = make_env(name, seed=3)
        env.reset(seed=3)
        with pytest.raises(ValueError, match="out of range"):
            env.step(env.n_actions)
        with pytest.raises(ValueError, match="out of range"):
            env.step(-1)

    def test_step_after_done_raises(self, name):
        env = make_env(name, seed=3)
        env.reset(seed=3)
        done = False
        while not done:
            _, _, done, _ = env.step(0)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)

    def test_step_before_reset_raises(self, name):
        env = make_env(name, seed=3)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)


class TestFactory:
    def test_unknown_env_name(self):
        with pytest.raises(KeyError, match="unknown env"):
            make_env("nope")

    def test_param_overrides_flow_through(self):
        env = make_env("alert_triage", seed=1, catch_gain=99.0)
        assert env._param("catch_gain", 10.0) == 99.0

    def test_factories_survive_missing_twin_sim(self):
        # twin_sim may or may not exist in this build; either way the factory works.
        for name in ENV_NAMES:
            env = make_env(name, seed=1)
            assert env.reset(seed=1).shape == (env.state_dim,)


class TestAlertTriageSemantics:
    def test_dismissing_true_positives_breaches_recall_floor(self):
        env = AlertTriageEnv(seed=0)
        env.reset(seed=0)
        breached = False
        for _ in range(env.max_steps):
            _, _, done, info = env.step(0)  # always dismiss
            if info["recall_floor_breached"]:
                breached = True
                assert info["recall"] < 0.7
            if done:
                break
        assert breached, "dismissing everything must eventually breach the recall floor"

    def test_escalation_catches_and_pays_minutes(self):
        env = AlertTriageEnv(seed=4)
        env.reset(seed=4)
        caught_any = False
        for _ in range(env.max_steps):
            _, _, done, info = env.step(2)  # always escalate
            assert info["minutes"] == 90.0
            caught_any = caught_any or info["caught"]
            if done:
                break
        assert caught_any


class TestLiquiditySemantics:
    def test_rewards_never_positive(self):
        env = LiquidityActionEnv(seed=11)
        env.reset(seed=11)
        done = False
        rng = np.random.default_rng(1)
        while not done:
            _, reward, done, _ = env.step(int(rng.integers(0, env.n_actions)))
            assert reward <= 0.0

    def test_doing_nothing_eventually_breaches(self):
        env = LiquidityActionEnv(seed=2, initial_cash=25.0)
        env.reset(seed=2)
        breached = False
        done = False
        while not done:
            _, _, done, info = env.step(0)
            breached = breached or info["breach"]
        assert breached

    def test_funding_actions_cost_money(self):
        env = LiquidityActionEnv(seed=5)
        env.reset(seed=5)
        _, _, _, info = env.step(3)  # term funding
        assert info["funding_cost"] == pytest.approx(20.0 * 0.04)
        assert info["raised"] == pytest.approx(20.0)


class TestQueueSemantics:
    def test_understaffing_breaches_more_than_full_staffing(self):
        def total_breaches(action: int) -> float:
            env = QueueRoutingEnv(seed=9)
            env.reset(seed=9)
            total, done = 0.0, False
            while not done:
                _, _, done, info = env.step(action)
                total += info["breaches"]
            return total

        assert total_breaches(0) > total_breaches(3)

    def test_staffing_cost_charged(self):
        env = QueueRoutingEnv(seed=9, initial_queue=0.0, initial_rate=4.0)
        env.reset(seed=9)
        _, reward, _, info = env.step(3)
        assert info["staff"] == 8
        assert reward <= -0.5 * 8 + 1e-9  # at least the staffing cost
