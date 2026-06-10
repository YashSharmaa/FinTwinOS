"""End-to-end pipeline test: buffer -> CQL -> OPE -> shadow -> gate, fast."""

from __future__ import annotations

import time

import pytest

from fintwinos.rl.ope import OPEResult
from fintwinos.rl.shadow import ShadowReport
from fintwinos.rl.train_offline_policy import behaviour_policy_for, main


class TestEndToEnd:
    def test_alert_triage_pipeline_runs_fast_and_complete(self):
        started = time.perf_counter()
        result = main(env_name="alert_triage", epochs=10, n_episodes=25, seed=7, quiet=True)
        elapsed = time.perf_counter() - started
        assert elapsed < 30.0

        assert result["env"] == "alert_triage"
        assert result["n_transitions"] == 25 * 64  # episodes x max_steps
        assert result["verdict"] in {"deploy", "shadow_only", "reject"}

        for method in ("ips", "snips", "doubly_robust"):
            ope = result["ope"][method]
            assert isinstance(ope, OPEResult)
            assert ope.lo_ci <= ope.estimate <= ope.hi_ci
            assert ope.ess > 0

        assert isinstance(result["shadow_buffer"], ShadowReport)
        assert isinstance(result["shadow_env"], ShadowReport)
        assert 0.0 <= result["shadow_buffer"].divergence_rate <= 1.0
        assert result["training"]["states_visited"] > 0
        assert result["q_table"]["n_actions"] == 3

    def test_pipeline_is_deterministic(self):
        a = main(env_name="alert_triage", epochs=5, n_episodes=10, seed=3, quiet=True)
        b = main(env_name="alert_triage", epochs=5, n_episodes=10, seed=3, quiet=True)
        assert a["verdict"] == b["verdict"]
        assert a["behaviour_value"] == pytest.approx(b["behaviour_value"])
        assert a["ope"]["doubly_robust"].estimate == pytest.approx(
            b["ope"]["doubly_robust"].estimate
        )
        assert a["shadow_env"].reward_delta == pytest.approx(b["shadow_env"].reward_delta)

    @pytest.mark.parametrize("env_name", ["liquidity_action", "queue_routing"])
    def test_other_envs_run_end_to_end(self, env_name):
        result = main(env_name=env_name, epochs=5, n_episodes=12, seed=1, quiet=True)
        assert result["verdict"] in {"deploy", "shadow_only", "reject"}
        assert result["n_transitions"] > 0

    def test_explicit_threshold_drives_the_gate(self):
        # An absurdly high threshold must always reject.
        result = main(
            env_name="alert_triage", epochs=5, n_episodes=10, seed=2,
            threshold=1e9, quiet=True,
        )
        assert result["verdict"] == "reject"

    def test_unknown_env_rejected(self):
        with pytest.raises(KeyError, match="unknown env"):
            main(env_name="nope", quiet=True)

    def test_rich_rendering_smoke(self, capsys):
        main(env_name="alert_triage", epochs=2, n_episodes=5, seed=1, quiet=False)
        captured = capsys.readouterr()
        assert "Offline policy evaluation" in captured.out
        assert "verdict" in captured.out


class TestBehaviourPolicy:
    def test_reports_exact_propensities(self):
        import numpy as np

        from fintwinos.rl.envs import make_env

        env = make_env("alert_triage", seed=0)
        policy = behaviour_policy_for(env)
        state = env.reset(seed=0)
        rng = np.random.default_rng(0)
        seen_probs = set()
        for _ in range(50):
            action, prob = policy(state, rng)
            assert 0 <= action < env.n_actions
            assert 0.0 < prob <= 1.0
            seen_probs.add(round(prob, 6))
        # Epsilon mixture: either the preferred mass or the uniform tail mass.
        assert seen_probs <= {round(0.7 + 0.3 / 3, 6), round(0.3 / 3, 6)}
