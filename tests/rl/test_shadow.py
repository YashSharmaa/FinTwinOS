"""Tests for the shadow runner and the lower-confidence-bound deployment gate."""

from __future__ import annotations

import numpy as np
import pytest

from fintwinos.core.audit import AuditTrail
from fintwinos.rl.envs import make_alert_triage_env
from fintwinos.rl.episodes import ReplayBuffer, Transition
from fintwinos.rl.ope import OPEResult
from fintwinos.rl.shadow import ShadowRunner, deployment_gate


def _ope(estimate: float, lo: float, hi: float, ess: float = 500.0) -> OPEResult:
    return OPEResult(
        method="doubly_robust", estimate=estimate, lo_ci=lo, hi_ci=hi, ess=ess, n=1000,
        mean_weight=1.0,
    )


class TestDeploymentGate:
    def test_deploy_when_lower_bound_clears(self):
        assert deployment_gate(_ope(1.0, 0.8, 1.2), threshold=0.5) == "deploy"

    def test_deploy_at_exact_boundary(self):
        assert deployment_gate(_ope(1.0, 0.5, 1.2), threshold=0.5) == "deploy"

    def test_shadow_only_when_mean_clears_but_bound_does_not(self):
        assert deployment_gate(_ope(0.6, 0.4, 0.9), threshold=0.5) == "shadow_only"

    def test_reject_when_mean_fails(self):
        assert deployment_gate(_ope(0.4, 0.2, 0.6), threshold=0.5) == "reject"

    def test_low_ess_downgrades_deploy_to_shadow_only(self):
        result = _ope(1.0, 0.8, 1.2, ess=5.0)
        assert deployment_gate(result, threshold=0.5, min_ess=30.0) == "shadow_only"
        assert deployment_gate(result, threshold=0.5) == "deploy"

    def test_low_ess_does_not_rescue_a_reject(self):
        assert deployment_gate(_ope(0.1, 0.0, 0.2, ess=5.0), threshold=0.5, min_ess=30) == "reject"

    def test_audit_record_appended(self):
        audit = AuditTrail()
        deployment_gate(_ope(1.0, 0.8, 1.2), threshold=0.5, audit=audit)
        records = audit.records(action="rl.deployment_gate")
        assert len(records) == 1
        assert records[0].payload["verdict"] == "deploy"
        assert audit.verify()


def _buffer_from_policy(policy, n: int = 200, seed: int = 0) -> ReplayBuffer:
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(n_actions=2)
    for _ in range(n):
        state = rng.uniform(size=2)
        action = policy(state)
        buffer.add(
            Transition(
                state=state,
                action=action,
                reward=float(action),  # action 1 is strictly better
                next_state=state,
                done=True,
                behaviour_prob=0.9,
            )
        )
    return buffer


class TestShadowRunnerBuffer:
    def test_zero_divergence_when_candidate_matches_log(self):
        policy = lambda s: int(s[0] > 0.5)  # noqa: E731
        buffer = _buffer_from_policy(policy)
        report = ShadowRunner(policy, policy, seed=0).run_buffer(buffer)
        assert report.mode == "buffer"
        assert report.divergence_rate == 0.0
        assert report.n_units == len(buffer)

    def test_better_candidate_shows_positive_delta(self):
        logged = lambda s: 0  # noqa: E731 - always the bad action
        candidate = lambda s: 1  # noqa: E731 - always the good action
        buffer = _buffer_from_policy(logged)
        # Behaviour must have support on action 1 for the counterfactual to exist.
        rng = np.random.default_rng(1)
        for _ in range(100):
            state = rng.uniform(size=2)
            buffer.add(
                Transition(
                    state=state, action=1, reward=1.0, next_state=state, done=True,
                    behaviour_prob=0.1,
                )
            )
        report = ShadowRunner(logged, candidate, seed=0).run_buffer(buffer)
        assert report.divergence_rate > 0.5
        assert report.reward_delta > 0.0
        assert report.candidate_mean_reward > report.baseline_mean_reward
        assert set(report.action_counts) == {1}

    def test_empty_buffer_rejected(self):
        runner = ShadowRunner(lambda s: 0, lambda s: 0, seed=0)
        with pytest.raises(ValueError, match="empty"):
            runner.run_buffer(ReplayBuffer())

    def test_audit_record_appended(self):
        audit = AuditTrail()
        policy = lambda s: 0  # noqa: E731
        runner = ShadowRunner(policy, policy, seed=0, audit=audit)
        runner.run_buffer(_buffer_from_policy(policy, n=20))
        assert len(audit.records(action="rl.shadow_run")) == 1
        assert audit.verify()


class TestShadowRunnerEnv:
    def test_identical_policies_show_zero_delta(self):
        policy = lambda s: 1  # noqa: E731
        runner = ShadowRunner(policy, policy, seed=3)
        report = runner.run_env(make_alert_triage_env(seed=0), n_episodes=4)
        assert report.mode == "env"
        assert report.divergence_rate == 0.0
        assert report.reward_delta == pytest.approx(0.0)
        assert report.n_units == 4

    def test_paired_rollouts_are_deterministic(self):
        baseline = lambda s: 0  # noqa: E731
        candidate = lambda s: 1 if s[0] > 0.5 else 0  # noqa: E731
        reports = [
            ShadowRunner(baseline, candidate, seed=5).run_env(
                make_alert_triage_env(seed=0), n_episodes=3
            )
            for _ in range(2)
        ]
        assert reports[0].reward_delta == pytest.approx(reports[1].reward_delta)
        assert reports[0].divergence_rate == pytest.approx(reports[1].divergence_rate)
        assert reports[0].details["baseline_returns"] == reports[1].details["baseline_returns"]

    def test_divergence_counts_candidate_disagreements(self):
        baseline = lambda s: 0  # noqa: E731
        candidate = lambda s: 2  # noqa: E731
        report = ShadowRunner(baseline, candidate, seed=1).run_env(
            make_alert_triage_env(seed=0), n_episodes=2
        )
        assert report.divergence_rate == 1.0
        assert report.details["decision_points"] == report.action_counts[2]

    def test_report_serialisable(self):
        import json

        policy = lambda s: 0  # noqa: E731
        report = ShadowRunner(policy, policy, seed=0).run_env(
            make_alert_triage_env(seed=0), n_episodes=2
        )
        assert json.dumps(report.as_dict())

    def test_invalid_episode_count(self):
        runner = ShadowRunner(lambda s: 0, lambda s: 0)
        with pytest.raises(ValueError, match="n_episodes"):
            runner.run_env(make_alert_triage_env(seed=0), n_episodes=0)
