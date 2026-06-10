"""Shadow execution and the deployment gate for candidate policies.

The founding brief's deployment doctrine: a learned policy never goes live on
the strength of a point estimate. It must (1) run in *shadow* — scored against
the incumbent on identical inputs without acting — and (2) clear a gate that
judges the **lower confidence bound** of its off-policy estimate, not the
mean. :func:`deployment_gate` mirrors the brief's ``mark_for_shadow_only``
semantics:

- ``deploy`` — even the pessimistic bound beats the threshold;
- ``shadow_only`` — the mean clears but the bound does not (promising,
  keep shadowing until the interval tightens);
- ``reject`` — the mean itself fails.

:class:`ShadowRunner` produces the evidence: divergence statistics and
counterfactual reward deltas, either over a live bounded env (paired episodes
under common random numbers) or over a logged
:class:`~fintwinos.rl.episodes.ReplayBuffer` (importance-weighted
counterfactuals via SNIPS). Every shadow run can append to the shared
:class:`~fintwinos.core.audit.AuditTrail` so the promotion decision is
reconstructible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fintwinos.core.audit import AuditTrail
from fintwinos.rl.episodes import ReplayBuffer
from fintwinos.rl.ope import OPEResult, snips

__all__ = ["ShadowReport", "ShadowRunner", "deployment_gate"]

#: A deterministic policy under test: ``state -> action``.
Policy = Callable[[np.ndarray], int]


@dataclass(slots=True)
class ShadowReport:
    """Outcome of one shadow comparison between baseline and candidate.

    Attributes:
        mode: ``"env"`` (paired live rollouts) or ``"buffer"`` (logged data).
        n_units: Episodes compared (env mode) or transitions scored (buffer).
        divergence_rate: Fraction of decision points where the candidate's
            action differed from the baseline/logged action.
        baseline_mean_reward: Mean reward of the baseline (per episode in env
            mode, per step in buffer mode).
        candidate_mean_reward: Mean (env) or counterfactual SNIPS-estimated
            (buffer) reward of the candidate, same unit as the baseline.
        reward_delta: ``candidate_mean_reward - baseline_mean_reward``.
        action_counts: Candidate action histogram, action index -> count.
        details: Estimator diagnostics and per-episode breakdowns.
    """

    mode: str
    n_units: int
    divergence_rate: float
    baseline_mean_reward: float
    candidate_mean_reward: float
    reward_delta: float
    action_counts: dict[int, int] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (used for audit payloads)."""
        return {
            "mode": self.mode,
            "n_units": self.n_units,
            "divergence_rate": self.divergence_rate,
            "baseline_mean_reward": self.baseline_mean_reward,
            "candidate_mean_reward": self.candidate_mean_reward,
            "reward_delta": self.reward_delta,
            "action_counts": {str(k): v for k, v in self.action_counts.items()},
            "details": self.details,
        }


class ShadowRunner:
    """Runs a candidate policy alongside a baseline without letting it act.

    Args:
        baseline_policy: The incumbent ``state -> action`` policy.
        candidate_policy: The challenger ``state -> action`` policy.
        seed: Master seed for paired-episode generation (full determinism).
        epsilon: Smoothing used to turn the deterministic candidate into an
            epsilon-soft distribution for importance weighting in buffer mode
            (keeps every weight finite).
        audit: Optional shared audit trail; each run appends one
            ``rl.shadow_run`` record.
    """

    def __init__(
        self,
        baseline_policy: Policy,
        candidate_policy: Policy,
        seed: int = 0,
        epsilon: float = 0.05,
        audit: AuditTrail | None = None,
    ):
        if not 0.0 < epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        self.baseline_policy = baseline_policy
        self.candidate_policy = candidate_policy
        self.seed = int(seed)
        self.epsilon = float(epsilon)
        self.audit = audit

    # -- env mode ------------------------------------------------------------------

    def run_env(self, env: Any, n_episodes: int = 8) -> ShadowReport:
        """Paired rollouts: same per-episode seeds for baseline and candidate.

        Because the bounded envs consume a fixed number of random draws per
        step (common random numbers), seeding both rollouts identically makes
        the comparison a paired experiment — the reward delta reflects the
        policies, not the noise.

        Divergence is measured on the baseline's visited states: at every step
        the candidate is asked what it *would* have done.
        """
        if n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
        master = np.random.default_rng(self.seed)
        baseline_returns: list[float] = []
        candidate_returns: list[float] = []
        diverged = 0
        decision_points = 0
        action_counts: dict[int, int] = {}

        for _ in range(n_episodes):
            episode_seed = int(master.integers(0, 2**31 - 1))

            # Baseline rollout, recording visited states for divergence scoring.
            state = env.reset(seed=episode_seed)
            total = 0.0
            done = False
            while not done:
                action = int(self.baseline_policy(state))
                shadow_action = int(self.candidate_policy(state))
                decision_points += 1
                if shadow_action != action:
                    diverged += 1
                action_counts[shadow_action] = action_counts.get(shadow_action, 0) + 1
                state, reward, done, _ = env.step(action)
                total += reward
            baseline_returns.append(total)

            # Candidate rollout on the identical seed.
            state = env.reset(seed=episode_seed)
            total = 0.0
            done = False
            while not done:
                state, reward, done, _ = env.step(int(self.candidate_policy(state)))
                total += reward
            candidate_returns.append(total)

        baseline_mean = float(np.mean(baseline_returns))
        candidate_mean = float(np.mean(candidate_returns))
        report = ShadowReport(
            mode="env",
            n_units=n_episodes,
            divergence_rate=diverged / decision_points if decision_points else 0.0,
            baseline_mean_reward=baseline_mean,
            candidate_mean_reward=candidate_mean,
            reward_delta=candidate_mean - baseline_mean,
            action_counts=action_counts,
            details={
                "baseline_returns": [float(r) for r in baseline_returns],
                "candidate_returns": [float(r) for r in candidate_returns],
                "decision_points": decision_points,
                "env": getattr(env, "name", "unknown"),
            },
        )
        self._audit(report)
        return report

    # -- buffer mode ------------------------------------------------------------------

    def run_buffer(self, buffer: ReplayBuffer) -> ShadowReport:
        """Score the candidate counterfactually over logged transitions.

        Divergence compares the candidate's action against the *logged*
        action; the counterfactual value of the candidate is the SNIPS
        estimate of its epsilon-smoothed distribution, compared against the
        logged mean per-step reward.
        """
        if len(buffer) == 0:
            raise ValueError("cannot shadow-run over an empty ReplayBuffer")
        n_actions = buffer.n_actions
        diverged = 0
        action_counts: dict[int, int] = {}
        for transition in buffer:
            action = int(self.candidate_policy(transition.state))
            action_counts[action] = action_counts.get(action, 0) + 1
            if action != transition.action:
                diverged += 1

        epsilon = self.epsilon

        def target_probs(state: np.ndarray) -> np.ndarray:
            probs = np.full(n_actions, epsilon / n_actions)
            probs[int(self.candidate_policy(state))] += 1.0 - epsilon
            return probs

        estimate = snips(buffer, target_probs, seed=self.seed)
        baseline_mean = buffer.mean_reward()
        report = ShadowReport(
            mode="buffer",
            n_units=len(buffer),
            divergence_rate=diverged / len(buffer),
            baseline_mean_reward=baseline_mean,
            candidate_mean_reward=estimate.estimate,
            reward_delta=estimate.estimate - baseline_mean,
            action_counts=action_counts,
            details={"snips": estimate.as_dict(), "epsilon": epsilon},
        )
        self._audit(report)
        return report

    def _audit(self, report: ShadowReport) -> None:
        if self.audit is not None:
            self.audit.append("rl.shadow_runner", "rl.shadow_run", report.as_dict())


def deployment_gate(
    ope_result: OPEResult,
    threshold: float,
    *,
    min_ess: float | None = None,
    audit: AuditTrail | None = None,
) -> str:
    """Gate a candidate policy on the *lower confidence bound* of its value.

    Decision rule (mirroring the founding brief's ``mark_for_shadow_only``):

    - ``"deploy"`` — ``lo_ci >= threshold``: even the pessimistic read clears.
    - ``"shadow_only"`` — ``estimate >= threshold`` but ``lo_ci < threshold``:
      promising yet unproven; keep shadowing until the interval tightens.
    - ``"reject"`` — ``estimate < threshold``: not even the mean clears.

    A ``"deploy"`` additionally downgrades to ``"shadow_only"`` when the
    effective sample size behind the estimate is below ``min_ess`` — a tight
    interval built on a handful of effective samples is an artefact, not
    evidence.

    Args:
        ope_result: The off-policy estimate (typically doubly-robust).
        threshold: Minimum acceptable policy value — conventionally the
            logged behaviour policy's value, so deployment requires beating
            the incumbent with confidence.
        min_ess: Optional floor on the effective sample size.
        audit: Optional shared audit trail; the verdict is appended as
            ``rl.deployment_gate``.

    Returns:
        ``"deploy"``, ``"shadow_only"`` or ``"reject"``.
    """
    if ope_result.estimate < threshold:
        verdict = "reject"
    elif ope_result.lo_ci >= threshold:
        verdict = "deploy"
        if min_ess is not None and ope_result.ess < min_ess:
            verdict = "shadow_only"
    else:
        verdict = "shadow_only"

    if audit is not None:
        audit.append(
            "rl.deployment_gate",
            "rl.deployment_gate",
            {
                "verdict": verdict,
                "threshold": threshold,
                "min_ess": min_ess,
                "ope": ope_result.as_dict(),
            },
        )
    return verdict
