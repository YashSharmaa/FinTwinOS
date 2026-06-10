"""Off-policy evaluation: IPS, SNIPS and doubly-robust, with bootstrap CIs.

Before any learned policy is allowed near the shadow runner — let alone a
deployment gate — its value must be estimated *from the logged data alone*.
This module implements the three standard estimators over a
:class:`~fintwinos.rl.episodes.ReplayBuffer` whose transitions carry behaviour
propensities:

- :func:`ips` — inverse propensity scoring, unbiased but high variance.
- :func:`snips` — self-normalised IPS, biased ``O(1/n)`` but far lower
  variance; the workhorse.
- :func:`doubly_robust` — combines a fitted reward model with importance
  weighting; unbiased if *either* the propensities or the model is right.

Every estimator returns an :class:`OPEResult` carrying a bootstrap percentile
confidence interval — the lower bound is what
:func:`fintwinos.rl.shadow.deployment_gate` acts on — plus the
:func:`effective_sample_size` diagnostic, because a great point estimate built
on twelve effective samples is a trap, not a green light.

Transitions are treated as logged contextual-bandit rounds (immediate reward,
one decision per context), which matches how FinTwinOS logs triage / routing /
liquidity decisions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from fintwinos.rl.episodes import ReplayBuffer

__all__ = [
    "OPEResult",
    "doubly_robust",
    "effective_sample_size",
    "fit_reward_model",
    "ips",
    "snips",
]

#: A target policy: either a callable ``state -> action-probability vector``
#: or a pre-computed ``(n, n_actions)`` matrix aligned with the buffer.
TargetPolicy = Callable[[np.ndarray], np.ndarray] | np.ndarray


@dataclass(frozen=True, slots=True)
class OPEResult:
    """One off-policy estimate with its uncertainty and diagnostics.

    Iterating the result yields ``(estimate, lo_ci, hi_ci)`` so call sites can
    unpack it as a 3-tuple, per the module contract.
    """

    method: str
    estimate: float
    lo_ci: float
    hi_ci: float
    ess: float
    n: int
    mean_weight: float

    def __iter__(self) -> Iterator[float]:
        yield self.estimate
        yield self.lo_ci
        yield self.hi_ci

    def as_dict(self) -> dict[str, float | str | int]:
        """JSON-serialisable representation."""
        return {
            "method": self.method,
            "estimate": self.estimate,
            "lo_ci": self.lo_ci,
            "hi_ci": self.hi_ci,
            "ess": self.ess,
            "n": self.n,
            "mean_weight": self.mean_weight,
        }


def effective_sample_size(weights: np.ndarray | list[float]) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` of importance weights.

    Equals ``n`` for uniform weights and collapses towards 1 as a few samples
    dominate — the canonical "is this estimate built on anything?" diagnostic.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        raise ValueError("effective_sample_size requires at least one weight")
    denominator = float((w**2).sum())
    if denominator == 0.0:
        return 0.0
    return float(w.sum() ** 2 / denominator)


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _target_probability_matrix(buffer: ReplayBuffer, target_policy: TargetPolicy) -> np.ndarray:
    """Materialise target probabilities as an ``(n, n_actions)`` matrix."""
    n = len(buffer)
    if n == 0:
        raise ValueError("cannot evaluate over an empty ReplayBuffer")
    n_actions = buffer.n_actions
    if isinstance(target_policy, np.ndarray):
        matrix = np.asarray(target_policy, dtype=np.float64)
        if matrix.shape != (n, n_actions):
            raise ValueError(
                f"target probability matrix must have shape ({n}, {n_actions}), got {matrix.shape}"
            )
    else:
        matrix = np.stack(
            [np.asarray(target_policy(t.state), dtype=np.float64) for t in buffer]
        )
        if matrix.shape != (n, n_actions):
            raise ValueError(
                f"target_policy must return {n_actions} probabilities per state, "
                f"got shape {matrix.shape}"
            )
    if np.any(matrix < -1e-12):
        raise ValueError("target probabilities must be non-negative")
    return np.clip(matrix, 0.0, 1.0)


def _importance_weights(
    buffer: ReplayBuffer,
    target_matrix: np.ndarray,
    behaviour_probs: np.ndarray | None,
    clip_weight: float | None,
) -> np.ndarray:
    actions = buffer.actions()
    behaviour = (
        np.asarray(behaviour_probs, dtype=np.float64).ravel()
        if behaviour_probs is not None
        else buffer.behaviour_probs()
    )
    if behaviour.shape[0] != len(buffer):
        raise ValueError(
            f"behaviour_probs must have length {len(buffer)}, got {behaviour.shape[0]}"
        )
    if np.any(behaviour <= 0.0):
        raise ValueError("behaviour probabilities must be strictly positive")
    weights = target_matrix[np.arange(len(buffer)), actions] / behaviour
    if clip_weight is not None:
        if clip_weight <= 0:
            raise ValueError(f"clip_weight must be positive, got {clip_weight}")
        weights = np.minimum(weights, clip_weight)
    return weights


def _bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic`` over resampled index sets."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        replicates[b] = statistic(indices)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(replicates, [tail, 1.0 - tail])
    return float(lo), float(hi)


def _result(
    method: str,
    estimate: float,
    statistic: Callable[[np.ndarray], float],
    weights: np.ndarray,
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> OPEResult:
    lo, hi = _bootstrap_ci(statistic, weights.size, n_bootstrap, seed, confidence)
    return OPEResult(
        method=method,
        estimate=float(estimate),
        lo_ci=lo,
        hi_ci=hi,
        ess=effective_sample_size(weights),
        n=int(weights.size),
        mean_weight=float(weights.mean()),
    )


# -----------------------------------------------------------------------------
# Estimators
# -----------------------------------------------------------------------------


def ips(
    buffer: ReplayBuffer,
    target_policy: TargetPolicy,
    *,
    behaviour_probs: np.ndarray | None = None,
    clip_weight: float | None = None,
    n_bootstrap: int = 400,
    seed: int = 0,
    confidence: float = 0.95,
) -> OPEResult:
    """Inverse propensity scoring estimate of the target policy's value.

    ``V_IPS = (1/n) * sum_i w_i * r_i`` with ``w_i = pi(a_i|s_i) / b_i``.

    Args:
        buffer: Logged transitions with behaviour propensities.
        target_policy: Callable ``state -> probs`` or an ``(n, n_actions)``
            probability matrix aligned with the buffer.
        behaviour_probs: Optional override of the logged propensities.
        clip_weight: Optional cap on importance weights (variance control at
            the cost of a pessimistic bias).
        n_bootstrap: Bootstrap replicates for the confidence interval.
        seed: Bootstrap rng seed (full determinism).
        confidence: Two-sided CI coverage (default 95%).

    Returns:
        An :class:`OPEResult`; unpacks as ``(estimate, lo_ci, hi_ci)``.
    """
    target_matrix = _target_probability_matrix(buffer, target_policy)
    weights = _importance_weights(buffer, target_matrix, behaviour_probs, clip_weight)
    rewards = buffer.rewards()
    contributions = weights * rewards

    def statistic(indices: np.ndarray) -> float:
        return float(contributions[indices].mean())

    return _result(
        "ips", contributions.mean(), statistic, weights, n_bootstrap, seed, confidence
    )


def snips(
    buffer: ReplayBuffer,
    target_policy: TargetPolicy,
    *,
    behaviour_probs: np.ndarray | None = None,
    clip_weight: float | None = None,
    n_bootstrap: int = 400,
    seed: int = 0,
    confidence: float = 0.95,
) -> OPEResult:
    """Self-normalised IPS: ``V_SNIPS = sum(w r) / sum(w)``.

    Trades the ``O(1/n)`` normalisation bias for a large variance reduction;
    invariant to constant scaling of the propensities. Arguments mirror
    :func:`ips`.
    """
    target_matrix = _target_probability_matrix(buffer, target_policy)
    weights = _importance_weights(buffer, target_matrix, behaviour_probs, clip_weight)
    rewards = buffer.rewards()
    total_weight = float(weights.sum())
    estimate = float((weights * rewards).sum() / total_weight) if total_weight > 0 else 0.0

    def statistic(indices: np.ndarray) -> float:
        w = weights[indices]
        denominator = float(w.sum())
        if denominator == 0.0:
            return 0.0
        return float((w * rewards[indices]).sum() / denominator)

    return _result("snips", estimate, statistic, weights, n_bootstrap, seed, confidence)


def fit_reward_model(buffer: ReplayBuffer, ridge: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a per-action ridge regression ``q_hat(s, a)`` on the logged data.

    Each action's rewards are regressed on ``[1, s]``; actions never observed
    fall back to the global mean reward. Used as the default direct model
    inside :func:`doubly_robust`.

    Returns:
        Callable ``state -> (n_actions,) predicted-reward vector``.
    """
    if len(buffer) == 0:
        raise ValueError("cannot fit a reward model on an empty ReplayBuffer")
    states = buffer.states()
    actions = buffer.actions()
    rewards = buffer.rewards()
    n_actions = buffer.n_actions
    global_mean = float(rewards.mean())
    dim = states.shape[1] + 1

    thetas: list[np.ndarray | None] = []
    for action in range(n_actions):
        mask = actions == action
        if not mask.any():
            thetas.append(None)
            continue
        design = np.hstack([np.ones((int(mask.sum()), 1)), states[mask]])
        gram = design.T @ design + ridge * np.eye(dim)
        thetas.append(np.linalg.solve(gram, design.T @ rewards[mask]))

    def q_hat(state: np.ndarray) -> np.ndarray:
        features = np.concatenate([[1.0], np.asarray(state, dtype=np.float64).ravel()])
        out = np.full(n_actions, global_mean)
        for action, theta in enumerate(thetas):
            if theta is not None:
                out[action] = float(features @ theta)
        return out

    return q_hat


def doubly_robust(
    buffer: ReplayBuffer,
    target_policy: TargetPolicy,
    *,
    q_model: Callable[[np.ndarray], np.ndarray] | None = None,
    behaviour_probs: np.ndarray | None = None,
    clip_weight: float | None = None,
    n_bootstrap: int = 400,
    seed: int = 0,
    confidence: float = 0.95,
) -> OPEResult:
    """Doubly-robust estimate combining a direct model with importance weights.

    ``V_DR = (1/n) sum_i [ sum_a pi(a|s_i) q_hat(s_i, a)
    + w_i (r_i - q_hat(s_i, a_i)) ]``

    Unbiased whenever the propensities *or* the reward model are correct, and
    typically lower variance than IPS because the model absorbs the bulk of
    the signal while the correction term re-centres its bias.

    Args:
        buffer: Logged transitions with behaviour propensities.
        target_policy: Callable or ``(n, n_actions)`` probability matrix.
        q_model: Optional reward model ``state -> (n_actions,)``; when omitted
            a per-action ridge regression is fitted on the buffer via
            :func:`fit_reward_model`.
        behaviour_probs / clip_weight / n_bootstrap / seed / confidence:
            As in :func:`ips`.

    Returns:
        An :class:`OPEResult`; unpacks as ``(estimate, lo_ci, hi_ci)``.
    """
    target_matrix = _target_probability_matrix(buffer, target_policy)
    weights = _importance_weights(buffer, target_matrix, behaviour_probs, clip_weight)
    rewards = buffer.rewards()
    actions = buffer.actions()
    model = q_model or fit_reward_model(buffer)

    q_matrix = np.stack([np.asarray(model(t.state), dtype=np.float64) for t in buffer])
    if q_matrix.shape != target_matrix.shape:
        raise ValueError(
            f"q_model must return {target_matrix.shape[1]} values per state, "
            f"got shape {q_matrix.shape}"
        )
    baseline = (target_matrix * q_matrix).sum(axis=1)
    correction = weights * (rewards - q_matrix[np.arange(len(buffer)), actions])
    contributions = baseline + correction

    def statistic(indices: np.ndarray) -> float:
        return float(contributions[indices].mean())

    return _result(
        "doubly_robust", contributions.mean(), statistic, weights, n_bootstrap, seed, confidence
    )
