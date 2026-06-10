"""Contextual bandits in pure numpy: LinUCB and Gaussian Thompson sampling.

Bandits cover the FinTwinOS decisions that are genuinely one-shot — which
analyst queue gets an alert, which remediation template to propose — where a
full MDP formulation is overkill but logged-feedback learning still pays.

Both learners share the same interface:

- ``select(context, rng=None) -> arm`` — choose an arm for a context vector.
- ``update(arm, context, reward)`` — incorporate observed feedback.
- per-arm posterior inspection (``theta`` / ``posterior``) for audit.

Both maintain per-arm ridge-regression sufficient statistics
``A = lambda * I + sum(x x^T)`` and ``b = sum(r x)``; LinUCB acts on the upper
confidence bound, Thompson samples from the Gaussian posterior. The inverse
``A^{-1}`` is maintained incrementally via the Sherman–Morrison identity so a
``select`` never pays a matrix inversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["LinUCB", "ThompsonGaussian"]


def _validate_context(context: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(context, dtype=np.float64).ravel()
    if x.shape != (dim,):
        raise ValueError(f"context must have shape ({dim},), got {x.shape}")
    return x


class _RidgeArms:
    """Shared per-arm ridge statistics with Sherman–Morrison inverse updates."""

    def __init__(self, n_arms: int, dim: int, ridge: float):
        if n_arms < 2:
            raise ValueError(f"need at least 2 arms, got {n_arms}")
        if dim < 1:
            raise ValueError(f"context dim must be >= 1, got {dim}")
        if ridge <= 0:
            raise ValueError(f"ridge must be positive, got {ridge}")
        self.n_arms = int(n_arms)
        self.dim = int(dim)
        self.ridge = float(ridge)
        self._a = [self.ridge * np.eye(self.dim) for _ in range(self.n_arms)]
        self._a_inv = [np.eye(self.dim) / self.ridge for _ in range(self.n_arms)]
        self._b = [np.zeros(self.dim) for _ in range(self.n_arms)]
        self.pulls = np.zeros(self.n_arms, dtype=np.int64)

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        a_inv = self._a_inv[arm]
        self._a[arm] += np.outer(x, x)
        ax = a_inv @ x
        self._a_inv[arm] = a_inv - np.outer(ax, ax) / (1.0 + float(x @ ax))
        self._b[arm] += float(reward) * x
        self.pulls[arm] += 1

    def theta(self, arm: int) -> np.ndarray:
        return self._a_inv[arm] @ self._b[arm]

    def check_arm(self, arm: int) -> int:
        arm = int(arm)
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"arm {arm} out of range [0, {self.n_arms})")
        return arm


class LinUCB:
    """The LinUCB contextual bandit (Li et al., 2010), disagreement-free numpy.

    For each arm ``a`` with ridge statistics ``(A_a, b_a)`` and estimate
    ``theta_a = A_a^{-1} b_a``, the score for context ``x`` is

    ``p_a(x) = theta_a . x + alpha * sqrt(x^T A_a^{-1} x)``

    and ``select`` plays the argmax (ties broken by lowest arm index, or
    uniformly via ``rng`` when one is supplied).

    Args:
        n_arms: Number of arms.
        dim: Context dimensionality.
        alpha: Exploration width multiplier on the confidence term.
        ridge: L2 regularisation of the per-arm design matrix.
    """

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, ridge: float = 1.0):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.alpha = float(alpha)
        self._arms = _RidgeArms(n_arms, dim, ridge)

    @property
    def n_arms(self) -> int:
        """Number of arms."""
        return self._arms.n_arms

    @property
    def dim(self) -> int:
        """Context dimensionality."""
        return self._arms.dim

    def scores(self, context: np.ndarray) -> np.ndarray:
        """Upper confidence bound score per arm for one context."""
        x = _validate_context(context, self.dim)
        out = np.empty(self.n_arms)
        for arm in range(self.n_arms):
            a_inv = self._arms._a_inv[arm]
            mean = float(self._arms.theta(arm) @ x)
            width = float(np.sqrt(max(x @ a_inv @ x, 0.0)))
            out[arm] = mean + self.alpha * width
        return out

    def select(self, context: np.ndarray, rng: np.random.Generator | None = None) -> int:
        """Pick the arm with the highest UCB score for ``context``.

        Args:
            context: Feature vector of shape ``(dim,)``.
            rng: Optional generator used only to break exact ties uniformly;
                without it ties resolve to the lowest arm index.
        """
        scores = self.scores(context)
        best = np.flatnonzero(scores == scores.max())
        if rng is not None and best.size > 1:
            return int(best[rng.integers(0, best.size)])
        return int(best[0])

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Incorporate the observed ``reward`` for ``arm`` under ``context``."""
        arm = self._arms.check_arm(arm)
        x = _validate_context(context, self.dim)
        self._arms.update(arm, x, reward)

    def theta(self, arm: int) -> np.ndarray:
        """Current ridge estimate of arm ``arm``'s reward coefficients."""
        return self._arms.theta(self._arms.check_arm(arm))

    def arm_state(self, arm: int) -> dict[str, Any]:
        """Audit snapshot of one arm: theta, design matrix, pulls."""
        arm = self._arms.check_arm(arm)
        return {
            "theta": self._arms.theta(arm).tolist(),
            "pulls": int(self._arms.pulls[arm]),
            "a_diagonal": np.diag(self._arms._a[arm]).tolist(),
        }


class ThompsonGaussian:
    """Gaussian Thompson sampling over per-arm Bayesian linear regression.

    Each arm carries the conjugate posterior of a linear-Gaussian reward model
    ``r = theta_a . x + eps``, ``eps ~ N(0, noise_variance)`` with prior
    ``theta_a ~ N(0, prior_variance * I)``. ``select`` draws one ``theta_a``
    per arm from its posterior and plays the argmax of ``theta_a . x`` —
    probability matching, which explores exactly in proportion to posterior
    uncertainty.

    Posterior (with ``A = I / prior_variance + X^T X / noise_variance`` and
    ``z = X^T r / noise_variance``): ``theta | data ~ N(A^{-1} z, A^{-1})``.

    Args:
        n_arms: Number of arms.
        dim: Context dimensionality.
        noise_variance: Known observation noise variance.
        prior_variance: Prior variance of each coefficient.
        seed: Seed of the internal generator used when ``select`` is called
            without an explicit ``rng``.
    """

    def __init__(
        self,
        n_arms: int,
        dim: int,
        noise_variance: float = 0.25,
        prior_variance: float = 1.0,
        seed: int = 0,
    ):
        if noise_variance <= 0 or prior_variance <= 0:
            raise ValueError("noise_variance and prior_variance must be positive")
        self.noise_variance = float(noise_variance)
        self.prior_variance = float(prior_variance)
        # Reuse the ridge machinery: A = I/prior_var + X'X/noise_var is the ridge
        # design with lambda = noise_var/prior_var, scaled by 1/noise_var.
        self._arms = _RidgeArms(n_arms, dim, ridge=self.noise_variance / self.prior_variance)
        self._rng = np.random.default_rng(seed)

    @property
    def n_arms(self) -> int:
        """Number of arms."""
        return self._arms.n_arms

    @property
    def dim(self) -> int:
        """Context dimensionality."""
        return self._arms.dim

    def posterior(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        """Posterior ``(mean, covariance)`` of arm ``arm``'s coefficients."""
        arm = self._arms.check_arm(arm)
        mean = self._arms.theta(arm)
        covariance = self.noise_variance * self._arms._a_inv[arm]
        return mean, covariance

    def sample_thetas(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """One posterior draw of the coefficient vector for every arm."""
        rng = rng or self._rng
        draws = np.empty((self.n_arms, self.dim))
        for arm in range(self.n_arms):
            mean, covariance = self.posterior(arm)
            jitter = 1e-10 * np.eye(self.dim)
            chol = np.linalg.cholesky(covariance + jitter)
            draws[arm] = mean + chol @ rng.standard_normal(self.dim)
        return draws

    def select(self, context: np.ndarray, rng: np.random.Generator | None = None) -> int:
        """Thompson step: sample each arm's theta, play the argmax payoff."""
        x = _validate_context(context, self.dim)
        thetas = self.sample_thetas(rng)
        return int(np.argmax(thetas @ x))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Incorporate the observed ``reward`` for ``arm`` under ``context``."""
        arm = self._arms.check_arm(arm)
        x = _validate_context(context, self.dim)
        # _RidgeArms stores A*noise_var implicitly via its ridge scaling; feed the
        # raw sufficient statistics so theta() returns the posterior mean exactly.
        self._arms.update(arm, x, reward)

    def arm_state(self, arm: int) -> dict[str, Any]:
        """Audit snapshot of one arm: posterior mean/variance diagonal, pulls."""
        arm = self._arms.check_arm(arm)
        mean, covariance = self.posterior(arm)
        return {
            "posterior_mean": mean.tolist(),
            "posterior_variance_diagonal": np.diag(covariance).tolist(),
            "pulls": int(self._arms.pulls[arm]),
        }
