"""Conservative offline Q-learning over discretised states (pure numpy).

Offline RL on logged financial decisions has one dominant failure mode:
ordinary Q-learning *over-values actions the log never tried*, because the
bootstrapped max happily extrapolates into unsupported regions. The
:class:`ConservativeQLearner` here applies the CQL-style regulariser

``alpha * (logsumexp_a Q(s, a) - Q(s, a_logged))``

whose gradient pushes *down* the value of every action in proportion to its
softmax weight while pushing *up* the logged action — so actions without data
support end up pessimistically valued and a greedy policy stays inside the
behaviour distribution unless the data genuinely supports leaving it.

States are continuous feature vectors; :class:`StateDiscretiser` maps them to
tabular codes via per-dimension quantile bins, which keeps the learner exact,
auditable (the full Q-table is exportable) and dependency-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from fintwinos.rl.episodes import ReplayBuffer

__all__ = ["ConservativeQLearner", "StateDiscretiser"]


class StateDiscretiser:
    """Per-dimension quantile binning of continuous states into integer codes.

    For each feature dimension, ``bins_per_dim`` quantile edges are fitted on
    the training states (duplicate edges collapse, so constant dimensions cost
    a single bin). A state's per-dimension bin indices are packed into one
    integer code via mixed-radix encoding, giving a stable tabular state id.
    """

    def __init__(self, bins_per_dim: int = 6):
        if bins_per_dim < 1:
            raise ValueError(f"bins_per_dim must be >= 1, got {bins_per_dim}")
        self.bins_per_dim = int(bins_per_dim)
        self._edges: list[np.ndarray] | None = None
        self._radices: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._edges is not None

    @property
    def n_states(self) -> int:
        """Total number of distinct codes (product of per-dim bin counts)."""
        if self._radices is None:
            raise RuntimeError("StateDiscretiser must be fitted first")
        return int(np.prod(self._radices))

    def fit(self, states: np.ndarray) -> StateDiscretiser:
        """Fit quantile edges on an ``(n, d)`` matrix of states.

        Returns ``self`` for chaining.
        """
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 2 or states.shape[0] == 0:
            raise ValueError(f"states must be a non-empty (n, d) matrix, got shape {states.shape}")
        interior = np.linspace(0.0, 1.0, self.bins_per_dim + 1)[1:-1]
        edges: list[np.ndarray] = []
        for dim in range(states.shape[1]):
            column = states[:, dim]
            if interior.size == 0 or column.min() == column.max():
                # Constant dimensions carry no information: one bin, no edges.
                dim_edges = np.empty(0)
            else:
                dim_edges = np.unique(np.quantile(column, interior))
            edges.append(dim_edges)
        self._edges = edges
        self._radices = np.asarray([e.size + 1 for e in edges], dtype=np.int64)
        return self

    def transform(self, state: np.ndarray) -> int:
        """Encode one state vector into its integer code."""
        return int(self.transform_batch(np.asarray(state, dtype=np.float64).reshape(1, -1))[0])

    def transform_batch(self, states: np.ndarray) -> np.ndarray:
        """Vectorised encoding of an ``(n, d)`` matrix into ``n`` codes."""
        if self._edges is None or self._radices is None:
            raise RuntimeError("StateDiscretiser must be fitted first")
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != len(self._edges):
            raise ValueError(
                f"expected (n, {len(self._edges)}) states, got shape {states.shape}"
            )
        codes = np.zeros(states.shape[0], dtype=np.int64)
        multiplier = 1
        for dim, edges in enumerate(self._edges):
            idx = np.searchsorted(edges, states[:, dim], side="right")
            codes += idx * multiplier
            multiplier *= int(self._radices[dim])
        return codes

    def export(self) -> dict[str, Any]:
        """JSON-serialisable description of the fitted bins."""
        if self._edges is None or self._radices is None:
            raise RuntimeError("StateDiscretiser must be fitted first")
        return {
            "bins_per_dim": self.bins_per_dim,
            "edges": [e.tolist() for e in self._edges],
            "radices": self._radices.tolist(),
            "n_states": self.n_states,
        }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _logsumexp(values: np.ndarray) -> float:
    peak = float(values.max())
    return peak + math.log(float(np.exp(values - peak).sum()))


class ConservativeQLearner:
    """Tabular conservative Q-learning on discretised states.

    Args:
        n_actions: Size of the discrete action space.
        alpha: Weight of the conservative (CQL) penalty. ``0`` recovers plain
            tabular Q-learning; larger values increase pessimism on actions
            the log never took.
        gamma: Discount factor.
        learning_rate: TD step size.
        bins_per_dim: Quantile bins per state dimension when the learner fits
            its own :class:`StateDiscretiser`.
        discretiser: Optional pre-built (fitted or unfitted) discretiser.
        seed: Seed for the epoch-shuffling rng (full determinism).
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 1.0,
        gamma: float = 0.95,
        learning_rate: float = 0.2,
        bins_per_dim: int = 6,
        discretiser: StateDiscretiser | None = None,
        seed: int = 0,
    ):
        if n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.discretiser = discretiser or StateDiscretiser(bins_per_dim=bins_per_dim)
        self._rng = np.random.default_rng(seed)
        self._q: dict[int, np.ndarray] = {}
        self._visits: dict[int, np.ndarray] = {}
        self.history: list[dict[str, float]] = []

    # -- training ------------------------------------------------------------------

    def fit(self, buffer: ReplayBuffer, epochs: int = 50) -> dict[str, Any]:
        """Run conservative Q iteration over the logged transitions.

        Each epoch sweeps every transition once in a shuffled order, applying
        the TD update followed by the conservative penalty gradient

        ``Q(s, .) -= lr * alpha * (softmax(Q(s, .)) - onehot(a_logged))``

        which is exactly the gradient of
        ``alpha * (logsumexp_a Q(s, a) - Q(s, a_logged))``.

        Args:
            buffer: Logged transitions (must be non-empty).
            epochs: Number of full sweeps.

        Returns:
            Diagnostics: per-epoch mean absolute TD error and mean
            conservative gap, plus dataset/coverage statistics.
        """
        if len(buffer) == 0:
            raise ValueError("cannot fit on an empty ReplayBuffer")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")

        states = buffer.states()
        if not self.discretiser.fitted:
            self.discretiser.fit(states)
        codes = self.discretiser.transform_batch(states)
        next_codes = self.discretiser.transform_batch(buffer.next_states())
        actions = buffer.actions()
        rewards = buffer.rewards()
        dones = buffer.dones()
        n = len(buffer)

        self.history = []
        for _ in range(epochs):
            order = self._rng.permutation(n)
            td_total = 0.0
            gap_total = 0.0
            for i in order:
                code = int(codes[i])
                action = int(actions[i])
                q_s = self._q.setdefault(code, np.zeros(self.n_actions))
                visits = self._visits.setdefault(code, np.zeros(self.n_actions))
                visits[action] += 1.0

                if dones[i]:
                    target = float(rewards[i])
                else:
                    q_next = self._q.setdefault(int(next_codes[i]), np.zeros(self.n_actions))
                    target = float(rewards[i]) + self.gamma * float(q_next.max())
                td_error = target - q_s[action]
                q_s[action] += self.learning_rate * td_error

                if self.alpha > 0.0:
                    grad = _softmax(q_s)
                    grad[action] -= 1.0
                    q_s -= self.learning_rate * self.alpha * grad
                    gap_total += _logsumexp(q_s) - float(q_s[action])
                td_total += abs(td_error)
            self.history.append(
                {
                    "mean_abs_td_error": td_total / n,
                    "mean_conservative_gap": gap_total / n if self.alpha > 0 else 0.0,
                }
            )

        return {
            "epochs": epochs,
            "transitions": n,
            "states_visited": len(self._q),
            "final_mean_abs_td_error": self.history[-1]["mean_abs_td_error"],
            "final_mean_conservative_gap": self.history[-1]["mean_conservative_gap"],
        }

    # -- inference -----------------------------------------------------------------

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Action values for one continuous state (zeros if never visited)."""
        code = self.discretiser.transform(state)
        values = self._q.get(code)
        return values.copy() if values is not None else np.zeros(self.n_actions)

    def greedy_policy(self) -> Callable[[np.ndarray], int]:
        """A deterministic policy ``state -> argmax_a Q(s, a)``."""

        def policy(state: np.ndarray) -> int:
            return int(np.argmax(self.q_values(state)))

        return policy

    def policy_probabilities(self, state: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
        """Epsilon-soft greedy action distribution (for off-policy evaluation).

        The greedy action receives ``1 - epsilon + epsilon / n``; every other
        action receives ``epsilon / n``, keeping all importance weights finite.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        probs = np.full(self.n_actions, epsilon / self.n_actions)
        probs[int(np.argmax(self.q_values(state)))] += 1.0 - epsilon
        return probs

    # -- export --------------------------------------------------------------------

    def q_table(self) -> dict[int, np.ndarray]:
        """Copy of the learned table: state code -> action-value vector."""
        return {code: values.copy() for code, values in self._q.items()}

    def visit_counts(self) -> dict[int, np.ndarray]:
        """Copy of per-(state, action) visit counts accumulated during fit."""
        return {code: counts.copy() for code, counts in self._visits.items()}

    def export_q_table(self) -> dict[str, Any]:
        """Fully JSON-serialisable snapshot for audit and review.

        Includes the discretiser bins so the table is interpretable without
        the live learner object.
        """
        return {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "discretiser": self.discretiser.export() if self.discretiser.fitted else None,
            "q": {str(code): values.tolist() for code, values in sorted(self._q.items())},
            "visits": {
                str(code): counts.tolist() for code, counts in sorted(self._visits.items())
            },
        }
