"""Transitions, episodes and the replay buffer used by every RL component.

The RL layer of FinTwinOS is strictly *offline*: it learns from logged twin
episodes, never by acting against production systems. This module defines the
data shapes everything else consumes:

- :class:`Transition` — one ``(state, action, reward, next_state, done)`` step,
  annotated with the behaviour policy's propensity so off-policy estimators
  (:mod:`fintwinos.rl.ope`) can importance-weight it.
- :class:`Episode` — an ordered list of transitions with an id and metadata.
- :class:`ReplayBuffer` — a flat, episode-aware store with deterministic
  sampling and JSONL round-tripping for audit-friendly persistence.
- :func:`build_episodes_from_replay` — the bridge from the twin's
  ``ReplayEngine`` (streams of ``EventEnvelope``) into RL episodes.

All randomness flows through ``numpy.random.Generator`` instances passed in by
the caller, so every sample is reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Episode",
    "ReplayBuffer",
    "Transition",
    "build_episodes_from_replay",
    "collect_episodes",
]

#: Payload keys that carry RL bookkeeping rather than state features.
_RESERVED_PAYLOAD_KEYS = frozenset({"action", "behaviour_prob", "reward", "done"})


@dataclass(slots=True)
class Transition:
    """One logged decision step.

    Attributes:
        state: Feature vector observed before acting (1-D ``float64`` array).
        action: Index of the discrete action that was taken.
        reward: Scalar reward realised for that action.
        next_state: Feature vector observed after acting.
        done: Whether this transition terminated its episode.
        behaviour_prob: Probability the *logging* (behaviour) policy assigned
            to ``action`` in ``state``; required for importance weighting.
        info: Free-form diagnostics (event ids, raw payload fields, ...).
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool = False
    behaviour_prob: float = 1.0
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=np.float64).ravel()
        self.next_state = np.asarray(self.next_state, dtype=np.float64).ravel()
        self.action = int(self.action)
        self.reward = float(self.reward)
        self.done = bool(self.done)
        self.behaviour_prob = float(self.behaviour_prob)
        if not 0.0 < self.behaviour_prob <= 1.0:
            raise ValueError(
                f"behaviour_prob must be in (0, 1], got {self.behaviour_prob}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (used by JSONL persistence)."""
        return {
            "state": self.state.tolist(),
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.tolist(),
            "done": self.done,
            "behaviour_prob": self.behaviour_prob,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Transition:
        """Inverse of :meth:`to_dict`."""
        return cls(
            state=np.asarray(payload["state"], dtype=np.float64),
            action=int(payload["action"]),
            reward=float(payload["reward"]),
            next_state=np.asarray(payload["next_state"], dtype=np.float64),
            done=bool(payload.get("done", False)),
            behaviour_prob=float(payload.get("behaviour_prob", 1.0)),
            info=dict(payload.get("info", {})),
        )


@dataclass(slots=True)
class Episode:
    """An ordered sequence of transitions sharing one episode id."""

    episode_id: str
    transitions: list[Transition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self.transitions)

    @property
    def total_reward(self) -> float:
        """Undiscounted sum of rewards across the episode."""
        return float(sum(t.reward for t in self.transitions))

    def discounted_return(self, gamma: float = 1.0) -> float:
        """Discounted return ``sum_t gamma^t r_t`` from the first step."""
        return float(
            sum(t.reward * gamma**i for i, t in enumerate(self.transitions))
        )


class ReplayBuffer:
    """Flat, episode-aware container of logged transitions.

    The buffer keeps a flat list (what learners and estimators iterate over)
    plus an episode index (so trajectory structure survives JSONL round-trips).

    Args:
        n_actions: Optional explicit size of the discrete action space. When
            omitted it is inferred as ``max(action) + 1`` over the contents.
    """

    def __init__(self, n_actions: int | None = None):
        self._transitions: list[Transition] = []
        self._episode_index: dict[str, list[int]] = {}
        self._explicit_n_actions = n_actions

    # -- write -----------------------------------------------------------------

    def add(self, transition: Transition, episode_id: str = "default") -> None:
        """Append one transition, attributing it to ``episode_id``."""
        self._episode_index.setdefault(episode_id, []).append(len(self._transitions))
        self._transitions.append(transition)

    def add_episode(self, episode: Episode) -> None:
        """Append every transition of ``episode`` under its own id."""
        for transition in episode.transitions:
            self.add(transition, episode_id=episode.episode_id)

    @classmethod
    def from_episodes(cls, episodes: Iterable[Episode], n_actions: int | None = None) -> ReplayBuffer:
        """Build a buffer from a collection of episodes."""
        buffer = cls(n_actions=n_actions)
        for episode in episodes:
            buffer.add_episode(episode)
        return buffer

    # -- read ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._transitions)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._transitions)

    def __getitem__(self, index: int) -> Transition:
        return self._transitions[index]

    @property
    def transitions(self) -> list[Transition]:
        """The flat transition list (a shallow copy; mutate via :meth:`add`)."""
        return list(self._transitions)

    @property
    def episode_ids(self) -> list[str]:
        """Episode ids in first-seen order."""
        return list(self._episode_index)

    def episode(self, episode_id: str) -> Episode:
        """Reassemble one episode from the flat store."""
        if episode_id not in self._episode_index:
            raise KeyError(f"unknown episode '{episode_id}'")
        return Episode(
            episode_id=episode_id,
            transitions=[self._transitions[i] for i in self._episode_index[episode_id]],
        )

    @property
    def n_actions(self) -> int:
        """Size of the discrete action space (explicit, else inferred)."""
        if self._explicit_n_actions is not None:
            return self._explicit_n_actions
        if not self._transitions:
            return 0
        return max(t.action for t in self._transitions) + 1

    # -- vectorised views --------------------------------------------------------

    def states(self) -> np.ndarray:
        """All states stacked into an ``(n, d)`` matrix."""
        return np.stack([t.state for t in self._transitions]) if self._transitions else np.empty((0, 0))

    def next_states(self) -> np.ndarray:
        """All next-states stacked into an ``(n, d)`` matrix."""
        return np.stack([t.next_state for t in self._transitions]) if self._transitions else np.empty((0, 0))

    def actions(self) -> np.ndarray:
        """All actions as an int vector."""
        return np.asarray([t.action for t in self._transitions], dtype=np.int64)

    def rewards(self) -> np.ndarray:
        """All rewards as a float vector."""
        return np.asarray([t.reward for t in self._transitions], dtype=np.float64)

    def dones(self) -> np.ndarray:
        """All terminal flags as a boolean vector."""
        return np.asarray([t.done for t in self._transitions], dtype=bool)

    def behaviour_probs(self) -> np.ndarray:
        """All behaviour-policy propensities as a float vector."""
        return np.asarray([t.behaviour_prob for t in self._transitions], dtype=np.float64)

    def mean_reward(self) -> float:
        """Mean logged per-step reward (the on-policy behaviour value)."""
        if not self._transitions:
            return 0.0
        return float(self.rewards().mean())

    # -- sampling ----------------------------------------------------------------

    def sample(self, batch: int, rng: np.random.Generator) -> list[Transition]:
        """Draw ``batch`` transitions deterministically given ``rng``.

        Samples without replacement when the buffer is large enough, with
        replacement otherwise. Raises ``ValueError`` on an empty buffer.
        """
        if not self._transitions:
            raise ValueError("cannot sample from an empty ReplayBuffer")
        if batch <= 0:
            raise ValueError(f"batch must be positive, got {batch}")
        replace = batch > len(self._transitions)
        indices = rng.choice(len(self._transitions), size=batch, replace=replace)
        return [self._transitions[int(i)] for i in indices]

    # -- persistence ----------------------------------------------------------------

    def to_jsonl(self, path: Path | str) -> int:
        """Write one JSON line per transition; returns the number written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        index_to_episode: dict[int, str] = {}
        for episode_id, indices in self._episode_index.items():
            for i in indices:
                index_to_episode[i] = episode_id
        with path.open("w", encoding="utf-8") as fh:
            for i, transition in enumerate(self._transitions):
                row = transition.to_dict()
                row["episode_id"] = index_to_episode.get(i, "default")
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        return len(self._transitions)

    @classmethod
    def from_jsonl(cls, path: Path | str, n_actions: int | None = None) -> ReplayBuffer:
        """Load a buffer previously written by :meth:`to_jsonl`."""
        buffer = cls(n_actions=n_actions)
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                episode_id = str(row.pop("episode_id", "default"))
                buffer.add(Transition.from_dict(row), episode_id=episode_id)
        return buffer


# -----------------------------------------------------------------------------
# Bridges from the twin into RL episodes
# -----------------------------------------------------------------------------


def _numeric_feature_keys(payloads: Iterable[dict[str, Any]]) -> list[str]:
    """Sorted union of numeric, non-reserved payload keys across envelopes."""
    keys: set[str] = set()
    for payload in payloads:
        for key, value in payload.items():
            if key in _RESERVED_PAYLOAD_KEYS:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                keys.add(key)
    return sorted(keys)


def _features(payload: dict[str, Any], keys: list[str]) -> np.ndarray:
    """Extract a feature vector for ``keys`` from one payload (missing -> 0)."""
    return np.asarray(
        [float(payload.get(key, 0.0) or 0.0) for key in keys], dtype=np.float64
    )


def build_episodes_from_replay(
    replay_engine: Any,
    reward_fn: Callable[[Any], float],
    *,
    feature_keys: list[str] | None = None,
) -> list[Episode]:
    """Bridge twin replay episodes into RL :class:`Episode` objects.

    Walks every episode recorded by the twin's ``ReplayEngine`` (anything
    satisfying :class:`fintwinos.core.interfaces.ReplayEngine`), turning each
    ``EventEnvelope`` into one :class:`Transition`:

    - **state** — the envelope payload's numeric fields, ordered by a stable
      sorted key list shared across the whole replay (so feature positions are
      consistent between episodes). Pass ``feature_keys`` to pin the schema.
    - **action** — ``payload["action"]`` (default 0 when absent).
    - **behaviour_prob** — ``payload["behaviour_prob"]`` (default 1.0).
    - **reward** — ``reward_fn(envelope)``, letting callers express business
      reward shaping (see :mod:`fintwinos.rl.rewards`) over raw twin events.
    - **next_state** — the following envelope's features; the final envelope
      transitions to itself and is marked ``done``.

    Args:
        replay_engine: Provider of ``episodes()`` and ``episode(episode_id)``.
        reward_fn: Maps one envelope to a scalar reward.
        feature_keys: Optional explicit feature schema (sorted order is *not*
            imposed on an explicit list — it is used verbatim).

    Returns:
        One :class:`Episode` per non-empty twin episode, in engine order.
    """
    episode_ids = list(replay_engine.episodes())
    envelopes_by_episode: dict[str, list[Any]] = {
        episode_id: list(replay_engine.episode(episode_id)) for episode_id in episode_ids
    }
    if feature_keys is None:
        all_payloads = [
            dict(envelope.payload)
            for envelopes in envelopes_by_episode.values()
            for envelope in envelopes
        ]
        feature_keys = _numeric_feature_keys(all_payloads)

    episodes: list[Episode] = []
    for episode_id in episode_ids:
        envelopes = envelopes_by_episode[episode_id]
        if not envelopes:
            continue
        transitions: list[Transition] = []
        for i, envelope in enumerate(envelopes):
            payload = dict(envelope.payload)
            is_last = i == len(envelopes) - 1
            next_payload = payload if is_last else dict(envelopes[i + 1].payload)
            transitions.append(
                Transition(
                    state=_features(payload, feature_keys),
                    action=int(payload.get("action", 0)),
                    reward=float(reward_fn(envelope)),
                    next_state=_features(next_payload, feature_keys),
                    done=is_last,
                    behaviour_prob=float(payload.get("behaviour_prob", 1.0)),
                    info={
                        "event_id": getattr(envelope, "event_id", None),
                        "kind": getattr(envelope, "kind", None),
                        "source": getattr(envelope, "source", None),
                    },
                )
            )
        episodes.append(
            Episode(
                episode_id=episode_id,
                transitions=transitions,
                metadata={"feature_keys": list(feature_keys), "n_events": len(envelopes)},
            )
        )
    return episodes


def collect_episodes(
    env: Any,
    policy: Callable[[np.ndarray, np.random.Generator], tuple[int, float]],
    n_episodes: int,
    seed: int = 0,
    episode_prefix: str = "ep",
) -> list[Episode]:
    """Roll a stochastic policy through a bounded env, logging propensities.

    Args:
        env: Anything with ``reset(seed) -> state`` and
            ``step(action) -> (state, reward, done, info)`` (see
            :mod:`fintwinos.rl.envs`).
        policy: Callable ``(state, rng) -> (action, behaviour_prob)`` — it must
            report the probability with which it picked the sampled action so
            the log supports off-policy evaluation later.
        n_episodes: Number of episodes to roll out.
        seed: Master seed; per-episode seeds and the policy rng derive from it.
        episode_prefix: Prefix for generated episode ids.

    Returns:
        Fully-populated :class:`Episode` objects, deterministic given ``seed``.
    """
    master = np.random.default_rng(seed)
    episodes: list[Episode] = []
    for index in range(n_episodes):
        episode_seed = int(master.integers(0, 2**31 - 1))
        policy_rng = np.random.default_rng(episode_seed + 1)
        state = env.reset(seed=episode_seed)
        transitions: list[Transition] = []
        done = False
        while not done:
            action, prob = policy(state, policy_rng)
            next_state, reward, done, info = env.step(action)
            transitions.append(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    behaviour_prob=prob,
                    info=dict(info),
                )
            )
            state = next_state
        episodes.append(
            Episode(
                episode_id=f"{episode_prefix}_{index:04d}",
                transitions=transitions,
                metadata={"seed": episode_seed, "env": getattr(env, "name", "unknown")},
            )
        )
    return episodes
