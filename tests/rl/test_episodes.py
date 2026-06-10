"""Tests for transitions, episodes, the replay buffer and the twin bridge."""

from __future__ import annotations

import numpy as np
import pytest

from fintwinos.core.types import EventEnvelope
from fintwinos.rl.envs import make_alert_triage_env
from fintwinos.rl.episodes import (
    Episode,
    ReplayBuffer,
    Transition,
    build_episodes_from_replay,
    collect_episodes,
)


def _transition(action: int = 0, reward: float = 1.0, done: bool = False) -> Transition:
    return Transition(
        state=np.array([0.1, 0.2]),
        action=action,
        reward=reward,
        next_state=np.array([0.3, 0.4]),
        done=done,
        behaviour_prob=0.5,
        info={"k": "v"},
    )


class TestTransition:
    def test_coerces_types(self):
        t = Transition(state=[1, 2], action=np.int64(1), reward=np.float64(2.5), next_state=[3, 4])
        assert t.state.dtype == np.float64
        assert isinstance(t.action, int)
        assert isinstance(t.reward, float)
        assert t.behaviour_prob == 1.0

    def test_rejects_invalid_propensity(self):
        with pytest.raises(ValueError, match="behaviour_prob"):
            Transition(state=[0.0], action=0, reward=0.0, next_state=[0.0], behaviour_prob=0.0)
        with pytest.raises(ValueError, match="behaviour_prob"):
            Transition(state=[0.0], action=0, reward=0.0, next_state=[0.0], behaviour_prob=1.5)

    def test_dict_round_trip(self):
        t = _transition(action=2, reward=-1.5, done=True)
        clone = Transition.from_dict(t.to_dict())
        assert np.allclose(clone.state, t.state)
        assert np.allclose(clone.next_state, t.next_state)
        assert clone.action == t.action
        assert clone.reward == t.reward
        assert clone.done is True
        assert clone.behaviour_prob == t.behaviour_prob
        assert clone.info == t.info


class TestEpisode:
    def test_total_and_discounted_return(self):
        episode = Episode(
            episode_id="ep1",
            transitions=[_transition(reward=1.0), _transition(reward=2.0), _transition(reward=4.0)],
        )
        assert len(episode) == 3
        assert episode.total_reward == pytest.approx(7.0)
        assert episode.discounted_return(gamma=0.5) == pytest.approx(1.0 + 1.0 + 1.0)


class TestReplayBuffer:
    def test_add_and_vector_views(self):
        buffer = ReplayBuffer()
        for i in range(4):
            buffer.add(_transition(action=i % 2, reward=float(i)), episode_id=f"ep{i // 2}")
        assert len(buffer) == 4
        assert buffer.n_actions == 2
        assert buffer.states().shape == (4, 2)
        assert buffer.actions().tolist() == [0, 1, 0, 1]
        assert buffer.rewards().tolist() == [0.0, 1.0, 2.0, 3.0]
        assert buffer.behaviour_probs().tolist() == [0.5] * 4
        assert buffer.mean_reward() == pytest.approx(1.5)
        assert buffer.episode_ids == ["ep0", "ep1"]
        assert len(buffer.episode("ep1")) == 2

    def test_explicit_n_actions_wins(self):
        buffer = ReplayBuffer(n_actions=5)
        buffer.add(_transition(action=1))
        assert buffer.n_actions == 5

    def test_sample_is_deterministic_given_rng(self):
        buffer = ReplayBuffer()
        for i in range(20):
            buffer.add(_transition(reward=float(i)))
        first = [t.reward for t in buffer.sample(8, np.random.default_rng(42))]
        second = [t.reward for t in buffer.sample(8, np.random.default_rng(42))]
        assert first == second
        # Without replacement when the buffer is big enough.
        assert len(set(first)) == 8

    def test_sample_with_replacement_when_batch_exceeds_size(self):
        buffer = ReplayBuffer()
        buffer.add(_transition())
        batch = buffer.sample(5, np.random.default_rng(0))
        assert len(batch) == 5

    def test_sample_errors(self):
        with pytest.raises(ValueError, match="empty"):
            ReplayBuffer().sample(1, np.random.default_rng(0))
        buffer = ReplayBuffer()
        buffer.add(_transition())
        with pytest.raises(ValueError, match="batch"):
            buffer.sample(0, np.random.default_rng(0))

    def test_jsonl_round_trip(self, tmp_path):
        buffer = ReplayBuffer()
        for i in range(6):
            buffer.add(
                _transition(action=i % 3, reward=float(i), done=(i % 3 == 2)),
                episode_id=f"ep{i // 3}",
            )
        path = tmp_path / "log.jsonl"
        assert buffer.to_jsonl(path) == 6

        loaded = ReplayBuffer.from_jsonl(path)
        assert len(loaded) == 6
        assert loaded.actions().tolist() == buffer.actions().tolist()
        assert loaded.rewards().tolist() == buffer.rewards().tolist()
        assert loaded.dones().tolist() == buffer.dones().tolist()
        assert np.allclose(loaded.states(), buffer.states())
        assert loaded.episode_ids == ["ep0", "ep1"]
        assert len(loaded.episode("ep0")) == 3


class _FakeReplayEngine:
    """Minimal stand-in satisfying the ReplayEngine protocol surface we use."""

    def __init__(self, episodes: dict[str, list[EventEnvelope]]):
        self._episodes = episodes

    def episodes(self) -> list[str]:
        return list(self._episodes)

    def episode(self, episode_id: str) -> list[EventEnvelope]:
        return self._episodes[episode_id]


class TestBuildEpisodesFromReplay:
    def _engine(self) -> _FakeReplayEngine:
        def envelope(score: float, amount: float, action: int, prob: float) -> EventEnvelope:
            return EventEnvelope(
                kind="alert.decision",
                source="twin",
                payload={
                    "score": score,
                    "amount": amount,
                    "action": action,
                    "behaviour_prob": prob,
                    "label": "ignored-non-numeric",
                },
            )

        return _FakeReplayEngine(
            {
                "epA": [envelope(0.9, 100.0, 2, 0.6), envelope(0.2, 5.0, 0, 0.7)],
                "epB": [envelope(0.5, 50.0, 1, 0.5)],
                "empty": [],
            }
        )

    def test_bridges_envelopes_into_episodes(self):
        episodes = build_episodes_from_replay(
            self._engine(), reward_fn=lambda env: float(env.payload["score"]) * 10.0
        )
        assert [e.episode_id for e in episodes] == ["epA", "epB"]

        ep_a = episodes[0]
        assert len(ep_a) == 2
        # Feature keys are the sorted numeric payload keys minus the reserved ones.
        assert ep_a.metadata["feature_keys"] == ["amount", "score"]
        first, last = ep_a.transitions
        assert np.allclose(first.state, [100.0, 0.9])
        assert np.allclose(first.next_state, [5.0, 0.2])
        assert first.action == 2
        assert first.behaviour_prob == pytest.approx(0.6)
        assert first.reward == pytest.approx(9.0)
        assert first.done is False
        assert last.done is True
        # Terminal envelope transitions to itself.
        assert np.allclose(last.next_state, last.state)
        assert first.info["kind"] == "alert.decision"

    def test_explicit_feature_keys_pin_the_schema(self):
        episodes = build_episodes_from_replay(
            self._engine(), reward_fn=lambda env: 0.0, feature_keys=["score", "missing"]
        )
        state = episodes[0].transitions[0].state
        assert np.allclose(state, [0.9, 0.0])

    def test_buffer_from_episodes(self):
        episodes = build_episodes_from_replay(self._engine(), reward_fn=lambda env: 1.0)
        buffer = ReplayBuffer.from_episodes(episodes)
        assert len(buffer) == 3
        assert buffer.episode_ids == ["epA", "epB"]


class TestCollectEpisodes:
    def test_deterministic_and_propensity_logged(self):
        env = make_alert_triage_env(seed=3)

        def policy(state, rng):
            probs = np.array([0.2, 0.5, 0.3])
            action = int(rng.choice(3, p=probs))
            return action, float(probs[action])

        first = collect_episodes(env, policy, n_episodes=2, seed=11)
        second = collect_episodes(make_alert_triage_env(seed=99), policy, n_episodes=2, seed=11)
        assert len(first) == 2
        for ep1, ep2 in zip(first, second, strict=True):
            assert ep1.total_reward == pytest.approx(ep2.total_reward)
            for t1, t2 in zip(ep1, ep2, strict=True):
                assert t1.action == t2.action
                assert np.allclose(t1.state, t2.state)
                assert t1.behaviour_prob in (0.2, 0.5, 0.3)
        assert all(ep.transitions[-1].done for ep in first)
