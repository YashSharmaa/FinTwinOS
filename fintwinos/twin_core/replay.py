"""Episode recorder and replay engine for event envelopes.

Every envelope ingested into the twin is recorded into an *episode* — an
ordered stream of :class:`EventEnvelope`s. Episodes are the substrate for
counterfactual analysis ("replay yesterday's book through a different policy
pack"), for rebuilding twin state from scratch, and for evaluation fixtures.

Episodes live in memory by default; when constructed with a directory path
each episode is also persisted as a JSONL file (one envelope per line), so a
twin can be torn down and reconstructed bit-for-bit later.

Implements the :class:`fintwinos.core.interfaces.ReplayEngine` protocol.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from fintwinos.core.types import EventEnvelope

DEFAULT_EPISODE_ID = "default"

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


class JsonlReplayEngine:
    """Records envelopes into episodes; replays them through any handler.

    Parameters
    ----------
    path:
        Optional directory for persistence. When given, each episode is
        appended to ``<path>/<episode_id>.jsonl`` as it is recorded, and any
        existing ``*.jsonl`` episodes found at construction time are loaded
        back into memory. Episode ids are sanitised for use as file names
        (non ``[A-Za-z0-9._-]`` characters become ``_``); the authoritative
        episode id is always the one stored inside each envelope.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._episodes: dict[str, list[EventEnvelope]] = {}
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    # -- record ----------------------------------------------------------------------

    def record(self, envelope: EventEnvelope) -> None:
        """Append an envelope to its episode (``envelope.episode_id`` or 'default').

        Envelopes without an episode id are stored (and persisted) with the
        default episode id filled in, so a JSONL round-trip is loss-free.
        """
        episode_id = envelope.episode_id or DEFAULT_EPISODE_ID
        stored = (
            envelope
            if envelope.episode_id
            else envelope.model_copy(update={"episode_id": episode_id})
        )
        self._episodes.setdefault(episode_id, []).append(stored)
        if self.path is not None:
            target = self._file_for(episode_id)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(stored.model_dump_json() + "\n")

    # -- read ------------------------------------------------------------------------

    def episode(self, episode_id: str) -> list[EventEnvelope]:
        """The recorded envelopes of one episode, in record order (copy)."""
        return list(self._episodes.get(episode_id, []))

    def episodes(self) -> list[str]:
        """All known episode ids, sorted."""
        return sorted(self._episodes)

    def __len__(self) -> int:
        return sum(len(envelopes) for envelopes in self._episodes.values())

    # -- replay -----------------------------------------------------------------------

    def replay(self, episode_id: str, handler: Callable[[EventEnvelope], None]) -> int:
        """Feed every envelope of an episode through ``handler``, in order.

        Returns the number of envelopes replayed (0 for an unknown episode).
        The handler is typically a fresh :class:`TwinIngestor`'s ``ingest`` —
        replaying an episode through an empty twin reconstructs its state
        deterministically.
        """
        envelopes = self._episodes.get(episode_id, [])
        for envelope in envelopes:
            handler(envelope)
        return len(envelopes)

    # -- internal ----------------------------------------------------------------------

    def _file_for(self, episode_id: str) -> Path:
        assert self.path is not None
        safe = _SAFE_FILENAME_RE.sub("_", episode_id) or DEFAULT_EPISODE_ID
        return self.path / f"{safe}.jsonl"

    def _load_existing(self) -> None:
        assert self.path is not None
        for file in sorted(self.path.glob("*.jsonl")):
            with file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    envelope = EventEnvelope.model_validate_json(line)
                    episode_id = envelope.episode_id or file.stem
                    self._episodes.setdefault(episode_id, []).append(envelope)
