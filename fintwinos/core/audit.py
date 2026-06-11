"""Tamper-evident audit trail.

Every observation, tool call, policy check, simulation branch, human approval and
write-back event is appended as a hash-chained record: each record's hash covers its
canonical JSON body plus the previous record's hash, so any retrospective edit breaks
the chain and is detectable with ``verify()``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fintwinos.core.types import utcnow

GENESIS_HASH = "0" * 64


class AuditRecord(BaseModel):
    seq: int
    ts: datetime = Field(default_factory=utcnow)
    actor: str
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str
    hash: str = ""

    def body_for_hash(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "ts": self.ts.isoformat(),
                "actor": self.actor,
                "action": self.action,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
            default=str,
        )

    def compute_hash(self) -> str:
        return hashlib.sha256(self.body_for_hash().encode("utf-8")).hexdigest()


class AuditTrail:
    """In-memory hash chain with optional JSONL persistence."""

    def __init__(self, path: Path | str | None = None):
        self._records: list[AuditRecord] = []
        self._lock = threading.Lock()
        self.path = Path(path) if path is not None else None
        if self.path is not None and self.path.exists():
            self._load()

    # -- write ---------------------------------------------------------------

    def append(self, actor: str, action: str, payload: dict[str, Any] | None = None) -> AuditRecord:
        with self._lock:
            prev = self._records[-1].hash if self._records else GENESIS_HASH
            record = AuditRecord(
                seq=len(self._records),
                actor=actor,
                action=action,
                payload=payload or {},
                prev_hash=prev,
            )
            record.hash = record.compute_hash()
            self._records.append(record)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(record.model_dump_json() + "\n")
            return record

    # -- read ----------------------------------------------------------------

    def records(self, action: str | None = None, actor: str | None = None) -> list[AuditRecord]:
        out = list(self._records)
        if action is not None:
            out = [r for r in out if r.action == action]
        if actor is not None:
            out = [r for r in out if r.actor == actor]
        return out

    def __len__(self) -> int:
        return len(self._records)

    def last(self) -> AuditRecord | None:
        return self._records[-1] if self._records else None

    # -- integrity -------------------------------------------------------------

    def verify(self) -> bool:
        """Walk the chain; return False on any broken link or recomputed-hash mismatch.

        Detects in-place edits, insertions and deletions anywhere before the
        current tail. It cannot, by construction, detect truncation of the
        *tail itself* (a shortened chain is still internally consistent) —
        deployments that need truncation evidence should anchor ``last().hash``
        externally (e.g. ship it to a log sink or print it in run reports).
        """
        prev = GENESIS_HASH
        for record in self._records:
            if record.prev_hash != prev:
                return False
            if record.compute_hash() != record.hash:
                return False
            prev = record.hash
        return True

    # -- persistence -------------------------------------------------------------

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._records.append(AuditRecord.model_validate_json(line))

    @classmethod
    def load(cls, path: Path | str) -> AuditTrail:
        return cls(path=path)
