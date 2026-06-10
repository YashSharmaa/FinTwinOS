"""Change-data-capture ingestion from JSONL logs.

``JsonlCdcConnector`` tails a JSON-lines CDC file where each line is one change
record::

    {"table": "accounts", "op": "update", "key": "acc_001",
     "before": {"balance": 100.0}, "after": {"balance": 75.0},
     "ts": "2026-01-05T09:30:00Z"}

Each record becomes an :class:`~fintwinos.core.types.EventEnvelope` with kind
``"<table>.<op>"`` (``op`` must be ``insert``, ``update`` or ``delete``). The byte
offset of the last *consumed* line is persisted under
``settings.data_dir / "connectors" / "offsets"``, so a restarted pipeline resumes
where it left off. Offsets are saved only after the consumer takes the next
envelope, giving at-least-once delivery across crashes (a consumer that dies
mid-envelope sees that envelope again on resume).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fintwinos.connectors.base import BaseConnector, ConnectorError
from fintwinos.core.config import Settings, get_settings
from fintwinos.core.types import EntityRef, EventEnvelope, Provenance, utcnow

VALID_OPS = frozenset({"insert", "update", "delete"})

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _parse_record_ts(raw: Any) -> datetime | None:
    """Parse a record's ``ts`` field (ISO-8601); naive values are assumed UTC."""
    if raw is None or not isinstance(raw, str) or not raw.strip():
        return None
    dt = datetime.fromisoformat(raw.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class JsonlCdcConnector(BaseConnector):
    """Tail a JSONL CDC file into ``<table>.<op>`` envelopes with resumable offsets.

    Args:
        path: The JSONL file to read.
        name: Connector name; also keys the offset file. Defaults to
            ``cdc_<file stem>``.
        source: Stamped as ``EventEnvelope.source`` (default ``"cdc"``).
        entity_types: Optional ``table -> entity_type`` map for the
            :class:`~fintwinos.core.types.EntityRef` built from each record's key;
            unmapped tables use the table name as the entity type.
        key_fields: Record fields probed (in order) for the entity identifier.
        settings: Settings providing ``data_dir`` for offset persistence.
        resume: When true (default), start from the persisted offset; when false,
            always read from the top (offsets are still written).
        follow: When true, keep polling for appended lines after EOF until
            ``idle_timeout`` seconds pass with no new data. When false (default),
            stop at EOF — batch semantics.
        poll_interval: Sleep between EOF polls in follow mode, seconds.
        idle_timeout: Follow-mode give-up threshold, seconds (``None`` = forever).
        strict: When true (default), malformed lines abort the stream with
            :class:`~fintwinos.connectors.base.ConnectorError`; when false they are
            skipped, counted in :attr:`skipped`, and their offset is committed so
            they are never re-read.
        episode_id: Optional episode tag stamped on every envelope.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        name: str | None = None,
        source: str = "cdc",
        entity_types: dict[str, str] | None = None,
        key_fields: Sequence[str] = ("key", "pk", "id"),
        settings: Settings | None = None,
        resume: bool = True,
        follow: bool = False,
        poll_interval: float = 0.05,
        idle_timeout: float | None = 1.0,
        strict: bool = True,
        episode_id: str | None = None,
    ):
        self.path = Path(path)
        self.entity_types = dict(entity_types or {})
        self.key_fields = tuple(key_fields)
        self.settings = settings or get_settings()
        self.resume = resume
        self.follow = follow
        self.poll_interval = poll_interval
        self.idle_timeout = idle_timeout
        self.strict = strict
        self.episode_id = episode_id
        self.skipped = 0
        super().__init__(name=name or f"cdc_{self.path.stem}", source=source)

    # -- offset persistence -----------------------------------------------------

    @property
    def offset_path(self) -> Path:
        """Location of this connector's persisted offset file."""
        safe = _SAFE_NAME_RE.sub("_", self.name)
        return self.settings.data_dir / "connectors" / "offsets" / f"{safe}.json"

    def _load_offset(self) -> tuple[int, int]:
        """Return ``(byte_offset, lines_consumed)``, falling back to ``(0, 0)``.

        The offset is discarded when resume is disabled, the offset file is
        unreadable, it refers to a different source path, or the source file has
        shrunk below the stored offset (rotation/truncation).
        """
        if not self.resume or not self.offset_path.exists():
            return 0, 0
        try:
            data = json.loads(self.offset_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0, 0
        if data.get("source_path") != str(self.path):
            return 0, 0
        offset = int(data.get("offset", 0))
        lines = int(data.get("lines", 0))
        try:
            size = self.path.stat().st_size
        except OSError:
            return 0, 0
        if offset > size:
            return 0, 0  # source was truncated or rotated; start over
        return offset, lines

    def _save_offset(self, offset: int, lines: int) -> None:
        """Persist the resume point atomically (write temp file, then rename)."""
        self.offset_path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            {
                "source_path": str(self.path),
                "offset": offset,
                "lines": lines,
                "updated_at": utcnow().isoformat(),
            },
            sort_keys=True,
        )
        tmp = self.offset_path.with_suffix(".json.tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(self.offset_path)

    def reset_offset(self) -> None:
        """Delete the persisted offset so the next stream re-reads from the top."""
        self.offset_path.unlink(missing_ok=True)

    # -- mapping ------------------------------------------------------------------

    def _envelope_for_line(self, line: str, line_number: int) -> EventEnvelope:
        """Map one raw JSONL line to an envelope.

        Raises:
            ValueError: On malformed JSON, a non-object record, a missing/empty
                ``table``, or an ``op`` outside ``insert``/``update``/``delete``.
        """
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: record must be a JSON object")

        table = record.get("table")
        if not isinstance(table, str) or not table.strip():
            raise ValueError(f"line {line_number}: missing or empty 'table' field")
        table = table.strip()

        op = record.get("op")
        if not isinstance(op, str) or op.strip().lower() not in VALID_OPS:
            raise ValueError(
                f"line {line_number}: 'op' must be one of {sorted(VALID_OPS)}, got {op!r}"
            )
        op = op.strip().lower()

        try:
            occurred_at = _parse_record_ts(record.get("ts"))
        except ValueError as exc:
            raise ValueError(f"line {line_number}: cannot parse 'ts': {exc}") from exc

        entities: list[EntityRef] = []
        for field in self.key_fields:
            value = record.get(field)
            if value is not None and str(value).strip():
                entity_type = self.entity_types.get(table, table)
                entities.append(EntityRef(entity_type=entity_type, entity_id=str(value)))
                break

        provenance = Provenance(
            source_system=self.source,
            record_hash=hashlib.sha256(line.encode("utf-8")).hexdigest(),
            notes=f"{self.path}#line{line_number}",
        )
        kwargs: dict[str, Any] = {
            "kind": f"{table}.{op}",
            "source": self.source,
            "entities": entities,
            "payload": dict(record),
            "episode_id": self.episode_id,
            "provenance": provenance,
        }
        if occurred_at is not None:
            kwargs["occurred_at"] = occurred_at
        return EventEnvelope(**kwargs)

    # -- streaming ------------------------------------------------------------------

    async def stream(self) -> AsyncIterator[EventEnvelope]:
        """Yield envelopes for every CDC line after the persisted offset.

        In batch mode (``follow=False``) the stream ends at EOF. In follow mode it
        keeps polling for appended lines until ``idle_timeout`` seconds elapse with
        no new complete line. Partial lines (no trailing newline yet) are left in
        place in follow mode and retried on the next poll.
        """
        if not self.path.exists():
            raise ConnectorError(f"CDC file not found: {self.path}")

        offset, lines_consumed = self._load_offset()
        idle = 0.0
        with self.path.open("rb") as fh:
            fh.seek(offset)
            while True:
                line_start = fh.tell()
                raw = fh.readline()
                if not raw:
                    if not self.follow:
                        break
                    if self.idle_timeout is not None and idle >= self.idle_timeout:
                        break
                    await asyncio.sleep(self.poll_interval)
                    idle += self.poll_interval
                    continue
                if self.follow and not raw.endswith(b"\n"):
                    # A writer is mid-line; rewind and retry on the next poll.
                    fh.seek(line_start)
                    if self.idle_timeout is not None and idle >= self.idle_timeout:
                        break
                    await asyncio.sleep(self.poll_interval)
                    idle += self.poll_interval
                    continue

                idle = 0.0
                offset = fh.tell()
                lines_consumed += 1
                line = raw.decode("utf-8").strip()
                if not line:
                    self._save_offset(offset, lines_consumed)
                    continue
                try:
                    envelope = self._envelope_for_line(line, lines_consumed)
                except ValueError as exc:
                    if self.strict:
                        raise ConnectorError(f"{self.path}: {exc}") from exc
                    self.skipped += 1
                    self._save_offset(offset, lines_consumed)
                    continue

                yield envelope
                # Reached only once the consumer asks for the next envelope, so a
                # crash mid-envelope re-delivers it on resume (at-least-once).
                self._save_offset(offset, lines_consumed)
                await asyncio.sleep(0)

        self._save_offset(offset, lines_consumed)
