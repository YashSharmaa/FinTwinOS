"""Batch CSV ingestion: declarative column-to-envelope mapping.

``CsvBatchConnector`` turns each row of a CSV export (trades, payments, customer
extracts, ...) into one :class:`~fintwinos.core.types.EventEnvelope` according to a
:class:`ColumnMapping`. The mapping is fully declarative — kind, source, entity
columns, payload columns, timestamp column and format — so the same connector class
covers every flat-file feed without subclassing.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fintwinos.connectors.base import BaseConnector, ConnectorError
from fintwinos.core.types import EntityRef, EventEnvelope, Provenance

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def coerce_value(raw: str | None) -> Any:
    """Coerce one CSV cell to a useful Python scalar.

    Rules (in order): ``None`` and empty/whitespace cells become ``None``;
    case-insensitive ``true``/``false`` become booleans; integer-looking text
    becomes ``int``; float-looking text (including exponents) becomes ``float``;
    everything else stays a stripped string.
    """
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def parse_timestamp(raw: str, fmt: str | None) -> datetime:
    """Parse a timestamp cell; naive results are assumed UTC.

    Args:
        raw: The cell text.
        fmt: A ``strptime`` format string, or ``None`` for ISO-8601
            (``datetime.fromisoformat``, which accepts a trailing ``Z``).
    """
    text = raw.strip()
    dt = datetime.strptime(text, fmt) if fmt else datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class ColumnMapping(BaseModel):
    """Declarative recipe mapping one CSV row to one ``EventEnvelope``.

    Attributes:
        kind: Event kind stamped on every envelope (e.g. ``"trade.executed"``).
        source: Source-system name for ``EventEnvelope.source`` and provenance.
        kind_column: Optional column whose (non-empty) value overrides ``kind``
            per row, for mixed-kind exports.
        entity_columns: ``entity_type -> column name``; each non-empty cell becomes
            an :class:`~fintwinos.core.types.EntityRef` on the envelope.
        payload_columns: Columns copied into the payload. ``None`` means every
            column except the timestamp column (entity columns are kept — the raw
            identifiers are useful downstream).
        timestamp_column: Column providing ``occurred_at``; required to be present
            and non-empty on every row when set. ``None`` uses ingestion time.
        timestamp_format: ``strptime`` format for the timestamp column; ``None``
            means ISO-8601.
        coerce_numeric: When true (default), payload cells pass through
            :func:`coerce_value`; when false they stay raw strings.
        episode_id: Optional episode tag stamped on every envelope so a whole file
            lands in one replayable episode.
    """

    kind: str = Field(min_length=1)
    source: str = Field(min_length=1)
    kind_column: str | None = None
    entity_columns: dict[str, str] = Field(default_factory=dict)
    payload_columns: list[str] | None = None
    timestamp_column: str | None = None
    timestamp_format: str | None = None
    coerce_numeric: bool = True
    episode_id: str | None = None


def row_to_envelope(
    mapping: ColumnMapping,
    row: dict[str, Any],
    *,
    row_number: int,
    origin: str,
) -> EventEnvelope:
    """Apply a :class:`ColumnMapping` to one parsed CSV row.

    Args:
        mapping: The declarative mapping.
        row: One ``csv.DictReader`` row (header -> cell text).
        row_number: 1-based physical line number, used in error messages and
            provenance notes.
        origin: Label for the file the row came from.

    Raises:
        ValueError: On a missing/empty timestamp cell, an unparseable timestamp,
            or a payload column named in the mapping but absent from the row.
    """
    kind = mapping.kind
    if mapping.kind_column:
        cell = row.get(mapping.kind_column)
        override = str(cell).strip() if cell is not None else ""
        if override:
            kind = override

    entities: list[EntityRef] = []
    for entity_type, column in mapping.entity_columns.items():
        cell = row.get(column)
        value = str(cell).strip() if cell is not None else ""
        if value:
            entities.append(EntityRef(entity_type=entity_type, entity_id=value))

    if mapping.timestamp_column is not None:
        raw_ts = row.get(mapping.timestamp_column)
        if raw_ts is None or str(raw_ts).strip() == "":
            raise ValueError(
                f"row {row_number}: missing timestamp column '{mapping.timestamp_column}'"
            )
        try:
            occurred_at = parse_timestamp(str(raw_ts), mapping.timestamp_format)
        except ValueError as exc:
            raise ValueError(
                f"row {row_number}: cannot parse timestamp {raw_ts!r} "
                f"with format {mapping.timestamp_format!r}: {exc}"
            ) from exc
    else:
        occurred_at = None

    if mapping.payload_columns is not None:
        columns = mapping.payload_columns
        for column in columns:
            if column not in row:
                raise ValueError(f"row {row_number}: missing payload column '{column}'")
    else:
        columns = [c for c in row if c is not None and c != mapping.timestamp_column]

    payload: dict[str, Any] = {}
    for column in columns:
        cell = row.get(column)
        payload[column] = coerce_value(cell) if mapping.coerce_numeric else cell

    record_hash = hashlib.sha256(
        json.dumps(
            {k: v for k, v in row.items() if k is not None}, sort_keys=True, default=str
        ).encode("utf-8")
    ).hexdigest()
    provenance = Provenance(
        source_system=mapping.source,
        record_hash=record_hash,
        notes=f"{origin}#row{row_number}",
    )

    kwargs: dict[str, Any] = {
        "kind": kind,
        "source": mapping.source,
        "entities": entities,
        "payload": payload,
        "episode_id": mapping.episode_id,
        "provenance": provenance,
    }
    if occurred_at is not None:
        kwargs["occurred_at"] = occurred_at
    return EventEnvelope(**kwargs)


class CsvBatchConnector(BaseConnector):
    """Stream a CSV file as envelopes, one per data row.

    The file is parsed off the event loop (``asyncio.to_thread``) and rows are
    yielded cooperatively. Mapping failures on individual rows either abort the
    stream (``strict=True``, the default — batch loads should be loud) or skip the
    row and increment :attr:`rows_skipped` (``strict=False``).

    Attributes:
        rows_read: Data rows parsed from the file on the last stream.
        rows_skipped: Rows dropped due to mapping errors (non-strict mode only).
    """

    def __init__(
        self,
        path: Path | str,
        mapping: ColumnMapping,
        *,
        name: str | None = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        strict: bool = True,
    ):
        self.path = Path(path)
        self.mapping = mapping
        self.delimiter = delimiter
        self.encoding = encoding
        self.strict = strict
        self.rows_read = 0
        self.rows_skipped = 0
        super().__init__(
            name=name or f"csv_batch[{self.path.name}]",
            source=mapping.source,
        )

    def _read_rows(self) -> list[dict[str, Any]]:
        """Blocking CSV parse, run in a worker thread by :meth:`stream`."""
        if not self.path.exists():
            raise ConnectorError(f"CSV file not found: {self.path}")
        with self.path.open("r", encoding=self.encoding, newline="") as fh:
            reader = csv.DictReader(fh, delimiter=self.delimiter)
            if reader.fieldnames is None:
                raise ConnectorError(f"CSV file has no header row: {self.path}")
            return list(reader)

    async def stream(self) -> AsyncIterator[EventEnvelope]:
        """Yield one envelope per CSV data row, in file order."""
        rows = await asyncio.to_thread(self._read_rows)
        self.rows_read = len(rows)
        self.rows_skipped = 0
        for index, row in enumerate(rows):
            row_number = index + 2  # 1-based physical line, after the header
            try:
                envelope = row_to_envelope(
                    self.mapping, row, row_number=row_number, origin=str(self.path)
                )
            except ValueError as exc:
                if self.strict:
                    raise ConnectorError(f"{self.path}: {exc}") from exc
                self.rows_skipped += 1
                continue
            yield envelope
            await asyncio.sleep(0)
