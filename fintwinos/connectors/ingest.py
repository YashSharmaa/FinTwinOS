"""Operator-facing ingestion helpers: a file or an EDGAR CIK into a twin.

Thin orchestration over the connectors and the canonical
:func:`fintwinos.connectors.base.pump` driver, used by the ``fintwinos ingest``
CLI command. This module introduces **no new ingestion semantics**: every
envelope still flows through ``runtime.ingestor`` (CONTRACTS.md rule 5) and the
tamper-evident audit trail. It only chooses the right connector for a path and,
for EDGAR, remaps the connector's ``filing.indexed`` kind onto the twin's
``document.*`` routing band so filings land in the document store.

>>> from fintwinos.twin_core.runtime import build_runtime
>>> rt = build_runtime(with_demo_data=False)
>>> summary = ingest_path("trades.csv", rt, kind="trade.executed", entity_type="trade")  # doctest: +SKIP
"""

from __future__ import annotations

import asyncio
import csv
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

from fintwinos.connectors.base import BaseConnector, ConnectorError, pump
from fintwinos.connectors.csv_batch import ColumnMapping, CsvBatchConnector
from fintwinos.connectors.edgar import EdgarConnector
from fintwinos.connectors.jsonl_cdc import JsonlCdcConnector
from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import EventEnvelope

#: File extensions handled as flat CSV/TSV exports.
CSV_SUFFIXES = {".csv", ".tsv"}
#: File extensions handled as JSON-lines change-data-capture logs.
CDC_SUFFIXES = {".jsonl", ".ndjson"}

#: Header names probed (in order) for a CSV row's ``occurred_at`` timestamp.
_TS_CANDIDATES = (
    "occurred_at",
    "timestamp",
    "ts",
    "event_time",
    "datetime",
    "date",
    "time",
)


class _TransformingConnector(BaseConnector):
    """Wrap a connector, applying a per-envelope transform before emission.

    Used to stamp an episode id onto the EDGAR connector's envelopes (its
    ``filing.*`` kind already routes into the document store) without copying the
    connector's network/cache logic.
    """

    def __init__(
        self,
        inner: Any,
        transform: Callable[[EventEnvelope], EventEnvelope],
        *,
        name: str | None = None,
    ) -> None:
        self._inner = inner
        self._transform = transform
        super().__init__(
            name=name or getattr(inner, "name", "transforming"),
            source=getattr(inner, "source", "unknown"),
        )

    async def stream(self) -> AsyncIterator[EventEnvelope]:
        async for envelope in self._inner.stream():
            yield self._transform(envelope)


def _read_header(path: Path, delimiter: str) -> list[str]:
    """Return the (stripped) header row of a CSV/TSV file, or ``[]`` if empty."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter=delimiter):
            return [cell.strip() for cell in row]
    return []


def _csv_mapping(
    path: Path,
    header: Sequence[str],
    *,
    kind: str | None,
    entity_type: str | None,
    episode_id: str | None,
) -> ColumnMapping:
    """Infer a :class:`ColumnMapping` from a CSV header + operator hints.

    The entity column is found by probing ``<type>_id``, ``<type>``, ``id`` and
    ``entity_id``; the timestamp column by probing :data:`_TS_CANDIDATES`. All
    columns become payload fields. ``kind`` defaults to ``"record.observed"``
    (which the ingestor records and audits as *unrouted* — pass a real routed
    kind such as ``"trade.executed"`` to land the rows in the stores).
    """
    columns = {c for c in header if c}
    entity_columns: dict[str, str] = {}
    if entity_type:
        for candidate in (f"{entity_type}_id", entity_type, "id", "entity_id"):
            if candidate in columns:
                entity_columns[entity_type] = candidate
                break
    timestamp_column = next((c for c in _TS_CANDIDATES if c in columns), None)
    return ColumnMapping(
        kind=kind or "record.observed",
        source=f"cli_ingest:{path.name}",
        entity_columns=entity_columns,
        timestamp_column=timestamp_column,
        episode_id=episode_id or f"ingest:{path.stem}",
    )


def _drain(connector: Any, runtime: TwinRuntime, *, limit: int | None) -> dict[str, Any]:
    """Pump a connector into ``runtime.ingestor`` and enrich the summary.

    ``pump``'s ``ingested`` counter only means "did not raise" — an envelope
    with an unrouted ``kind`` does not raise, so this adds a ``routed`` figure
    (the change in ``ingestor.ingested_count``) that reflects envelopes that
    actually reached a store.
    """
    ingestor = getattr(runtime, "ingestor", None)
    if ingestor is None:
        raise ConnectorError("runtime has no ingestor; build it with build_runtime()")
    before = int(getattr(ingestor, "ingested_count", 0))
    summary = asyncio.run(pump(connector, ingestor, limit=limit))
    after = int(getattr(ingestor, "ingested_count", before))
    summary["routed"] = after - before
    summary["unrouted"] = summary["ingested"] - summary["routed"]
    return summary


def ingest_path(
    path: Path | str,
    runtime: TwinRuntime,
    *,
    kind: str | None = None,
    entity_type: str | None = None,
    episode_id: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Ingest a CSV/TSV export or a JSONL change-data-capture log into a twin.

    The file format is chosen by suffix:

    - ``.csv`` / ``.tsv`` -> :class:`CsvBatchConnector` with a mapping inferred
      from the header (see :func:`_csv_mapping`). ``kind`` overrides the event
      kind for every row; ``entity_type`` names the entity-id column to extract.
    - ``.jsonl`` / ``.ndjson`` -> :class:`JsonlCdcConnector`. Each record's kind
      is derived as ``"<table>.<op>"``; the table name is used as the entity
      type. (``kind``/``entity_type`` apply to CSV rows only.)

    Rows that fail mapping are skipped (non-strict) and reported; rows that fail
    ingestion are isolated by :func:`pump` and counted in ``failed``.

    Returns:
        The :func:`pump` summary, plus ``source_path``, ``format``, ``routed``
        and ``unrouted`` keys.
    """
    p = Path(path)
    if not p.exists():
        raise ConnectorError(f"ingest source not found: {p}")
    suffix = p.suffix.lower()
    ep = episode_id or f"ingest:{p.stem}"

    if suffix in CDC_SUFFIXES:
        connector: Any = JsonlCdcConnector(
            p,
            resume=False,
            strict=False,
            episode_id=ep,
            entity_types={entity_type: entity_type} if entity_type else None,
        )
        fmt = "jsonl-cdc"
    elif suffix in CSV_SUFFIXES:
        delimiter = "\t" if suffix == ".tsv" else ","
        header = _read_header(p, delimiter)
        if not header:
            raise ConnectorError(f"CSV file has no header row: {p}")
        mapping = _csv_mapping(
            p, header, kind=kind, entity_type=entity_type, episode_id=ep
        )
        connector = CsvBatchConnector(p, mapping, delimiter=delimiter, strict=False)
        fmt = "csv"
    else:
        raise ConnectorError(
            f"unsupported ingest format {suffix!r} for {p}; "
            f"expected one of {sorted(CSV_SUFFIXES | CDC_SUFFIXES)}"
        )

    summary = _drain(connector, runtime, limit=limit)
    summary["source_path"] = str(p)
    summary["format"] = fmt
    return summary


def ingest_edgar(
    cik: int | str,
    runtime: TwinRuntime,
    *,
    limit: int | None = 5,
    forms: Sequence[str] | None = None,
    episode_id: str | None = None,
) -> dict[str, Any]:
    """Ingest SEC EDGAR filings for one CIK into a twin's document store.

    The :class:`EdgarConnector` emits ``filing.indexed`` envelopes, which the
    ingestor routes into the document store via its ``filing.*`` alias of the
    ``document.*`` handler (each filing is linked to its legal-entity and
    instrument nodes). This helper additionally stamps an episode id so a whole
    CIK lands in one replayable episode. Offline with no cached EDGAR JSON, the
    underlying connector raises ``EdgarOfflineError`` — prime the cache by
    running once online or by placing fixture JSON under ``<data_dir>/edgar/``
    (see the EDGAR connector docs).

    Returns:
        The :func:`pump` summary, plus ``cik``, ``format``, ``routed`` and
        ``unrouted`` keys.
    """
    inner = EdgarConnector(
        cik,
        limit_per_cik=limit,
        forms=list(forms) if forms else None,
    )
    ep = episode_id or f"edgar:{cik}"

    def tag_episode(envelope: EventEnvelope) -> EventEnvelope:
        return envelope.model_copy(update={"episode_id": ep})

    connector = _TransformingConnector(inner, tag_episode, name=f"edgar_ingest[{cik}]")
    summary = _drain(connector, runtime, limit=None)
    summary["cik"] = str(cik)
    summary["format"] = "edgar"
    return summary
