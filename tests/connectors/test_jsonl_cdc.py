"""Tests for the JSONL change-data-capture connector and its resume offsets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fintwinos.connectors.base import ConnectorError
from fintwinos.connectors.jsonl_cdc import JsonlCdcConnector

RECORDS = [
    {
        "table": "accounts",
        "op": "insert",
        "key": "acc_001",
        "after": {"balance": 100.0, "currency": "USD"},
        "ts": "2026-01-05T09:00:00+00:00",
    },
    {
        "table": "accounts",
        "op": "update",
        "key": "acc_001",
        "before": {"balance": 100.0},
        "after": {"balance": 75.0},
        "ts": "2026-01-05T09:30:00+00:00",
    },
    {
        "table": "trades",
        "op": "delete",
        "key": "trd_009",
        "before": {"quantity": 10},
        "ts": "2026-01-05T10:00:00Z",
    },
]


def write_jsonl(path: Path, records: list[dict], append: bool = False) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def make_connector(path: Path, settings, **kwargs) -> JsonlCdcConnector:
    kwargs.setdefault("entity_types", {"accounts": "account", "trades": "trade"})
    return JsonlCdcConnector(path, settings=settings, **kwargs)


async def collect(connector):
    return [envelope async for envelope in connector.stream()]


async def test_cdc_records_map_to_table_op_kinds(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    write_jsonl(path, RECORDS)
    connector = make_connector(path, offline_settings)

    envelopes = await collect(connector)

    assert [e.kind for e in envelopes] == [
        "accounts.insert",
        "accounts.update",
        "trades.delete",
    ]
    update = envelopes[1]
    assert update.source == "cdc"
    assert update.payload["before"] == {"balance": 100.0}
    assert update.payload["after"] == {"balance": 75.0}
    assert update.occurred_at == datetime(2026, 1, 5, 9, 30, tzinfo=UTC)
    # entity ref uses the configured entity type mapping
    assert [ref.key() for ref in update.entities] == ["account:acc_001"]
    assert [ref.key() for ref in envelopes[2].entities] == ["trade:trd_009"]
    # provenance hashes the raw line
    assert update.provenance is not None
    assert update.provenance.record_hash
    assert "#line2" in (update.provenance.notes or "")


async def test_resume_offsets_skip_already_consumed_lines(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    write_jsonl(path, RECORDS)

    first = await collect(make_connector(path, offline_settings))
    assert len(first) == 3

    # nothing new: a fresh connector instance resumes past everything
    second = await collect(make_connector(path, offline_settings))
    assert second == []

    # appended lines are picked up from the persisted offset
    write_jsonl(
        path,
        [{"table": "accounts", "op": "insert", "key": "acc_002", "after": {"balance": 5}}],
        append=True,
    )
    third = await collect(make_connector(path, offline_settings))
    assert len(third) == 1
    assert third[0].kind == "accounts.insert"
    assert third[0].payload["key"] == "acc_002"


async def test_reset_offset_replays_from_the_top(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    write_jsonl(path, RECORDS)

    connector = make_connector(path, offline_settings)
    assert len(await collect(connector)) == 3
    assert connector.offset_path.exists()

    connector.reset_offset()
    replayed = await collect(make_connector(path, offline_settings))
    assert len(replayed) == 3


async def test_resume_disabled_reads_from_top(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    write_jsonl(path, RECORDS)
    assert len(await collect(make_connector(path, offline_settings))) == 3
    again = await collect(make_connector(path, offline_settings, resume=False))
    assert len(again) == 3


async def test_truncated_source_resets_offset(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    write_jsonl(path, RECORDS)
    assert len(await collect(make_connector(path, offline_settings))) == 3

    # simulate log rotation: the file is rewritten shorter than the stored offset
    write_jsonl(path, RECORDS[:1])
    envelopes = await collect(make_connector(path, offline_settings))
    assert len(envelopes) == 1


async def test_malformed_line_strict_raises(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    path.write_text('{"table": "accounts", "op": "insert"}\nnot json at all\n')
    with pytest.raises(ConnectorError, match="invalid JSON"):
        await collect(make_connector(path, offline_settings))


async def test_malformed_lines_skipped_when_not_strict(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    good = json.dumps(RECORDS[0])
    path.write_text(f"{good}\nnot json\n{json.dumps(RECORDS[1])}\n")
    connector = make_connector(path, offline_settings, strict=False)
    envelopes = await collect(connector)
    assert [e.kind for e in envelopes] == ["accounts.insert", "accounts.update"]
    assert connector.skipped == 1
    # the bad line's offset was committed: nothing is re-read on resume
    assert await collect(make_connector(path, offline_settings, strict=False)) == []


async def test_invalid_op_rejected(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    write_jsonl(path, [{"table": "accounts", "op": "upsert", "key": "a"}])
    with pytest.raises(ConnectorError, match="'op' must be one of"):
        await collect(make_connector(path, offline_settings))


async def test_missing_table_rejected(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    write_jsonl(path, [{"op": "insert", "key": "a"}])
    with pytest.raises(ConnectorError, match="'table'"):
        await collect(make_connector(path, offline_settings))


async def test_blank_lines_are_ignored(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    body = json.dumps(RECORDS[0]) + "\n\n\n" + json.dumps(RECORDS[1]) + "\n"
    path.write_text(body, encoding="utf-8")
    envelopes = await collect(make_connector(path, offline_settings))
    assert len(envelopes) == 2


async def test_follow_mode_stops_after_idle_timeout(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    write_jsonl(path, RECORDS)
    connector = make_connector(
        path, offline_settings, follow=True, poll_interval=0.01, idle_timeout=0.05
    )
    envelopes = await collect(connector)
    assert len(envelopes) == 3  # drains, then gives up after the idle window


async def test_missing_file_raises(tmp_path, offline_settings):
    connector = make_connector(tmp_path / "nope.jsonl", offline_settings)
    with pytest.raises(ConnectorError, match="not found"):
        await collect(connector)


async def test_unmapped_table_uses_table_name_as_entity_type(tmp_path, offline_settings):
    path = tmp_path / "cdc.jsonl"
    write_jsonl(path, [{"table": "loans", "op": "insert", "id": "ln_1"}])
    connector = JsonlCdcConnector(path, settings=offline_settings)
    envelopes = await collect(connector)
    assert [ref.key() for ref in envelopes[0].entities] == ["loans:ln_1"]
