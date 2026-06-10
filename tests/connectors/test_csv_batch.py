"""Tests for the CSV batch connector and its declarative column mapping."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fintwinos.connectors.base import ConnectorError, pump
from fintwinos.connectors.csv_batch import (
    ColumnMapping,
    CsvBatchConnector,
    coerce_value,
    row_to_envelope,
)

TRADES_CSV = """trade_id,account_id,instrument,side,quantity,price,executed_at
trd_001,acc_001,AAPL,buy,100,191.45,2024-05-02 14:31:07
trd_002,acc_002,MSFT,sell,50,402.10,2024-05-02 14:32:11
trd_003,acc_001,AAPL,sell,25,191.90,2024-05-02 15:05:42
"""


def write_csv(tmp_path: Path, body: str, name: str = "trades.csv") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def trades_mapping() -> ColumnMapping:
    return ColumnMapping(
        kind="trade.executed",
        source="oms_csv",
        entity_columns={"account": "account_id", "instrument": "instrument"},
        timestamp_column="executed_at",
        timestamp_format="%Y-%m-%d %H:%M:%S",
    )


async def collect(connector):
    return [envelope async for envelope in connector.stream()]


async def test_csv_rows_map_to_envelopes(tmp_path):
    path = write_csv(tmp_path, TRADES_CSV)
    connector = CsvBatchConnector(path, trades_mapping())

    envelopes = await collect(connector)

    assert len(envelopes) == 3
    assert connector.rows_read == 3
    first = envelopes[0]
    assert first.kind == "trade.executed"
    assert first.source == "oms_csv"
    # numeric coercion: ints stay ints, decimals become floats
    assert first.payload["quantity"] == 100
    assert isinstance(first.payload["quantity"], int)
    assert first.payload["price"] == pytest.approx(191.45)
    assert isinstance(first.payload["price"], float)
    assert first.payload["side"] == "buy"
    # timestamp column is parsed into occurred_at and excluded from payload
    assert "executed_at" not in first.payload
    assert first.occurred_at == datetime(2024, 5, 2, 14, 31, 7, tzinfo=UTC)
    # entity refs come from the declared entity columns
    keys = {ref.key() for ref in first.entities}
    assert keys == {"account:acc_001", "instrument:AAPL"}
    # provenance traces back to the file and row
    assert first.provenance is not None
    assert first.provenance.source_system == "oms_csv"
    assert first.provenance.record_hash
    assert str(path) in (first.provenance.notes or "")
    assert "#row2" in (first.provenance.notes or "")


async def test_iso_timestamps_without_format(tmp_path):
    body = "id,amount,at\np1,10.5,2024-05-02T10:00:00Z\n"
    path = write_csv(tmp_path, body, "payments.csv")
    mapping = ColumnMapping(
        kind="payment.received",
        source="pay_csv",
        timestamp_column="at",
    )
    envelopes = await collect(CsvBatchConnector(path, mapping))
    assert envelopes[0].occurred_at == datetime(2024, 5, 2, 10, 0, tzinfo=UTC)


async def test_kind_column_overrides_default(tmp_path):
    body = "event,ref\ntrade.executed,r1\n,r2\npayment.received,r3\n"
    path = write_csv(tmp_path, body, "mixed.csv")
    mapping = ColumnMapping(kind="generic.event", source="mixed_csv", kind_column="event")
    envelopes = await collect(CsvBatchConnector(path, mapping))
    assert [e.kind for e in envelopes] == [
        "trade.executed",
        "generic.event",  # empty cell falls back to the mapping default
        "payment.received",
    ]


async def test_explicit_payload_columns_subset(tmp_path):
    path = write_csv(tmp_path, TRADES_CSV)
    mapping = trades_mapping().model_copy(update={"payload_columns": ["side", "quantity"]})
    envelopes = await collect(CsvBatchConnector(path, mapping))
    assert set(envelopes[0].payload) == {"side", "quantity"}


async def test_missing_declared_payload_column_is_strict_error(tmp_path):
    path = write_csv(tmp_path, TRADES_CSV)
    mapping = trades_mapping().model_copy(update={"payload_columns": ["nonexistent"]})
    with pytest.raises(ConnectorError, match="missing payload column"):
        await collect(CsvBatchConnector(path, mapping))


async def test_bad_timestamp_strict_raises_with_row_number(tmp_path):
    body = "id,at\nx1,not-a-date\n"
    path = write_csv(tmp_path, body, "bad.csv")
    mapping = ColumnMapping(kind="k.v", source="s", timestamp_column="at")
    with pytest.raises(ConnectorError, match="row 2"):
        await collect(CsvBatchConnector(path, mapping))


async def test_bad_rows_skipped_when_not_strict(tmp_path):
    body = "id,at\nx1,2024-01-01T00:00:00\nx2,not-a-date\nx3,2024-01-03T00:00:00\n"
    path = write_csv(tmp_path, body, "mixed_bad.csv")
    mapping = ColumnMapping(kind="k.v", source="s", timestamp_column="at")
    connector = CsvBatchConnector(path, mapping, strict=False)
    envelopes = await collect(connector)
    assert [e.payload["id"] for e in envelopes] == ["x1", "x3"]
    assert connector.rows_skipped == 1


async def test_missing_file_raises(tmp_path):
    connector = CsvBatchConnector(tmp_path / "nope.csv", trades_mapping())
    with pytest.raises(ConnectorError, match="not found"):
        await collect(connector)


async def test_episode_id_is_stamped(tmp_path):
    path = write_csv(tmp_path, TRADES_CSV)
    mapping = trades_mapping().model_copy(update={"episode_id": "ep_batch_1"})
    envelopes = await collect(CsvBatchConnector(path, mapping))
    assert all(e.episode_id == "ep_batch_1" for e in envelopes)


async def test_pump_csv_into_recording_ingestor(tmp_path, recorder_factory):
    path = write_csv(tmp_path, TRADES_CSV)
    connector = CsvBatchConnector(path, trades_mapping())
    recorder = recorder_factory()
    summary = await pump(connector, recorder)
    assert summary["ingested"] == 3
    assert len(recorder.envelopes) == 3


def test_coerce_value_rules():
    assert coerce_value(None) is None
    assert coerce_value("") is None
    assert coerce_value("   ") is None
    assert coerce_value("true") is True
    assert coerce_value("FALSE") is False
    assert coerce_value("42") == 42
    assert isinstance(coerce_value("42"), int)
    assert coerce_value("-3.5") == pytest.approx(-3.5)
    assert coerce_value("1e3") == pytest.approx(1000.0)
    assert coerce_value("GB00B03MLX29") == "GB00B03MLX29"
    assert coerce_value("  spaced  ") == "spaced"


def test_row_to_envelope_no_coercion():
    mapping = ColumnMapping(kind="k.v", source="s", coerce_numeric=False)
    envelope = row_to_envelope(
        mapping, {"qty": "100", "flag": "true"}, row_number=2, origin="x.csv"
    )
    assert envelope.payload == {"qty": "100", "flag": "true"}


def test_empty_entity_cells_are_dropped():
    mapping = ColumnMapping(
        kind="k.v", source="s", entity_columns={"account": "acct", "customer": "cust"}
    )
    envelope = row_to_envelope(
        mapping, {"acct": "a1", "cust": ""}, row_number=2, origin="x.csv"
    )
    assert [ref.key() for ref in envelope.entities] == ["account:a1"]
