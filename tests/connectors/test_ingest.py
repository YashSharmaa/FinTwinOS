"""Operator ingestion helpers: CSV/CDC dispatch, EDGAR routing, summaries."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fintwinos.connectors.base import ConnectorError
from fintwinos.connectors.ingest import ingest_edgar, ingest_path
from fintwinos.twin_core.runtime import build_runtime

FIXTURES = Path(__file__).parent / "fixtures"
APPLE_CIK10 = "0000320193"


def _runtime():
    return build_runtime(with_demo_data=False)


def test_ingest_csv_routes_accounts(tmp_path: Path):
    csv = tmp_path / "accounts.csv"
    csv.write_text(
        "account_id,customer_id,balance,currency,occurred_at\n"
        "acc_900,cus_900,15000.5,USD,2026-02-01T09:00:00Z\n"
        "acc_901,cus_901,250.0,EUR,2026-02-01T09:05:00Z\n",
        encoding="utf-8",
    )
    rt = _runtime()
    summary = ingest_path(csv, rt, kind="account.snapshot", entity_type="account")
    assert summary["format"] == "csv"
    assert summary["emitted"] == 2
    assert summary["routed"] == 2
    assert summary["failed"] == 0
    assert rt.graph.stats()["entities"] >= 2


def test_ingest_cdc_routes_changes(tmp_path: Path):
    jsonl = tmp_path / "changes.jsonl"
    jsonl.write_text(
        '{"table":"account","op":"update","key":"acc_900",'
        '"after":{"account_id":"acc_900","balance":14000.0},"ts":"2026-02-02T10:00:00Z"}\n',
        encoding="utf-8",
    )
    rt = _runtime()
    summary = ingest_path(jsonl, rt, entity_type="account")
    assert summary["format"] == "jsonl-cdc"
    assert summary["emitted"] == 1
    assert summary["routed"] == 1


def test_ingest_unsupported_format(tmp_path: Path):
    bad = tmp_path / "data.parquet"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ConnectorError, match="unsupported ingest format"):
        ingest_path(bad, _runtime())


def test_ingest_missing_file():
    with pytest.raises(ConnectorError, match="not found"):
        ingest_path("/nonexistent/path.csv", _runtime())


def test_ingest_edgar_routes_filings_to_document_store(tmp_path: Path, monkeypatch):
    # Prime the EDGAR cache so the connector resolves offline.
    monkeypatch.setenv("FINTWIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FINTWIN_OFFLINE", "1")
    from fintwinos.core.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    cache = tmp_path / "edgar"
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        FIXTURES / f"edgar_submissions_CIK{APPLE_CIK10}.json",
        cache / f"submissions_CIK{APPLE_CIK10}.json",
    )
    rt = build_runtime(with_demo_data=False)
    summary = ingest_edgar("320193", rt, limit=3)
    assert summary["format"] == "edgar"
    assert summary["routed"] == summary["emitted"] >= 1
    assert summary["failed"] == 0
    # The filings actually landed in the document store.
    assert rt.documents.count() >= 1
    get_settings.cache_clear()  # type: ignore[attr-defined]
