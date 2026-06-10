"""Tests for the SEC EDGAR connector: cache-backed offline parsing, the fetch
gate, companyfacts extraction, and an opt-in real-network smoke test."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fintwinos.connectors.base import ConnectorError, pump
from fintwinos.connectors.edgar import (
    COMPANYFACTS_URL,
    SUBMISSIONS_URL,
    EdgarConnector,
    EdgarOfflineError,
    fact_series,
    iter_recent_filings,
    normalise_cik,
)
from fintwinos.core.config import Settings

APPLE_CIK = "320193"
APPLE_CIK10 = "0000320193"


def prime_submissions_cache(connector: EdgarConnector, fixtures_dir: Path) -> Path:
    target = connector.cache_path("submissions", APPLE_CIK)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixtures_dir / f"edgar_submissions_CIK{APPLE_CIK10}.json", target)
    return target


def prime_companyfacts_cache(connector: EdgarConnector, fixtures_dir: Path) -> Path:
    target = connector.cache_path("companyfacts", APPLE_CIK)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixtures_dir / f"edgar_companyfacts_CIK{APPLE_CIK10}.json", target)
    return target


async def collect(connector):
    return [envelope async for envelope in connector.stream()]


# -- CIK normalisation -------------------------------------------------------


def test_normalise_cik_accepts_common_spellings():
    assert normalise_cik(320193) == APPLE_CIK10
    assert normalise_cik("320193") == APPLE_CIK10
    assert normalise_cik("CIK0000320193") == APPLE_CIK10
    assert normalise_cik("cik320193") == APPLE_CIK10
    assert normalise_cik(" 0000320193 ") == APPLE_CIK10


def test_normalise_cik_rejects_garbage():
    with pytest.raises(ConnectorError, match="invalid CIK"):
        normalise_cik("AAPL")


# -- offline streaming from the bundled fixture --------------------------------


async def test_stream_parses_cached_fixture_fully_offline(offline_settings, fixtures_dir):
    connector = EdgarConnector(APPLE_CIK, settings=offline_settings)
    prime_submissions_cache(connector, fixtures_dir)

    envelopes = await collect(connector)

    assert len(envelopes) == 6
    assert all(e.kind == "filing.indexed" for e in envelopes)
    assert all(e.source == "sec_edgar" for e in envelopes)

    ten_q = envelopes[0]
    assert ten_q.payload["doc_id"] == f"edgar:{APPLE_CIK10}:0000320193-24-000069"
    assert ten_q.payload["title"] == "Apple Inc. 10-Q 2024-05-03"
    assert "Apple Inc. filed form 10-Q" in ten_q.payload["text"]
    assert (
        ten_q.payload["url"]
        == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000069/aapl-20240330.htm"
    )
    meta = ten_q.payload["metadata"]
    assert meta["form"] == "10-Q"
    assert meta["cik"] == APPLE_CIK10
    assert meta["report_date"] == "2024-03-30"
    assert meta["tickers"] == ["AAPL"]
    assert ten_q.occurred_at == datetime(2024, 5, 3, tzinfo=UTC)

    # entities: the legal entity plus its listed ticker
    keys = [ref.key() for ref in ten_q.entities]
    assert f"legal_entity:CIK{APPLE_CIK10}" in keys
    assert "instrument:AAPL" in keys

    # provenance carries the public-domain licence and a traceable note
    assert ten_q.provenance is not None
    assert "public domain" in (ten_q.provenance.licence or "")
    assert ten_q.provenance.record_hash

    # an 8-K keeps its items list in metadata
    eight_k = next(e for e in envelopes if e.payload["metadata"]["form"] == "8-K")
    assert eight_k.payload["metadata"]["items"] == "2.02,9.01"


async def test_forms_filter_and_limit(offline_settings, fixtures_dir):
    connector = EdgarConnector(
        APPLE_CIK, forms=["10-K", "10-q"], settings=offline_settings
    )
    prime_submissions_cache(connector, fixtures_dir)
    envelopes = await collect(connector)
    assert [e.payload["metadata"]["form"] for e in envelopes] == ["10-Q", "10-K", "10-Q"]

    limited = EdgarConnector(
        APPLE_CIK, forms=["10-K", "10-Q"], limit_per_cik=2, settings=offline_settings
    )
    assert len(await collect(limited)) == 2


async def test_pump_edgar_into_recording_ingestor(offline_settings, fixtures_dir, recorder_factory):
    connector = EdgarConnector(APPLE_CIK, settings=offline_settings)
    prime_submissions_cache(connector, fixtures_dir)
    recorder = recorder_factory()
    summary = await pump(connector, recorder)
    assert summary["ingested"] == 6
    assert {e.kind for e in recorder.envelopes} == {"filing.indexed"}


# -- the fetch gate ------------------------------------------------------------


async def test_fetch_offline_without_cache_raises_clear_error(offline_settings):
    connector = EdgarConnector(APPLE_CIK, settings=offline_settings)
    with pytest.raises(EdgarOfflineError) as excinfo:
        await connector.submissions(APPLE_CIK)
    message = str(excinfo.value)
    assert "offline" in message
    assert SUBMISSIONS_URL.format(cik10=APPLE_CIK10) in message
    assert str(connector.cache_path("submissions", APPLE_CIK)) in message


async def test_stream_offline_without_cache_raises(offline_settings):
    connector = EdgarConnector(APPLE_CIK, settings=offline_settings)
    with pytest.raises(EdgarOfflineError):
        await collect(connector)


async def test_company_facts_offline_without_cache_raises(offline_settings):
    connector = EdgarConnector(APPLE_CIK, settings=offline_settings)
    with pytest.raises(EdgarOfflineError) as excinfo:
        await connector.company_facts(APPLE_CIK)
    assert COMPANYFACTS_URL.format(cik10=APPLE_CIK10) in str(excinfo.value)


async def test_corrupt_cache_is_reported(offline_settings):
    connector = EdgarConnector(APPLE_CIK, settings=offline_settings)
    path = connector.cache_path("submissions", APPLE_CIK)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConnectorError, match="corrupt"):
        await connector.submissions(APPLE_CIK)


def test_user_agent_resolution(offline_settings, monkeypatch):
    monkeypatch.delenv("FINTWIN_EDGAR_USER_AGENT", raising=False)
    connector = EdgarConnector(APPLE_CIK, settings=offline_settings)
    assert connector.user_agent == "FinTwinOS research contact@example.com"

    monkeypatch.setenv("FINTWIN_EDGAR_USER_AGENT", "ACME Risk Lab ops@acme.example")
    from_env = EdgarConnector(APPLE_CIK, settings=offline_settings)
    assert from_env.user_agent == "ACME Risk Lab ops@acme.example"

    explicit = EdgarConnector(
        APPLE_CIK, settings=offline_settings, user_agent="Explicit UA x@y.z"
    )
    assert explicit.user_agent == "Explicit UA x@y.z"


# -- companyfacts / XBRL ---------------------------------------------------------


async def test_company_facts_and_fact_series_from_cache(offline_settings, fixtures_dir):
    connector = EdgarConnector(APPLE_CIK, settings=offline_settings)
    prime_companyfacts_cache(connector, fixtures_dir)

    facts = await connector.company_facts(APPLE_CIK)
    assert facts["entityName"] == "Apple Inc."

    series = fact_series(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
    # restatements deduplicated by end date; sorted ascending
    assert series == [
        ("2022-09-24", 394328000000.0),
        ("2023-09-30", 383285000000.0),
        ("2024-03-30", 90753000000.0),
    ]

    shares = fact_series(
        facts, "EntityCommonStockSharesOutstanding", taxonomy="dei", unit="shares"
    )
    assert shares[0] == ("2023-10-20", 15552752000.0)

    assert fact_series(facts, "NoSuchTag") == []
    assert fact_series(facts, "RevenueFromContractWithCustomerExcludingAssessedTax", unit="EUR") == []


# -- submissions parsing helpers ---------------------------------------------------


def test_iter_recent_filings_handles_short_and_missing_arrays():
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["a-1", "a-2"],
                "form": ["10-K"],  # shorter than accessionNumber
                # filingDate entirely absent
            }
        }
    }
    filings = list(iter_recent_filings(payload))
    assert len(filings) == 2
    assert filings[0]["form"] == "10-K"
    assert filings[1]["form"] is None
    assert filings[0]["filingDate"] is None
    assert list(iter_recent_filings({})) == []


# -- opt-in real network smoke test ------------------------------------------------


@pytest.mark.network
async def test_real_submissions_fetch_and_cache(tmp_path):
    """Hits data.sec.gov for real; run with FINTWIN_RUN_NETWORK_TESTS=1."""
    settings = Settings(offline=False, openai_api_key=None, data_dir=tmp_path / "data")
    connector = EdgarConnector(APPLE_CIK, settings=settings, limit_per_cik=3)
    envelopes = await collect(connector)
    assert len(envelopes) == 3
    assert connector.cache_path("submissions", APPLE_CIK).exists()

    # second pass must be served from cache even when forced offline
    offline = Settings(offline=True, openai_api_key=None, data_dir=tmp_path / "data")
    cached = EdgarConnector(APPLE_CIK, settings=offline, limit_per_cik=3)
    assert len(await collect(cached)) == 3
