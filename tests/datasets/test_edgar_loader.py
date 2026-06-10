"""Tests for the EDGAR cache loader and the bundled offline sample corpus."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from fintwinos.core.types import PolicyDoc
from fintwinos.datasets.edgar_loader import load_cached_filings, sample_filing_corpus


class TestSampleCorpus:
    def test_size_and_uniqueness(self):
        docs = sample_filing_corpus()
        assert 6 <= len(docs) <= 8
        ids = [d["policy_id"] for d in docs]
        assert len(ids) == len(set(ids))

    def test_docs_validate_as_policydoc(self):
        for doc in sample_filing_corpus():
            parsed = PolicyDoc.model_validate(doc)
            assert parsed.policy_id.startswith("edgar_")
            assert len(parsed.body) > 400, "excerpts must be substantial"
            assert "edgar" in parsed.tags
            assert isinstance(parsed.effective_date, datetime)
            assert parsed.effective_date.tzinfo is not None

    def test_metadata_block(self):
        for doc in sample_filing_corpus():
            meta = doc["metadata"]
            assert meta["source"] == "sec-edgar"
            assert meta["company"]
            assert meta["form_type"] in {"10-K", "10-Q", "20-F"}
            # Fictional corpus must never point at real SEC URLs.
            assert "sec.gov" not in (meta["url"] or "")

    def test_deterministic_and_isolated(self):
        a, b = sample_filing_corpus(), sample_filing_corpus()
        assert a == b
        a[0]["title"] = "mutated"
        assert sample_filing_corpus()[0]["title"] != "mutated"

    def test_sorted_by_filing_date(self):
        dates = [d["effective_date"] for d in sample_filing_corpus()]
        assert dates == sorted(dates)


class TestLoadCachedFilings:
    def test_missing_dir_raises_helpfully(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="EDGAR cache directory"):
            load_cached_filings(tmp_path / "nope")

    def test_loads_per_filing_files(self, tmp_path):
        (tmp_path / "a.json").write_text(
            json.dumps(
                {
                    "accession_no": "0001111111-24-000001",
                    "cik": "0001111111",
                    "company": "Testco Holdings",
                    "form_type": "10-K",
                    "filed_at": "2024-01-31",
                    "text": "Item 1A. Risk Factors. " + "Liquidity risk remains elevated. " * 10,
                    "url": "https://example.invalid/a",
                }
            ),
            encoding="utf-8",
        )
        docs = load_cached_filings(tmp_path)
        assert len(docs) == 1
        doc = docs[0]
        assert doc["policy_id"] == "edgar_000111111124000001"
        assert doc["title"] == "Testco Holdings 10-K (2024-01-31)"
        assert "10-k" in doc["tags"]
        assert doc["metadata"]["cik"] == "0001111111"
        PolicyDoc.model_validate(doc)

    def test_key_aliases_and_sections(self, tmp_path):
        (tmp_path / "b.json").write_text(
            json.dumps(
                {
                    "accession_number": "0002222222-23-000009",
                    "company_name": "Aliased Corp",
                    "form": "10-Q",
                    "filing_date": "2023-08-04",
                    "sections": {"Risk Factors": "Rates.", "MD&A": "Margins."},
                }
            ),
            encoding="utf-8",
        )
        docs = load_cached_filings(tmp_path)
        assert len(docs) == 1
        assert docs[0]["metadata"]["company"] == "Aliased Corp"
        assert "## Risk Factors" in docs[0]["body"]
        assert "## MD&A" in docs[0]["body"]

    def test_index_envelope_and_lists(self, tmp_path):
        filings = [
            {"accession_no": f"000333333{i}-24-00000{i}", "company": f"C{i}",
             "form_type": "8-K", "filed_at": f"2024-03-0{i}", "text": f"Event {i} disclosure."}
            for i in (1, 2)
        ]
        (tmp_path / "index.json").write_text(
            json.dumps({"filings": filings}), encoding="utf-8"
        )
        docs = load_cached_filings(tmp_path)
        assert [d["metadata"]["company"] for d in docs] == ["C1", "C2"]

    def test_skips_junk_files(self, tmp_path):
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
        (tmp_path / "empty.json").write_text(json.dumps({"no": "text here"}), encoding="utf-8")
        assert load_cached_filings(tmp_path) == []

    def test_round_trip_with_sample_corpus_format(self, tmp_path):
        """The sample corpus records are valid cache records: write + reload them."""
        from fintwinos.datasets.edgar_loader import _SAMPLE_FILINGS

        for i, record in enumerate(_SAMPLE_FILINGS):
            (tmp_path / f"filing_{i}.json").write_text(json.dumps(record), encoding="utf-8")
        loaded = load_cached_filings(tmp_path)
        expected = sample_filing_corpus()
        assert [d["policy_id"] for d in loaded] == [d["policy_id"] for d in expected]
        assert [d["body"] for d in loaded] == [d["body"] for d in expected]
