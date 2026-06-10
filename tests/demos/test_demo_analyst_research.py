"""End-to-end offline tests for the analyst research demo."""

from __future__ import annotations

import io
from typing import Any

from rich.console import Console

from fintwinos.demos import analyst_research


def _run(seed: int = 7) -> dict[str, Any]:
    return analyst_research.run(seed=seed, console=Console(file=io.StringIO(), width=120))


def test_run_offline_end_to_end_returns_contracted_keys() -> None:
    result = _run()
    assert result["demo"] == "analyst_research"
    assert result["offline"] is True
    for key in ("corpus", "search", "brief", "critic", "audit", "warnings"):
        assert key in result


def test_corpus_loaded_into_document_store() -> None:
    corpus = _run()["corpus"]
    assert corpus["docs_loaded"] > 0
    assert corpus["docs_in_store"] >= corpus["docs_loaded"]
    assert corpus["source"]


def test_search_returns_scored_hits() -> None:
    search = _run()["search"]
    assert search["query"]
    assert len(search["hits"]) >= 1
    for hit in search["hits"]:
        assert isinstance(hit["doc_id"], str) and hit["doc_id"]
        assert isinstance(hit["score"], float)


def test_every_brief_claim_is_cited() -> None:
    result = _run()
    brief = result["brief"]
    assert len(brief) == len(analyst_research.RESEARCH_TOPICS)
    for claim in brief:
        assert claim["claim"].strip()
        assert len(claim["citations"]) >= 1, f"uncited claim for topic {claim['topic']}"
        for citation in claim["citations"]:
            assert isinstance(citation["doc_id"], str) and citation["doc_id"]
            assert isinstance(citation["score"], float)


def test_critic_passes_extractive_brief_offline() -> None:
    critic = _run()["critic"]
    assert critic["passed"] is True  # extractive claims are grounded by construction
    assert critic["coverage"] >= 0.4
    assert isinstance(critic["challenges"], list)
    assert not any(c["severity"] == "high" for c in critic["challenges"])
    assert critic["commentary"].strip()


def test_audit_chain_verifies_and_records_critique() -> None:
    audit = _run()["audit"]
    assert audit["verified"] is True
    actions = {row["action"] for row in audit["excerpt"]}
    assert "brief.critiqued" in actions


def test_determinism_same_seed_same_brief() -> None:
    first = _run(seed=33)
    second = _run(seed=33)
    assert first["brief"] == second["brief"]
    assert first["critic"]["challenges"] == second["critic"]["challenges"]
