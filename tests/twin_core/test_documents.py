"""TfidfDocumentStore: tokenisation, incremental indexing, search relevance."""

from __future__ import annotations

import pytest

from fintwinos.twin_core.documents import TfidfDocumentStore, tokenise


@pytest.fixture()
def store() -> TfidfDocumentStore:
    s = TfidfDocumentStore()
    s.add(
        "pol_aml",
        "Anti money laundering escalation: structuring alerts below the reporting "
        "threshold must be escalated to the MLRO for suspicious activity review.",
        metadata={"type": "policy"},
    )
    s.add(
        "pol_liquidity",
        "Liquidity contingency funding: treasury monitors the cash ladder and the "
        "liquid asset buffer under intraday stress.",
        metadata={"type": "policy"},
    )
    s.add(
        "pol_complaints",
        "Complaints handling: every customer complaint is acknowledged within three "
        "business days and investigated for redress.",
        metadata={"type": "policy"},
    )
    s.add(
        "pol_limits",
        "Trading limits: each account carries a daily gross notional limit and "
        "breaches are reported to market risk.",
        metadata={"type": "policy"},
    )
    return s


def test_tokenise_lowercase_alnum():
    assert tokenise("Hello, WORLD-42! (test)") == ["hello", "world", "42", "test"]


def test_get_and_count(store):
    assert store.count() == 4
    doc = store.get("pol_aml")
    assert doc["doc_id"] == "pol_aml"
    assert "laundering" in doc["text"]
    assert doc["metadata"] == {"type": "policy"}
    assert store.get("missing") is None


def test_search_relevance_ranking(store):
    results = store.search("anti money laundering escalation suspicious", k=3)
    assert results[0][0] == "pol_aml"
    results = store.search("intraday liquidity stress cash buffer", k=3)
    assert results[0][0] == "pol_liquidity"
    results = store.search("customer complaint redress", k=3)
    assert results[0][0] == "pol_complaints"


def test_search_scores_are_descending_cosines(store):
    results = store.search("liquidity cash treasury", k=4)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < score <= 1.0 + 1e-9 for score in scores)


def test_search_unknown_terms_and_empty_store():
    empty = TfidfDocumentStore()
    assert empty.search("anything") == []
    store = TfidfDocumentStore()
    store.add("d1", "alpha beta gamma")
    assert store.search("zzz qqq") == []
    assert store.search("", k=3) == []


def test_incremental_add_reindexes(store):
    assert store.search("model validation calibration", k=2) == []
    store.add(
        "pol_models",
        "Model governance: models are validated for calibration and conceptual "
        "soundness before use.",
    )
    results = store.search("model validation calibration", k=2)
    assert results and results[0][0] == "pol_models"


def test_replacing_document_updates_index(store):
    store.add("pol_aml", "completely unrelated gardening manual about roses")
    assert store.count() == 4  # replaced, not duplicated
    results = store.search("money laundering escalation", k=4)
    assert all(doc_id != "pol_aml" for doc_id, _ in results)
    roses = store.search("gardening roses", k=1)
    assert roses[0][0] == "pol_aml"


def test_search_determinism(store):
    first = store.search("limit breach market risk", k=4)
    second = store.search("limit breach market risk", k=4)
    assert first == second


def test_demo_policy_search(demo_runtime):
    results = demo_runtime.documents.search(
        "anti money laundering escalation suspicious activity report", k=3
    )
    assert results[0][0] == "pol_aml_escalation"
    results = demo_runtime.documents.search("liquidity stress cash ladder buffer", k=3)
    assert results[0][0] == "pol_liquidity_contingency"
