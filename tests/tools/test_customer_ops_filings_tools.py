"""Customer-ops and filings domain tests: queue state, staffing simulation,
compliant response drafting, the outbox send path, and document lookups."""

from __future__ import annotations

import json

from conftest import approval_for

from fintwinos.tools.registry import CallContext

CTX = CallContext(caller="test-custops")


# -- observe_queue_state --------------------------------------------------------------


async def test_observe_queue_state_derives_load_and_sla_risk(registry):
    result = await registry.call("observe_queue_state", {}, CTX)
    assert result.ok, result.error
    queues = {q["queue_id"]: q for q in result.data["queues"]}
    assert set(queues) == {"Q_COMPLAINTS", "Q_KYC"}

    complaints = queues["Q_COMPLAINTS"]
    assert complaints["offered_load_erlangs"] == 4.2  # 18/hr * 14min / 60
    assert complaints["utilisation"] == 0.7
    assert complaints["estimated_wait_minutes"] == 98.0  # 42 * 14 / 6
    assert complaints["sla_at_risk"] is True

    kyc = queues["Q_KYC"]
    assert kyc["sla_at_risk"] is False
    assert result.data["queues_at_risk"] == ["Q_COMPLAINTS"]


async def test_observe_queue_state_filter_and_unknown(registry):
    one = await registry.call("observe_queue_state", {"queue_id": "Q_KYC"}, CTX)
    assert one.ok and one.data["count"] == 1

    missing = await registry.call("observe_queue_state", {"queue_id": "Q_NOPE"}, CTX)
    assert not missing.ok and "not found" in missing.error


# -- simulate_staffing_change -----------------------------------------------------------


async def test_simulate_staffing_change_returns_cis(registry):
    arguments = {"queue_id": "Q_COMPLAINTS", "delta_agents": 2}
    result = await registry.call("simulate_staffing_change", arguments, CTX)
    assert result.ok, result.error
    assert result.data["confidence"]
    for low, high in result.data["confidence"].values():
        assert low <= high
    again = await registry.call("simulate_staffing_change", arguments, CTX)
    assert result.data["metrics"] == again.data["metrics"]


async def test_simulate_staffing_change_unknown_queue(registry):
    result = await registry.call(
        "simulate_staffing_change", {"queue_id": "Q_NOPE", "delta_agents": 1}, CTX
    )
    assert not result.ok and "not found" in result.error


# -- propose_response_draft ----------------------------------------------------------------


async def test_propose_response_draft_is_compliant_and_deterministic(registry):
    arguments = {
        "customer_id": "CUST_1",
        "case_id": "CMP-88",
        "category": "complaint",
        "key_points": ["We have paused the disputed fee", "A specialist now owns your case"],
    }
    first = await registry.call("propose_response_draft", arguments, CTX)
    second = await registry.call("propose_response_draft", arguments, CTX)
    assert first.ok, first.error
    body = first.data["body"]
    assert body.startswith("Dear Alice Howell,")
    assert "- We have paused the disputed fee" in body
    assert "ombudsman" in body  # mandatory regulatory footer
    assert "does not constitute financial advice" in body
    assert "case CMP-88" in first.data["subject"]
    assert first.data["requires_human_review"] is True
    assert first.data == second.data


async def test_propose_response_draft_rejects_unknown_category(registry):
    result = await registry.call(
        "propose_response_draft", {"customer_id": "CUST_1", "category": "marketing"}, CTX
    )
    assert not result.ok and "failed validation" in result.error


# -- execute_send_response ----------------------------------------------------------------


SEND = {
    "customer_id": "CUST_1",
    "channel": "secure_message",
    "subject": "Update on your complaint (case CMP-88)",
    "body": "Dear Alice Howell, ... final response ...",
    "ticket_id": "TCK-4004",
}


async def test_execute_send_response_blocked_without_approval(exec_registry):
    result = await exec_registry.call("execute_send_response", SEND, CallContext(caller="ops"))
    assert not result.ok and result.requires_approval is True


async def test_execute_send_response_approved_writes_outbox(exec_registry, exec_settings):
    context = CallContext(
        caller="custops", ticket_id="TCK-4004", approval=approval_for("execute_send_response")
    )
    result = await exec_registry.call("execute_send_response", SEND, context)
    assert result.ok, result.error
    files = sorted((exec_settings.data_dir / "outbox").glob("response_*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["customer_id"] == "CUST_1"
    assert record["channel"] == "secure_message"
    assert record["body"] == SEND["body"]
    assert record["status"] == "queued"
    assert "outbox.response_queued" in [r.action for r in exec_registry.audit.records()]


# -- observe_filing_search --------------------------------------------------------------


async def test_observe_filing_search_ranks_and_snippets(registry):
    result = await registry.call("observe_filing_search", {"query": "liquidity coverage"}, CTX)
    assert result.ok, result.error
    assert result.data["count"] >= 1
    top = result.data["results"][0]
    assert top["doc_id"] == "filing_10k_acme"
    assert top["score"] > 0
    assert "liquidity" in top["snippet"].lower()
    assert top["metadata"]["doc_type"] == "10-K"


async def test_observe_filing_search_doc_type_filter(registry):
    result = await registry.call(
        "observe_filing_search", {"query": "Acme Bank", "doc_type": "Pillar3"}, CTX
    )
    assert result.ok
    assert result.data["results"]
    assert all(r["metadata"]["doc_type"] == "Pillar3" for r in result.data["results"])


# -- observe_policy_lookup --------------------------------------------------------------


async def test_observe_policy_lookup_by_tag(registry):
    result = await registry.call("observe_policy_lookup", {"tag": "aml"}, CTX)
    assert result.ok, result.error
    assert [r["doc_id"] for r in result.data["results"]] == ["policy_aml_001"]
    assert result.data["results"][0]["metadata"]["kind"] == "policy"


async def test_observe_policy_lookup_by_query_prefers_policy_docs(registry):
    result = await registry.call("observe_policy_lookup", {"query": "liquidity"}, CTX)
    assert result.ok
    doc_ids = [r["doc_id"] for r in result.data["results"]]
    assert "policy_liq_001" in doc_ids
    assert all(r["metadata"]["kind"] == "policy" for r in result.data["results"])


async def test_observe_policy_lookup_requires_query_or_tag(registry):
    result = await registry.call("observe_policy_lookup", {}, CTX)
    assert not result.ok and "failed validation" in result.error
