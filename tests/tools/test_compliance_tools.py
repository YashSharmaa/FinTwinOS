"""Compliance domain tests: alerts, cases, egonets, threshold simulation,
deterministic narratives and the approved case-closure path."""

from __future__ import annotations

import json

from conftest import approval_for

from fintwinos.core.types import EntityRef
from fintwinos.tools.registry import CallContext

CTX = CallContext(caller="test-compliance")


# -- observe_alerts ----------------------------------------------------------------


async def test_observe_alerts_sorted_by_score(registry):
    result = await registry.call("observe_alerts", {}, CTX)
    assert result.ok, result.error
    alerts = result.data["alerts"]
    assert [a["alert_id"] for a in alerts] == ["ALR_1", "ALR_2", "ALR_3"]
    scores = [a["score"] for a in alerts]
    assert scores == sorted(scores, reverse=True)


async def test_observe_alerts_filters(registry):
    by_status = await registry.call("observe_alerts", {"status": "new"}, CTX)
    assert [a["alert_id"] for a in by_status.data["alerts"]] == ["ALR_1"]

    by_severity = await registry.call("observe_alerts", {"min_severity": "medium"}, CTX)
    assert [a["alert_id"] for a in by_severity.data["alerts"]] == ["ALR_1", "ALR_2"]

    by_kind = await registry.call("observe_alerts", {"kind": "velocity"}, CTX)
    assert [a["alert_id"] for a in by_kind.data["alerts"]] == ["ALR_2"]


# -- observe_case ------------------------------------------------------------------


async def test_observe_case_returns_links_and_related_alerts(registry):
    result = await registry.call("observe_case", {"case_id": "CASE_7"}, CTX)
    assert result.ok, result.error
    data = result.data
    assert data["case"]["kind"] == "aml_alert"
    linked_keys = {entry["key"] for entry in data["linked_entities"]}
    assert {"customer:CUST_2", "account:ACC_TRD_2"} <= linked_keys
    related_ids = {a["alert_id"] for a in data["related_alerts"]}
    assert related_ids == {"ALR_1", "ALR_2"}  # ALR_3 touches CUST_1 only


async def test_observe_case_not_found(registry):
    result = await registry.call("observe_case", {"case_id": "CASE_NOPE"}, CTX)
    assert not result.ok
    assert "not found" in result.error


# -- observe_entity_graph -------------------------------------------------------------


async def test_observe_entity_graph_egonet_depth_1(registry):
    result = await registry.call(
        "observe_entity_graph",
        {"entity_type": "customer", "entity_id": "CUST_2", "depth": 1},
        CTX,
    )
    assert result.ok, result.error
    keys = {node["key"] for node in result.data["nodes"]}
    assert "customer:CUST_2" in keys
    assert {"account:ACC_TRD_2", "counterparty:CP_9", "case:CASE_7"} <= keys
    assert result.data["edge_count"] >= 3
    assert result.data["node_count"] == len(result.data["nodes"])


async def test_observe_entity_graph_depth_2_expands(registry):
    depth1 = await registry.call(
        "observe_entity_graph",
        {"entity_type": "customer", "entity_id": "CUST_2", "depth": 1},
        CTX,
    )
    depth2 = await registry.call(
        "observe_entity_graph",
        {"entity_type": "customer", "entity_id": "CUST_2", "depth": 2},
        CTX,
    )
    assert depth2.data["node_count"] > depth1.data["node_count"]
    keys = {node["key"] for node in depth2.data["nodes"]}
    assert any(key.startswith("position:") for key in keys)  # account -> holds -> position


async def test_observe_entity_graph_kind_filter(registry):
    result = await registry.call(
        "observe_entity_graph",
        {"entity_type": "customer", "entity_id": "CUST_2", "depth": 1,
         "kind": "transacts_with"},
        CTX,
    )
    assert result.ok
    keys = {node["key"] for node in result.data["nodes"]}
    assert keys == {"customer:CUST_2", "counterparty:CP_9"}


async def test_observe_entity_graph_unknown_entity(registry):
    result = await registry.call(
        "observe_entity_graph", {"entity_type": "customer", "entity_id": "GHOST"}, CTX
    )
    assert not result.ok and "not found" in result.error


# -- simulate_alert_threshold -----------------------------------------------------------


async def test_simulate_alert_threshold_returns_trade_off(registry):
    result = await registry.call("simulate_alert_threshold", {"threshold": 0.7}, CTX)
    assert result.ok, result.error
    trade_off = result.data["trade_off"]
    assert 0.0 <= trade_off["precision"] <= 1.0
    assert 0.0 <= trade_off["recall"] <= 1.0
    assert trade_off["f1"] is not None
    assert trade_off["precision_ci"][0] <= trade_off["precision_ci"][1]
    assert result.data["calibration"]


async def test_simulate_alert_threshold_deterministic_and_threshold_sensitive(registry):
    first = await registry.call("simulate_alert_threshold", {"threshold": 0.7}, CTX)
    again = await registry.call("simulate_alert_threshold", {"threshold": 0.7}, CTX)
    other = await registry.call("simulate_alert_threshold", {"threshold": 0.4}, CTX)
    assert first.data["metrics"] == again.data["metrics"]
    assert first.data["metrics"] != other.data["metrics"]  # different derived seed


# -- propose_case_narrative ----------------------------------------------------------------


async def test_propose_case_narrative_structure_and_determinism(registry):
    first = await registry.call("propose_case_narrative", {"case_id": "CASE_7"}, CTX)
    second = await registry.call("propose_case_narrative", {"case_id": "CASE_7"}, CTX)
    assert first.ok, first.error
    narrative = first.data["narrative"]
    for heading in (
        "1. SUMMARY", "2. SUBJECTS", "3. ACTIVITY OBSERVED", "4. NETWORK CONTEXT",
        "5. ASSESSMENT", "6. RECOMMENDED ACTION",
    ):
        assert heading in narrative
    assert "CASE_7" in narrative
    assert "structuring" in narrative
    assert first.data["signal_strength"] == "high"  # ALR_1 scores 0.86
    assert "Escalate" in first.data["sections"]["recommended_action"]
    assert first.data["narrative"] == second.data["narrative"]
    assert first.data["generated_by"] == "rule_based_template"


async def test_propose_case_narrative_unknown_case(registry):
    result = await registry.call("propose_case_narrative", {"case_id": "CASE_NOPE"}, CTX)
    assert not result.ok and "not found" in result.error


# -- execute_close_case -----------------------------------------------------------------


CLOSURE = {
    "case_id": "CASE_7",
    "disposition": "escalated",
    "closure_note": "Escalated to the financial intelligence unit after review.",
    "ticket_id": "TCK-3003",
}


async def test_execute_close_case_blocked_without_approval(exec_registry):
    result = await exec_registry.call("execute_close_case", CLOSURE, CallContext(caller="ops"))
    assert not result.ok and result.requires_approval is True


async def test_execute_close_case_approved_appends_ledger_and_closes_case(
    exec_registry, exec_settings, fake_runtime
):
    context = CallContext(
        caller="aml-ops", ticket_id="TCK-3003", approval=approval_for("execute_close_case")
    )
    result = await exec_registry.call("execute_close_case", CLOSURE, context)
    assert result.ok, result.error
    assert result.data["status"] == "closed"

    ledger = exec_settings.data_dir / "outbox" / "case_closures.jsonl"
    assert ledger.exists()
    lines = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["case_id"] == "CASE_7"
    assert lines[0]["disposition"] == "escalated"
    assert lines[0]["approved_by"] == "approver@bank.example"

    case = fake_runtime.graph.get_entity(EntityRef(entity_type="case", entity_id="CASE_7"))
    assert case["status"] == "closed"
    assert case["disposition"] == "escalated"
    assert exec_registry.audit.verify()
    assert "case.closed" in [r.action for r in exec_registry.audit.records()]


async def test_execute_close_case_unknown_case_fails_cleanly(exec_registry):
    context = CallContext(caller="ops", approval=approval_for("execute_close_case"))
    result = await exec_registry.call(
        "execute_close_case", dict(CLOSURE, case_id="CASE_NOPE"), context
    )
    assert not result.ok and "not found" in result.error
