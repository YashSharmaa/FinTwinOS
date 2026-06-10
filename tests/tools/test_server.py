"""MCP-style JSON-RPC server tests: initialize, tools/list, tools/call, error shapes,
the -32001 approval-required envelope, dry runs and the approved execute path."""

from __future__ import annotations

import json

import pytest
from conftest import allow_execute_gate, approval_for, build_fake_runtime
from fastapi.testclient import TestClient

from fintwinos.tools.catalog import build_default_registry
from fintwinos.tools.server import APPROVAL_REQUIRED, TOOL_FAILED, create_app


def rpc(client: TestClient, method: str, params: dict | None = None, request_id="rq-1") -> dict:
    response = client.post(
        "/rpc",
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}},
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture
def client(registry) -> TestClient:
    """Server over the default (execute-disabled) registry."""
    return TestClient(create_app(registry))


@pytest.fixture
def exec_client(exec_registry) -> TestClient:
    """Server over the execute-enabled, policy-allowed registry."""
    return TestClient(create_app(exec_registry))


# -- plumbing ---------------------------------------------------------------------


def test_healthz(client, registry):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["tools"] == len(registry)
    assert body["audit_intact"] is True


def test_initialize(client):
    body = rpc(client, "initialize")
    assert body["jsonrpc"] == "2.0" and body["id"] == "rq-1"
    result = body["result"]
    assert result["serverInfo"]["name"] == "fintwinos-tools"
    assert result["protocolVersion"]
    assert result["capabilities"]["tools"] == {"listChanged": False}
    assert "Yash Sharma" in result["serverInfo"]["attribution"]


def test_parse_error_for_invalid_json(client):
    response = client.post(
        "/rpc", content="{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"]["code"] == -32700


def test_invalid_jsonrpc_version(client):
    response = client.post("/rpc", json={"jsonrpc": "1.0", "id": 1, "method": "tools/list"})
    assert response.json()["error"]["code"] == -32600


def test_unknown_method(client):
    body = rpc(client, "tools/destroy")
    assert body["error"]["code"] == -32601


# -- tools/list -----------------------------------------------------------------------


def test_tools_list_carries_governance_fields(client, registry):
    body = rpc(client, "tools/list")
    tools = body["result"]["tools"]
    assert len(tools) == len(registry) >= 18
    by_name = {tool["name"]: tool for tool in tools}
    stress = by_name["simulate_liquidity_stress"]
    assert stress["x_risk_tier"] == "high"
    assert stress["x_side_effect"] == "none"
    assert stress["x_requires_human_approval"] is False
    transfer = by_name["execute_post_transfer"]
    assert transfer["x_requires_human_approval"] is True
    assert transfer["x_side_effect"] == "reversible"
    for tool in tools:
        assert tool["input_schema"]["additionalProperties"] is False


# -- tools/call ----------------------------------------------------------------------


def test_tools_call_success(client):
    body = rpc(client, "tools/call", {"name": "observe_positions", "arguments": {}})
    result = body["result"]
    assert result["ok"] is True
    assert result["tool"] == "observe_positions"
    assert result["data"]["count"] == 5
    assert result["audit_ref"]


def test_tools_call_unknown_tool(client):
    body = rpc(client, "tools/call", {"name": "observe_nonexistent", "arguments": {}})
    assert body["error"]["code"] == -32602
    assert body["error"]["data"]["tool"] == "observe_nonexistent"


def test_tools_call_schema_violation(client):
    body = rpc(
        client,
        "tools/call",
        {"name": "simulate_liquidity_stress", "arguments": {"horizon_days": 5}},
    )
    assert body["error"]["code"] == -32602
    assert "failed validation" in body["error"]["message"]


def test_tools_call_missing_name(client):
    body = rpc(client, "tools/call", {"arguments": {}})
    assert body["error"]["code"] == -32602


def test_tools_call_handler_failure_is_tool_failed(client):
    body = rpc(client, "tools/call", {"name": "observe_case", "arguments": {"case_id": "NOPE"}})
    assert body["error"]["code"] == TOOL_FAILED
    assert "not found" in body["error"]["message"]


def test_tools_call_approval_required_error_envelope(client):
    body = rpc(
        client,
        "tools/call",
        {
            "name": "execute_post_transfer",
            "arguments": {
                "from_account": "ACC_TRD_2",
                "to_account": "ACC_TRD_1",
                "amount": 1000.0,
                "currency": "USD",
                "ticket_id": "TCK-1",
            },
        },
    )
    assert body["error"]["code"] == APPROVAL_REQUIRED == -32001
    assert body["error"]["data"]["requires_approval"] is True
    assert body["error"]["data"]["tool"] == "execute_post_transfer"


def test_tools_call_dry_run_via_context(client):
    body = rpc(
        client,
        "tools/call",
        {
            "name": "execute_post_transfer",
            "arguments": {
                "from_account": "ACC_TRD_2",
                "to_account": "ACC_TRD_1",
                "amount": 1000.0,
                "currency": "USD",
                "ticket_id": "TCK-1",
            },
            "context": {"caller": "ui", "dry_run": True},
        },
    )
    assert body["result"]["ok"] is True
    assert body["result"]["data"]["dry_run"] is True


def test_tools_call_approved_execute_via_server(exec_client, exec_settings):
    token = approval_for("execute_post_transfer")
    body = rpc(
        exec_client,
        "tools/call",
        {
            "name": "execute_post_transfer",
            "arguments": {
                "from_account": "ACC_TRD_2",
                "to_account": "ACC_TRD_1",
                "amount": 9_999.0,
                "currency": "USD",
                "ticket_id": "TCK-9",
            },
            "context": {
                "caller": "treasury-ui",
                "ticket_id": "TCK-9",
                "approval": json.loads(token.model_dump_json()),
            },
        },
    )
    assert "result" in body, body
    assert body["result"]["ok"] is True
    files = list((exec_settings.data_dir / "outbox").glob("transfer_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["amount"] == 9_999.0


def test_tools_call_rejects_malformed_approval(client):
    body = rpc(
        client,
        "tools/call",
        {
            "name": "observe_positions",
            "arguments": {},
            "context": {"approval": {"granted_by": 12345}},  # missing subject, bad types
        },
    )
    assert body["error"]["code"] == -32602
    assert "invalid call context" in body["error"]["message"]


def test_create_app_without_registry_requires_twin_core():
    """create_app(None) builds the full demo twin when siblings exist; otherwise raises."""
    try:
        import fintwinos.twin_core.runtime  # noqa: F401
        import fintwinos.twin_sim  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            create_app()
        return
    app = create_app()
    with TestClient(app) as live:
        assert live.get("/healthz").json()["tools"] >= 18


def test_server_and_registry_share_audit(fake_runtime, settings):
    registry = build_default_registry(
        fake_runtime, policy_gate=allow_execute_gate(), settings=settings
    )
    client = TestClient(create_app(registry))
    before = len(registry.audit)
    rpc(client, "tools/call", {"name": "observe_alerts", "arguments": {}})
    assert len(registry.audit) > before
    assert registry.audit is fake_runtime.audit


def test_fake_runtime_protocol_conformance():
    """The fixtures must satisfy the runtime Protocols the tools code relies on."""
    from fintwinos.core.interfaces import (
        DocumentStore,
        GraphStore,
        ReplayEngine,
        Simulator,
        TimeSeriesStore,
    )

    runtime = build_fake_runtime()
    assert isinstance(runtime.graph, GraphStore)
    assert isinstance(runtime.timeseries, TimeSeriesStore)
    assert isinstance(runtime.documents, DocumentStore)
    assert isinstance(runtime.replay, ReplayEngine)
    assert all(isinstance(sim, Simulator) for sim in runtime.simulators.values())
