"""Smoke tests for the FinTwinOS foundation: types, audit, registry, policy, LLM stub."""

from __future__ import annotations

import pytest

from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings
from fintwinos.core.types import (
    ApprovalToken,
    Decision,
    EventEnvelope,
    RiskTier,
    Scenario,
    SideEffectClass,
    ToolBand,
)
from fintwinos.models.llm_routing.client import LLMClient
from fintwinos.models.llm_routing.router import ModelRouter, TaskClass
from fintwinos.policy.gates import PolicyGate, PolicyRule, RuleEffect
from fintwinos.tools.envelope import mcp_call, parse_jsonrpc_request
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec

OFFLINE_SETTINGS = Settings(offline=True, openai_api_key=None)


def make_spec(**overrides):
    base = dict(
        name="observe_positions",
        description="Read positions from the twin.",
        input_schema={
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
            "additionalProperties": False,
        },
        band=ToolBand.observe,
    )
    base.update(overrides)
    return ToolSpec(**base)


# --- types ----------------------------------------------------------------------


def test_event_envelope_hash_is_stable():
    e1 = EventEnvelope(kind="trade.booked", source="oms", payload={"qty": 5})
    e2 = EventEnvelope(kind="trade.booked", source="oms", payload={"qty": 5})
    assert e1.content_hash() == e2.content_hash()
    assert e1.event_id != e2.event_id


def test_decision_rejects_bad_action_type():
    with pytest.raises(ValueError):
        Decision(objective="x", action_type="auto_yolo")


def test_scenario_defaults():
    s = Scenario(name="usd liquidity squeeze")
    assert s.kind == "stress" and s.assumptions_version == "v1"


# --- audit ----------------------------------------------------------------------


def test_audit_chain_appends_and_verifies(tmp_path):
    trail = AuditTrail(path=tmp_path / "trail.jsonl")
    trail.append("tester", "unit.test", {"i": 1})
    trail.append("tester", "unit.test", {"i": 2})
    assert len(trail) == 2 and trail.verify()
    reloaded = AuditTrail.load(tmp_path / "trail.jsonl")
    assert len(reloaded) == 2 and reloaded.verify()


def test_audit_detects_tampering():
    trail = AuditTrail()
    trail.append("tester", "a", {"v": 1})
    trail.append("tester", "b", {"v": 2})
    trail._records[0].payload["v"] = 999  # tamper
    assert not trail.verify()


# --- tool spec invariants ----------------------------------------------------------


def test_observe_tool_must_be_side_effect_free():
    with pytest.raises(ValueError):
        make_spec(side_effect=SideEffectClass.reversible)


def test_execute_tool_must_require_approval():
    with pytest.raises(ValueError):
        ToolSpec(
            name="execute_transfer",
            description="x",
            input_schema={"type": "object"},
            band=ToolBand.execute,
            side_effect=SideEffectClass.reversible,
            requires_human_approval=False,
        )


def test_band_prefix_enforced():
    with pytest.raises(ValueError):
        make_spec(name="positions_observe")


def test_public_dict_carries_governance_fields():
    pub = make_spec().to_public_dict()
    for key in ("x_risk_tier", "x_side_effect", "x_requires_human_approval",
                "x_provenance_required", "x_band"):
        assert key in pub


# --- registry dispatch ----------------------------------------------------------------


@pytest.fixture
def registry():
    reg = ToolRegistry(policy_gate=PolicyGate(), settings=OFFLINE_SETTINGS)
    reg.register(make_spec(), lambda args, ctx: {"positions": [], "account": args["account_id"]})
    reg.register(
        ToolSpec(
            name="execute_post_transfer",
            description="Post a transfer (stub).",
            input_schema={"type": "object", "properties": {"amount": {"type": "number"}},
                          "required": ["amount"]},
            band=ToolBand.execute,
            risk_tier=RiskTier.high,
            side_effect=SideEffectClass.reversible,
            requires_human_approval=True,
        ),
        lambda args, ctx: {"posted": args["amount"]},
    )
    return reg


async def test_call_validates_and_dispatches(registry):
    result = await registry.call("observe_positions", {"account_id": "acc_1"})
    assert result.ok and result.data["account"] == "acc_1"
    assert result.audit_ref is not None


async def test_call_rejects_bad_arguments(registry):
    result = await registry.call("observe_positions", {"nope": 1})
    assert not result.ok and "failed validation" in (result.error or "")


async def test_unknown_tool_is_soft_error(registry):
    result = await registry.call("observe_missing", {})
    assert not result.ok


async def test_execute_blocked_without_global_enable(registry):
    result = await registry.call("execute_post_transfer", {"amount": 10.0})
    assert not result.ok and result.requires_approval


async def test_execute_blocked_without_approval_even_when_enabled():
    settings = Settings(offline=True, execute_tools_enabled=True)
    reg = ToolRegistry(policy_gate=PolicyGate(), settings=settings)
    reg.register(
        ToolSpec(
            name="execute_post_transfer",
            description="x",
            input_schema={"type": "object"},
            band=ToolBand.execute,
            side_effect=SideEffectClass.reversible,
            requires_human_approval=True,
        ),
        lambda args, ctx: {"posted": True},
    )
    result = await reg.call("execute_post_transfer", {})
    assert not result.ok and result.requires_approval


async def test_execute_allowed_with_approval_and_allow_rule():
    settings = Settings(offline=True, execute_tools_enabled=True)
    gate = PolicyGate()
    gate.add_rule(
        PolicyRule(name="allow-transfers", effect=RuleEffect.allow,
                   bands=[ToolBand.execute], tools=["execute_post_transfer"]),
        prepend=True,
    )
    reg = ToolRegistry(policy_gate=gate, settings=settings)
    reg.register(
        ToolSpec(
            name="execute_post_transfer",
            description="x",
            input_schema={"type": "object"},
            band=ToolBand.execute,
            side_effect=SideEffectClass.reversible,
            requires_human_approval=True,
        ),
        lambda args, ctx: {"posted": True},
    )
    token = ApprovalToken(subject="execute_post_transfer", granted_by="ops_lead")
    result = await reg.call("execute_post_transfer", {}, CallContext(approval=token))
    assert result.ok and result.data == {"posted": True}


async def test_dry_run_execute_is_simulated(registry):
    result = await registry.call(
        "execute_post_transfer", {"amount": 5.0}, CallContext(dry_run=True)
    )
    assert result.ok and result.data["dry_run"] is True


async def test_idempotency_replay(registry):
    ctx = CallContext(idempotency_key="k1")
    r1 = await registry.call("observe_positions", {"account_id": "a"}, ctx)
    r2 = await registry.call("observe_positions", {"account_id": "a"}, ctx)
    assert r1.audit_ref == r2.audit_ref


# --- policy gate ----------------------------------------------------------------------


def test_execute_default_deny():
    gate = PolicyGate()
    spec = ToolSpec(
        name="execute_anything",
        description="x",
        input_schema={"type": "object"},
        band=ToolBand.execute,
        side_effect=SideEffectClass.irreversible,
        requires_human_approval=True,
    )
    verdict = gate.check_tool_call(spec, {})
    assert not verdict.allowed


def test_decision_gate_flags_execution_for_review():
    verdict = PolicyGate().check_decision(Decision(objective="x", action_type="execute"))
    assert verdict.allowed and verdict.requires_human_review


# --- envelope ----------------------------------------------------------------------


def test_mcp_call_shape_matches_spec():
    env = mcp_call("simulate_liquidity_stress", {"horizon_days": 5}, "req-1")
    assert env == {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "tools/call",
        "params": {"name": "simulate_liquidity_stress",
                   "arguments": {"horizon_days": 5}},
    }


def test_parse_jsonrpc_rejects_bad_version():
    out = parse_jsonrpc_request({"jsonrpc": "1.0", "id": 1, "method": "tools/list"})
    assert isinstance(out, dict) and "error" in out


# --- llm routing ----------------------------------------------------------------------


def test_router_maps_task_classes():
    router = ModelRouter(OFFLINE_SETTINGS)
    assert router.model_for(TaskClass.planning) == OFFLINE_SETTINGS.llm_model_primary
    assert router.model_for(TaskClass.drafting) == OFFLINE_SETTINGS.llm_model_fast
    assert router.model_for(TaskClass.classification) == OFFLINE_SETTINGS.llm_model_cheap


async def test_offline_llm_is_deterministic():
    client = LLMClient(settings=OFFLINE_SETTINGS)
    assert client.offline
    msgs = [{"role": "user", "content": "what is our USD liquidity position?"}]
    r1 = await client.complete(msgs)
    r2 = await client.complete(msgs)
    assert r1.offline and r1.text == r2.text


async def test_offline_llm_json_mode_parses():
    client = LLMClient(settings=OFFLINE_SETTINGS)
    response = await client.complete(
        [{"role": "user", "content": "plan"}],
        json_schema={"title": "plan", "type": "object"},
    )
    data = response.json_data()
    assert data is not None and data["offline"] is True
