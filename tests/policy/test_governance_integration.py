"""End-to-end governance test: packs + RBAC + approvals + break-glass + kill switch.

Drives one execute-band tool through a real ToolRegistry with the full
governance plane composed: a YAML treasury pack inside an RbacGate, tokens
minted by the maker-checker workflow, a break-glass wildcard token, and the
kill switch having the final word. Everything lands in one shared, verifiable
audit trail.
"""

from __future__ import annotations

import json

import pytest

from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings
from fintwinos.core.types import RiskTier, SideEffectClass, ToolBand
from fintwinos.policy.approvals import ApprovalWorkflow
from fintwinos.policy.breakglass import BreakGlass
from fintwinos.policy.killswitch import KillSwitch
from fintwinos.policy.rbac import RbacGate
from fintwinos.policy.rules import gate_from_packs
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec

TRANSFER_SPEC = ToolSpec(
    name="execute_post_transfer",
    description="Post a transfer to the local outbox.",
    input_schema={
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "currency": {"type": "string"},
        },
        "required": ["amount", "currency"],
        "additionalProperties": False,
    },
    band=ToolBand.execute,
    risk_tier=RiskTier.high,
    side_effect=SideEffectClass.reversible,
    requires_human_approval=True,
)


@pytest.fixture()
def world(tmp_path):
    settings = Settings(
        offline=True,
        openai_api_key=None,
        data_dir=tmp_path,
        execute_tools_enabled=True,
        dual_control_required=True,
    )
    audit = AuditTrail()
    # ToolRegistry uses `audit or AuditTrail()` and AuditTrail defines __len__, so an
    # empty (falsy) trail would be silently replaced; seed one record to share the trail.
    audit.append("test", "governance.session_started", {})
    gate = RbacGate(inner=gate_from_packs("treasury"))
    registry = ToolRegistry(policy_gate=gate, audit=audit, settings=settings)

    def post_transfer(arguments, context):
        outbox = settings.data_dir / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        with (outbox / "transfers.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(arguments) + "\n")
        return {"posted": True}

    registry.register(TRANSFER_SPEC, post_transfer)
    approvals = ApprovalWorkflow(audit=audit, settings=settings)
    breakglass = BreakGlass(audit=audit)
    killswitch = KillSwitch(settings=settings, audit=audit)
    killswitch.guard(registry)
    return settings, registry, approvals, breakglass, killswitch


async def test_full_governance_path(world):
    settings, registry, approvals, breakglass, killswitch = world

    # 1. No approval token: the registry's hard execute gate refuses the call.
    bare = await registry.call(
        "execute_post_transfer",
        {"amount": 10_000, "currency": "USD"},
        CallContext(caller="ops", extra={"role": "operator"}),
    )
    assert not bare.ok and bare.requires_approval

    # 2. Maker-checker with dual control mints tokens for the high-tier tool.
    request = approvals.request(TRANSFER_SPEC, requested_by="ops")
    assert request.required_approvals == 2
    approvals.approve(request.request_id, approver="risk-1")
    approvals.approve(request.request_id, approver="risk-2")
    token = request.tokens[0]

    # 3. With the token and an operator role, an in-limit transfer executes.
    allowed = await registry.call(
        "execute_post_transfer",
        {"amount": 10_000, "currency": "USD"},
        CallContext(caller="ops", approval=token, extra={"role": "operator"}),
    )
    assert allowed.ok
    assert (settings.data_dir / "outbox" / "transfers.jsonl").exists()

    # 4. The treasury pack still denies an over-limit transfer, token or not.
    oversize = await registry.call(
        "execute_post_transfer",
        {"amount": 900_000, "currency": "USD"},
        CallContext(caller="ops", approval=token, extra={"role": "operator"}),
    )
    assert not oversize.ok and "denied by policy" in oversize.error

    # 5. RBAC stops an analyst even when holding a valid token.
    analyst = await registry.call(
        "execute_post_transfer",
        {"amount": 10_000, "currency": "USD"},
        CallContext(caller="ana", approval=token, extra={"role": "analyst"}),
    )
    assert not analyst.ok and "analyst" in analyst.error

    # 6. A countersigned break-glass wildcard token also satisfies the hard gate.
    activation = breakglass.activate("ops", "payment rail outage", ttl_minutes=30)
    wildcard = breakglass.countersign(activation.activation_id, "risk-1")
    emergency = await registry.call(
        "execute_post_transfer",
        {"amount": 5_000, "currency": "EUR"},
        CallContext(caller="ops", approval=wildcard, extra={"role": "operator"}),
    )
    assert emergency.ok

    # 7. The kill switch overrides everything, including break-glass.
    killswitch.engage("ops", "incident-7: halting transfers", band=ToolBand.execute)
    halted = await registry.call(
        "execute_post_transfer",
        {"amount": 5_000, "currency": "EUR"},
        CallContext(caller="ops", approval=wildcard, extra={"role": "operator"}),
    )
    assert not halted.ok
    assert halted.error == "kill switch engaged: incident-7: halting transfers"

    # 8. Release restores normal, still-governed operation.
    killswitch.release("admin", "incident resolved", band=ToolBand.execute)
    restored = await registry.call(
        "execute_post_transfer",
        {"amount": 5_000, "currency": "EUR"},
        CallContext(caller="ops", approval=wildcard, extra={"role": "operator"}),
    )
    assert restored.ok

    # 9. Every governance event landed in one tamper-evident chain.
    actions = {record.action for record in registry.audit.records()}
    assert {
        "approval.requested",
        "approval.granted",
        "breakglass.activated",
        "breakglass.token_issued",
        "killswitch.engaged",
        "killswitch.blocked",
        "killswitch.released",
        "policy.checked",
        "tool.completed",
    } <= actions
    assert registry.audit.verify()
