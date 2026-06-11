"""Tests for the kill switch: persistence, band scoping and registry enforcement."""

from __future__ import annotations

import json

import pytest

from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings
from fintwinos.core.types import (
    ApprovalToken,
    RiskTier,
    SideEffectClass,
    ToolBand,
)
from fintwinos.policy.gates import PolicyGate, PolicyRule, RuleEffect
from fintwinos.policy.killswitch import KILLSWITCH_FILENAME, KillSwitch
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        offline=True,
        openai_api_key=None,
        data_dir=tmp_path,
        execute_tools_enabled=True,
        shadow_mode=False,  # these tests assert the real handler/outbox path
    )


@pytest.fixture()
def registry(settings) -> ToolRegistry:
    gate = PolicyGate(
        rules=[
            PolicyRule(
                name="allow-transfers",
                effect=RuleEffect.allow,
                tools=["execute_post_transfer"],
            )
        ]
    )
    reg = ToolRegistry(policy_gate=gate, audit=AuditTrail(), settings=settings)
    reg.register(
        ToolSpec(
            name="observe_balance",
            description="Read an account balance from the twin.",
            input_schema={"type": "object"},
            band=ToolBand.observe,
        ),
        lambda arguments, context: {"balance": 100.0},
    )

    def post_transfer(arguments, context):
        outbox = settings.data_dir / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        with (outbox / "transfers.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(arguments) + "\n")
        return {"posted": True, "amount": arguments.get("amount")}

    reg.register(
        ToolSpec(
            name="execute_post_transfer",
            description="Post a transfer to the outbox.",
            input_schema={"type": "object"},
            band=ToolBand.execute,
            risk_tier=RiskTier.high,
            side_effect=SideEffectClass.reversible,
            requires_human_approval=True,
        ),
        post_transfer,
    )
    return reg


def approved_context() -> CallContext:
    return CallContext(
        caller="ops",
        approval=ApprovalToken(subject="execute_post_transfer", granted_by="risk-lead"),
    )


async def test_execute_band_halt_blocks_only_execute(settings, registry):
    ks = KillSwitch(settings=settings, audit=registry.audit)
    ks.guard(registry)

    ok = await registry.call("observe_balance", {}, CallContext(caller="ops"))
    assert ok.ok

    ks.engage("ops", "incident-42: anomalous transfer pattern", band=ToolBand.execute)

    blocked = await registry.call("execute_post_transfer", {"amount": 10}, approved_context())
    assert not blocked.ok
    assert blocked.error == "kill switch engaged: incident-42: anomalous transfer pattern"
    assert not (settings.data_dir / "outbox").exists()  # handler never ran

    still_ok = await registry.call("observe_balance", {}, CallContext(caller="ops"))
    assert still_ok.ok
    assert registry.audit.records(action="killswitch.blocked")


def test_role_enforcement_when_role_supplied(settings):
    from fintwinos.core.errors import PolicyViolation

    ks = KillSwitch(settings=settings, audit=AuditTrail())
    # operator may engage but not release; admin may do both.
    ks.engage("ops", "incident", band="execute", role="operator")
    with pytest.raises(PolicyViolation, match="killswitch.release"):
        ks.release("ops", "resolved", band="execute", role="operator")
    ks.release("root", "resolved", band="execute", role="admin")
    assert not ks.is_engaged(ToolBand.execute)
    assert ks.audit.records(action="killswitch.blocked")


def test_role_none_skips_rbac(settings):
    ks = KillSwitch(settings=settings, audit=AuditTrail())
    # No role -> trusted-caller path, no RBAC check (current default behaviour).
    ks.engage("cli", "halt")
    ks.release("cli", "resolved")
    assert not ks.is_engaged()


async def test_global_halt_blocks_everything(settings, registry):
    ks = KillSwitch(settings=settings, audit=registry.audit)
    ks.guard(registry)
    ks.engage("ops", "full trading halt")

    observe = await registry.call("observe_balance", {}, CallContext(caller="ops"))
    assert not observe.ok and observe.error == "kill switch engaged: full trading halt"

    execute = await registry.call("execute_post_transfer", {"amount": 10}, approved_context())
    assert not execute.ok and execute.error == "kill switch engaged: full trading halt"


async def test_release_restores_calls_cleanly(settings, registry):
    ks = KillSwitch(settings=settings, audit=registry.audit)
    ks.guard(registry)
    ks.engage("ops", "incident-42", band="execute")
    ks.engage("ops", "broader halt")

    ks.release("admin", "incident resolved")  # global
    observe = await registry.call("observe_balance", {}, CallContext(caller="ops"))
    assert observe.ok
    execute = await registry.call("execute_post_transfer", {"amount": 10}, approved_context())
    assert not execute.ok  # execute band still halted

    ks.release("admin", "band cleared", band=ToolBand.execute)
    execute = await registry.call("execute_post_transfer", {"amount": 10}, approved_context())
    assert execute.ok and execute.data["posted"] is True
    assert (settings.data_dir / "outbox" / "transfers.jsonl").exists()


async def test_unguard_restores_original_call(settings, registry):
    ks = KillSwitch(settings=settings, audit=registry.audit)
    ks.guard(registry)
    ks.engage("ops", "halt")
    ks.unguard(registry)

    # with the guard removed the registry behaves normally again
    result = await registry.call("observe_balance", {}, CallContext(caller="ops"))
    assert result.ok
    assert getattr(registry, "_killswitch_guard", None) is None


async def test_guard_is_idempotent(settings, registry):
    ks = KillSwitch(settings=settings, audit=registry.audit)
    ks.guard(registry)
    first = registry.call
    ks.guard(registry)
    assert registry.call is first  # not double-wrapped


async def test_unknown_tools_are_delegated(settings, registry):
    ks = KillSwitch(settings=settings, audit=registry.audit)
    ks.guard(registry)
    ks.engage("ops", "halt")
    result = await registry.call("observe_nonexistent", {}, CallContext(caller="ops"))
    assert not result.ok and "unknown tool" in result.error


def test_state_persists_across_instances(settings):
    ks1 = KillSwitch(settings=settings, audit=AuditTrail())
    ks1.engage("ops", "incident-42", band=ToolBand.execute)
    assert (settings.data_dir / KILLSWITCH_FILENAME).exists()

    ks2 = KillSwitch(settings=settings, audit=AuditTrail())
    assert ks2.is_engaged(band=ToolBand.execute)
    assert not ks2.is_engaged()  # global flag is off
    assert ks2.blocking_reason(ToolBand.execute) == "incident-42"

    ks2.release("admin", "resolved", band="execute")
    assert not ks1.is_engaged(band="execute")  # ks1 re-reads the shared file
    history = ks1.status()["history"]
    assert [h["event"] for h in history] == ["engage", "release"]


def test_corrupt_state_file_fails_closed(settings):
    ks = KillSwitch(settings=settings, audit=AuditTrail())
    ks.path.parent.mkdir(parents=True, exist_ok=True)
    ks.path.write_text("{not json", encoding="utf-8")
    assert ks.is_engaged()
    assert "failing closed" in ks.blocking_reason(ToolBand.observe)


def test_engage_requires_reason(settings):
    ks = KillSwitch(settings=settings, audit=AuditTrail())
    with pytest.raises(ValueError, match="reason"):
        ks.engage("ops", "  ")


def test_unknown_band_rejected(settings):
    ks = KillSwitch(settings=settings, audit=AuditTrail())
    with pytest.raises(ValueError, match="unknown tool band"):
        ks.engage("ops", "halt", band="exfiltrate")


def test_engage_and_release_are_audited(settings):
    audit = AuditTrail()
    ks = KillSwitch(settings=settings, audit=audit)
    ks.engage("ops", "halt", band="execute")
    ks.release("admin", "resolved", band="execute")
    actions = [r.action for r in audit.records()]
    assert "killswitch.engaged" in actions and "killswitch.released" in actions
    assert audit.verify()
