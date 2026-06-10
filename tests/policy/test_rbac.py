"""Tests for the RBAC role/permission matrix and the composing RbacGate."""

from __future__ import annotations

from fintwinos.core.types import Decision, RiskTier, SideEffectClass, ToolBand
from fintwinos.policy.gates import PolicyGate, PolicyRule, RuleEffect
from fintwinos.policy.rbac import PERMISSIONS, Action, RbacGate, Role, can
from fintwinos.tools.registry import CallContext, ToolSpec


def spec_for(band: ToolBand) -> ToolSpec:
    side_effect = (
        SideEffectClass.reversible if band == ToolBand.execute else SideEffectClass.none
    )
    return ToolSpec(
        name=f"{band.value}_thing",
        description="test tool",
        input_schema={"type": "object"},
        band=band,
        risk_tier=RiskTier.medium,
        side_effect=side_effect,
        requires_human_approval=band == ToolBand.execute,
    )


def ctx(role: str | None) -> CallContext:
    extra = {} if role is None else {"role": role}
    return CallContext(caller="tester", extra=extra)


# --- matrix -----------------------------------------------------------------------


def test_viewer_is_read_only():
    assert can(Role.viewer, Action.observe)
    for action in Action:
        if action is not Action.observe:
            assert not can(Role.viewer, action), action


def test_analyst_investigates_but_never_approves_or_executes():
    assert can("analyst", "observe")
    assert can("analyst", "simulate")
    assert can("analyst", "propose")
    assert can("analyst", "approval.request")
    assert not can("analyst", "execute")
    assert not can("analyst", "approval.approve")
    assert not can("analyst", "killswitch.engage")


def test_approver_checks_but_never_executes():
    assert can(Role.approver, Action.approval_approve)
    assert can(Role.approver, Action.approval_reject)
    assert can(Role.approver, Action.breakglass_countersign)
    assert not can(Role.approver, Action.execute)
    assert not can(Role.approver, Action.propose)
    assert not can(Role.approver, Action.approval_request)


def test_operator_executes_but_cannot_approve_or_release():
    assert can(Role.operator, Action.execute)
    assert can(Role.operator, Action.killswitch_engage)
    assert can(Role.operator, Action.breakglass_activate)
    assert not can(Role.operator, Action.approval_approve)
    assert not can(Role.operator, Action.killswitch_release)


def test_admin_holds_every_action():
    assert all(can(Role.admin, action) for action in Action)
    assert PERMISSIONS[Role.admin] == frozenset(Action)


def test_unknown_role_or_action_fails_closed():
    assert not can("superuser", Action.observe)
    assert not can(Role.admin, "rm -rf /")


def test_no_single_role_below_admin_can_both_request_and_approve():
    for role in (Role.viewer, Role.analyst, Role.approver, Role.operator):
        assert not (
            can(role, Action.approval_request) and can(role, Action.approval_approve)
        ), f"{role} breaks maker-checker separation"


# --- RbacGate -----------------------------------------------------------------------


def test_rbac_gate_denies_band_outside_role():
    gate = RbacGate()
    verdict = gate.check_tool_call(spec_for(ToolBand.execute), {}, ctx("analyst"))
    assert not verdict.allowed
    assert "analyst" in verdict.reasons[0]


def test_rbac_gate_defaults_to_viewer_without_role():
    gate = RbacGate()
    assert gate.check_tool_call(spec_for(ToolBand.observe), {}, ctx(None)).allowed
    assert not gate.check_tool_call(spec_for(ToolBand.simulate), {}, ctx(None)).allowed
    assert not gate.check_tool_call(spec_for(ToolBand.execute), {}, None).allowed


def test_rbac_gate_rejects_unknown_role():
    gate = RbacGate()
    verdict = gate.check_tool_call(spec_for(ToolBand.observe), {}, ctx("superuser"))
    assert not verdict.allowed
    assert "unknown role" in verdict.reasons[0]


def test_rbac_gate_composes_with_inner_policy_gate():
    inner = PolicyGate(
        rules=[
            PolicyRule(name="allow-exec", effect=RuleEffect.allow, tools=["execute_thing"]),
        ]
    )
    gate = RbacGate(inner=inner)

    operator = gate.check_tool_call(spec_for(ToolBand.execute), {}, ctx("operator"))
    assert operator.allowed
    assert operator.matched_rules[0] == "rbac:role=operator"
    assert "allow-exec" in operator.matched_rules

    # the same call is stopped at the RBAC layer for an analyst, never reaching the inner gate
    analyst = gate.check_tool_call(spec_for(ToolBand.execute), {}, ctx("analyst"))
    assert not analyst.allowed


def test_rbac_gate_inner_deny_still_denies():
    inner = PolicyGate(
        rules=[PolicyRule(name="deny-exec", effect=RuleEffect.deny, bands=[ToolBand.execute])]
    )
    gate = RbacGate(inner=inner)
    verdict = gate.check_tool_call(spec_for(ToolBand.execute), {}, ctx("admin"))
    assert not verdict.allowed
    assert "deny-exec" in verdict.matched_rules


def test_rbac_gate_decision_checks():
    inner = PolicyGate()
    gate = RbacGate(inner=inner)
    decision = Decision(objective="x", action_type="execute", risk_tier=RiskTier.low)
    delegated = gate.check_decision(decision)
    assert delegated.allowed and delegated.requires_human_review

    bare_gate = RbacGate()
    proposal = Decision(objective="x", action_type="propose_only", risk_tier=RiskTier.low)
    assert not bare_gate.check_decision(proposal).requires_human_review
    high = Decision(objective="x", action_type="propose_only", risk_tier=RiskTier.high)
    assert bare_gate.check_decision(high).requires_human_review
