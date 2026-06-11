"""Role-based access control for the FinTwinOS governance plane.

Five roles cover the personas around a financial digital twin:

- ``viewer``    — read-only: may call ``observe_*`` tools, nothing else.
- ``analyst``   — investigates: observe, simulate and propose; may request
  approvals and escalate them, but can neither approve nor execute.
- ``approver``  — the checker in maker-checker: approves, rejects and
  escalates approval requests, countersigns break-glass activations and may
  revoke them; deliberately *cannot* execute, so no single person both
  approves and acts.
- ``operator``  — runs the platform: all four tool bands, requests approvals,
  activates and revokes break-glass and engages the kill switch. Cannot approve
  their own requests (maker-checker is enforced by the approval workflow) and
  cannot release an engaged kill switch — that is reserved for ``admin``.
- ``admin``     — every action, including kill-switch release.

The matrix is exposed two ways: :func:`can` for point checks, and
:class:`RbacGate` which plugs into the tool registry's ``policy_gate`` slot,
reading the caller's role from ``CallContext.extra["role"]`` and composing
with an inner gate (typically the foundation ``PolicyGate``).

Import directly from the submodule::

    from fintwinos.policy.rbac import Action, RbacGate, Role, can
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fintwinos.core.types import RISK_ORDER, Decision, RiskTier
from fintwinos.policy.gates import Verdict


class Role(StrEnum):
    """The five governance roles, ordered roughly by privilege."""

    viewer = "viewer"
    analyst = "analyst"
    approver = "approver"
    operator = "operator"
    admin = "admin"


class Action(StrEnum):
    """Everything the RBAC matrix gates: the four tool bands plus governance actions."""

    observe = "observe"
    simulate = "simulate"
    propose = "propose"
    execute = "execute"
    approval_request = "approval.request"
    approval_approve = "approval.approve"
    approval_reject = "approval.reject"
    approval_escalate = "approval.escalate"
    breakglass_activate = "breakglass.activate"
    breakglass_countersign = "breakglass.countersign"
    breakglass_revoke = "breakglass.revoke"
    killswitch_engage = "killswitch.engage"
    killswitch_release = "killswitch.release"


#: The role -> permitted-actions matrix. ``admin`` holds every action.
PERMISSIONS: dict[Role, frozenset[Action]] = {
    Role.viewer: frozenset({Action.observe}),
    Role.analyst: frozenset(
        {
            Action.observe,
            Action.simulate,
            Action.propose,
            Action.approval_request,
            Action.approval_escalate,
        }
    ),
    Role.approver: frozenset(
        {
            Action.observe,
            Action.simulate,
            Action.approval_approve,
            Action.approval_reject,
            Action.approval_escalate,
            Action.breakglass_countersign,
            Action.breakglass_revoke,
        }
    ),
    Role.operator: frozenset(
        {
            Action.observe,
            Action.simulate,
            Action.propose,
            Action.execute,
            Action.approval_request,
            Action.approval_escalate,
            Action.breakglass_activate,
            Action.breakglass_revoke,
            Action.killswitch_engage,
        }
    ),
    Role.admin: frozenset(Action),
}


def can(role: Role | str, action: Action | str) -> bool:
    """Return True when ``role`` is permitted to perform ``action``.

    Both arguments accept enum members or their string values. Unknown roles
    or actions are never permitted — RBAC fails closed.
    """
    try:
        resolved_role = Role(role)
    except ValueError:
        return False
    try:
        resolved_action = Action(action)
    except ValueError:
        return False
    return resolved_action in PERMISSIONS[resolved_role]


class RbacGate:
    """A policy gate that enforces the role matrix before any rule-based policy.

    Designed for the registry's ``policy_gate`` slot. The caller's role is read
    from ``CallContext.extra["role"]``; calls without a role fall back to
    ``default_role`` (``viewer`` — fail closed). When the role permits the
    tool's band, the verdict is delegated to ``inner`` (typically a
    ``PolicyGate`` built from YAML packs); with no inner gate the call is
    allowed, leaving the registry's structural execute-band gating intact.
    """

    def __init__(self, inner: Any | None = None, default_role: Role = Role.viewer):
        self.inner = inner
        self.default_role = default_role

    # -- tool calls ---------------------------------------------------------------

    def check_tool_call(
        self, spec: Any, arguments: dict[str, Any], context: Any = None
    ) -> Verdict:
        """Deny when the caller's role may not act in the tool's band, else delegate."""
        raw_role = None
        if context is not None:
            extra = getattr(context, "extra", None)
            if isinstance(extra, dict):
                raw_role = extra.get("role")
        if raw_role is None:
            role = self.default_role
        else:
            try:
                role = Role(raw_role)
            except ValueError:
                return Verdict(
                    allowed=False,
                    requires_human_review=True,
                    reasons=[f"unknown role '{raw_role}': RBAC fails closed"],
                    matched_rules=["rbac:unknown-role"],
                )

        band = spec.band.value if hasattr(spec.band, "value") else str(spec.band)
        if not can(role, band):
            return Verdict(
                allowed=False,
                requires_human_review=True,
                reasons=[f"role '{role.value}' may not call {band}-band tools"],
                matched_rules=[f"rbac:role={role.value}"],
            )
        if self.inner is not None:
            verdict = self.inner.check_tool_call(spec, arguments, context)
            return verdict.model_copy(
                update={"matched_rules": [f"rbac:role={role.value}", *verdict.matched_rules]}
            )
        return Verdict(allowed=True, matched_rules=[f"rbac:role={role.value}"])

    # -- planner decisions ----------------------------------------------------------

    def check_decision(self, decision: Decision) -> Verdict:
        """Delegate decision checks to the inner gate; default to conservative review."""
        if self.inner is not None and hasattr(self.inner, "check_decision"):
            return self.inner.check_decision(decision)
        requires_review = (
            decision.action_type == "execute"
            or RISK_ORDER[decision.risk_tier] >= RISK_ORDER[RiskTier.high]
        )
        return Verdict(
            allowed=True,
            requires_human_review=requires_review,
            reasons=(["execute or high-risk decisions require human review"] if requires_review else []),
            matched_rules=(["rbac:decision-review"] if requires_review else []),
        )
