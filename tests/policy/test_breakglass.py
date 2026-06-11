"""Tests for the break-glass emergency override: countersign, expiry, audit."""

from __future__ import annotations

from datetime import timedelta

import pytest

from fintwinos.core.audit import AuditTrail
from fintwinos.core.errors import PolicyViolation
from fintwinos.core.types import utcnow
from fintwinos.policy.approvals import MakerCheckerViolation
from fintwinos.policy.breakglass import (
    BreakGlass,
    BreakGlassError,
    BreakGlassExpired,
    BreakGlassStatus,
)


def make_breakglass() -> BreakGlass:
    return BreakGlass(audit=AuditTrail())


def test_activation_is_pending_with_no_token():
    bg = make_breakglass()
    activation = bg.activate("ops-1", "payment rail outage, manual drain required")
    assert activation.status == BreakGlassStatus.pending
    assert activation.token is None
    records = bg.audit.records(action="breakglass.activated")
    assert records and "BREAK-GLASS" in records[0].payload["banner"]


def test_blank_reason_is_rejected():
    bg = make_breakglass()
    with pytest.raises(ValueError, match="reason"):
        bg.activate("ops-1", "   ")


def test_activator_cannot_countersign_own_activation():
    bg = make_breakglass()
    activation = bg.activate("ops-1", "incident-42")
    with pytest.raises(MakerCheckerViolation):
        bg.countersign(activation.activation_id, "ops-1")
    assert activation.status == BreakGlassStatus.pending
    assert activation.token is None


def test_countersign_issues_wildcard_token():
    bg = make_breakglass()
    activation = bg.activate("ops-1", "incident-42", ttl_minutes=30)
    token = bg.countersign(activation.activation_id, "risk-lead")
    assert token.subject == "*"
    assert token.is_valid_for("execute_post_transfer")
    assert token.is_valid_for("execute_anything_else")
    assert token.expires_at == activation.expires_at
    assert activation.status == BreakGlassStatus.active
    assert activation.countersigned_by == "risk-lead"
    assert activation in bg.active()
    actions = {record.action for record in bg.audit.records()}
    assert {"breakglass.activated", "breakglass.countersigned", "breakglass.token_issued"} <= actions


def test_token_expires_with_the_window():
    bg = make_breakglass()
    start = utcnow()
    activation = bg.activate("ops-1", "incident-42", ttl_minutes=10, now=start)
    token = bg.countersign(activation.activation_id, "risk-lead", now=start)
    inside = start + timedelta(minutes=9)
    outside = start + timedelta(minutes=11)
    assert token.is_valid_for("execute_post_transfer", now=inside)
    assert not token.is_valid_for("execute_post_transfer", now=outside)
    # the listing flips the stale activation to expired
    assert bg.active(now=outside) == []
    listed = bg.activations(now=outside)
    assert listed[0].status == BreakGlassStatus.expired


def test_countersign_after_window_is_refused():
    bg = make_breakglass()
    start = utcnow()
    activation = bg.activate("ops-1", "incident-42", ttl_minutes=5, now=start)
    with pytest.raises(BreakGlassExpired):
        bg.countersign(activation.activation_id, "risk-lead", now=start + timedelta(minutes=6))
    assert activation.status == BreakGlassStatus.expired
    assert activation.token is None
    assert bg.audit.records(action="breakglass.expired")


def test_expired_activation_cannot_be_countersigned_twice():
    bg = make_breakglass()
    start = utcnow()
    activation = bg.activate("ops-1", "incident-42", ttl_minutes=5, now=start)
    with pytest.raises(BreakGlassExpired):
        bg.countersign(activation.activation_id, "risk-lead", now=start + timedelta(minutes=6))
    with pytest.raises(BreakGlassError, match="expired"):
        bg.countersign(activation.activation_id, "risk-lead", now=start)


def test_countersigner_role_must_hold_permission():
    bg = make_breakglass()
    activation = bg.activate("ops-1", "incident-42")
    with pytest.raises(PolicyViolation, match="breakglass.countersign"):
        bg.countersign(activation.activation_id, "intern", role="viewer")


def test_activator_role_must_hold_permission():
    bg = make_breakglass()
    with pytest.raises(PolicyViolation, match="breakglass.activate"):
        bg.activate("intern", "incident-42", role="viewer")


def test_revoke_invalidates_token_immediately():
    bg = make_breakglass()
    activation = bg.activate("ops-1", "incident-42", ttl_minutes=60)
    token = bg.countersign(activation.activation_id, "risk-lead")
    assert token.is_valid_for("execute_post_transfer")
    bg.revoke(activation.activation_id, "risk-lead", reason="incident resolved")
    assert activation.status == BreakGlassStatus.revoked
    assert not token.is_valid_for("execute_post_transfer")
    assert bg.audit.records(action="breakglass.revoked")


def test_revoke_rejects_role_without_permission():
    bg = make_breakglass()
    activation = bg.activate("ops-1", "incident-42", ttl_minutes=60)
    bg.countersign(activation.activation_id, "risk-lead")
    with pytest.raises(PolicyViolation, match="breakglass.revoke"):
        bg.revoke(activation.activation_id, "intern", role="viewer")
    assert activation.status != BreakGlassStatus.revoked


def test_unknown_activation_raises():
    bg = make_breakglass()
    with pytest.raises(BreakGlassError, match="unknown"):
        bg.countersign("bgx_missing", "risk-lead")


def test_every_activation_is_listed():
    bg = make_breakglass()
    a = bg.activate("ops-1", "incident-1", ttl_minutes=60)
    b = bg.activate("ops-2", "incident-2", ttl_minutes=60)
    bg.countersign(b.activation_id, "risk-lead")
    listed = bg.activations()
    assert {x.activation_id for x in listed} == {a.activation_id, b.activation_id}
    pending = bg.activations(status=BreakGlassStatus.pending)
    assert [x.activation_id for x in pending] == [a.activation_id]
    assert bg.audit.verify()
