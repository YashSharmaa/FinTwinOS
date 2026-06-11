"""Tests for the maker-checker approval workflow and dual control."""

from __future__ import annotations

from datetime import timedelta

import pytest

from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings
from fintwinos.core.errors import PolicyViolation
from fintwinos.core.types import ApprovalStatus, Decision, RiskTier, utcnow
from fintwinos.policy.approvals import (
    ApprovalError,
    ApprovalWorkflow,
    DualControlViolation,
    MakerCheckerViolation,
)

SETTINGS = Settings(offline=True, openai_api_key=None, dual_control_required=True)
SETTINGS_NO_DUAL = Settings(offline=True, openai_api_key=None, dual_control_required=False)


def make_workflow(settings: Settings = SETTINGS, **kwargs) -> ApprovalWorkflow:
    return ApprovalWorkflow(audit=AuditTrail(), settings=settings, **kwargs)


def test_request_creates_pending_and_audits():
    wf = make_workflow()
    req = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.medium)
    assert req.status == ApprovalStatus.pending
    assert req.required_approvals == 1
    assert req.expires_at is not None
    assert req in wf.pending()
    assert wf.audit.records(action="approval.requested")
    assert wf.audit.verify()


def test_request_rejects_role_without_permission():
    wf = make_workflow()
    with pytest.raises(PolicyViolation, match="approval.request"):
        wf.request(
            "execute_post_transfer", requested_by="vince",
            risk_tier=RiskTier.medium, role="viewer",
        )
    assert wf.audit.records(action="approval.blocked")


def test_request_allows_operator_role():
    wf = make_workflow()
    req = wf.request(
        "execute_post_transfer", requested_by="ops", risk_tier=RiskTier.medium, role="operator"
    )
    assert req.status == ApprovalStatus.pending


def test_maker_checker_rejects_self_approval():
    wf = make_workflow()
    req = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.medium)
    with pytest.raises(MakerCheckerViolation):
        wf.approve(req.request_id, approver="alice")
    assert req.status == ApprovalStatus.pending
    assert not req.tokens
    blocked = wf.audit.records(action="approval.blocked")
    assert blocked and "maker-checker" in blocked[-1].payload["reason"]


def test_single_approval_grants_token_for_medium_tier():
    wf = make_workflow()
    req = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.medium)
    wf.approve(req.request_id, approver="bob")
    assert req.status == ApprovalStatus.approved
    assert len(req.tokens) == 1
    token = req.tokens[0]
    assert token.is_valid_for("execute_post_transfer")
    assert token.expires_at is not None and token.expires_at > utcnow()
    assert token.scope["request_id"] == req.request_id
    assert req not in wf.pending()
    assert wf.audit.records(action="approval.granted")


def test_dual_control_requires_two_distinct_approvers():
    wf = make_workflow()
    req = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.high)
    assert req.required_approvals == 2

    wf.approve(req.request_id, approver="bob")
    assert req.status == ApprovalStatus.pending
    assert not req.tokens  # one signature is not enough

    with pytest.raises(DualControlViolation):
        wf.approve(req.request_id, approver="bob")  # same checker cannot sign twice

    wf.approve(req.request_id, approver="carol")
    assert req.status == ApprovalStatus.approved
    assert len(req.tokens) == 2
    assert {t.granted_by for t in req.tokens} == {"bob", "carol"}
    assert all(t.is_valid_for("execute_post_transfer") for t in req.tokens)


def test_dual_control_applies_to_critical_tier():
    wf = make_workflow()
    req = wf.request("execute_anything", requested_by="alice", risk_tier=RiskTier.critical)
    assert req.required_approvals == 2


def test_dual_control_disabled_needs_single_approver():
    wf = make_workflow(settings=SETTINGS_NO_DUAL)
    req = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.high)
    assert req.required_approvals == 1
    wf.approve(req.request_id, approver="bob")
    assert req.status == ApprovalStatus.approved


def test_reject_terminates_request():
    wf = make_workflow()
    req = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.medium)
    wf.reject(req.request_id, actor="bob", reason="amount looks wrong")
    assert req.status == ApprovalStatus.rejected
    assert req.resolution["reason"] == "amount looks wrong"
    assert req not in wf.pending()
    assert req in wf.history()
    with pytest.raises(ApprovalError):
        wf.approve(req.request_id, approver="carol")


def test_escalated_request_remains_approvable():
    wf = make_workflow()
    req = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.medium)
    wf.escalate(req.request_id, actor="bob", reason="needs treasury head sign-off")
    assert req.status == ApprovalStatus.escalated
    assert req in wf.pending()  # still open
    wf.approve(req.request_id, approver="carol")
    assert req.status == ApprovalStatus.approved


def test_unknown_request_raises():
    wf = make_workflow()
    with pytest.raises(ApprovalError, match="unknown approval request"):
        wf.approve("req_does_not_exist", approver="bob")


def test_role_without_approve_permission_is_blocked():
    wf = make_workflow()
    req = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.medium)
    with pytest.raises(PolicyViolation, match="approval.approve"):
        wf.approve(req.request_id, approver="bob", role="analyst")
    assert req.status == ApprovalStatus.pending


def test_request_from_decision_inherits_tier_and_subject():
    wf = make_workflow()
    decision = Decision(objective="rebalance", action_type="execute", risk_tier=RiskTier.high)
    req = wf.request(decision, requested_by="planner")
    assert req.subject == decision.decision_id
    assert req.risk_tier == RiskTier.high
    assert req.required_approvals == 2


def test_expired_request_cannot_be_approved():
    wf = make_workflow(request_ttl_minutes=30)
    req = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.medium)
    late = utcnow() + timedelta(hours=2)
    with pytest.raises(ApprovalError, match="expired"):
        wf.approve(req.request_id, approver="bob", now=late)
    assert req.status == ApprovalStatus.rejected
    assert wf.audit.records(action="approval.expired")


def test_tokens_expire():
    wf = make_workflow(token_ttl_minutes=10)
    req = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.medium)
    wf.approve(req.request_id, approver="bob")
    token = req.tokens[0]
    assert token.is_valid_for("execute_post_transfer")
    far_future = utcnow() + timedelta(hours=1)
    assert not token.is_valid_for("execute_post_transfer", now=far_future)


def test_full_transition_audit_chain_verifies():
    wf = make_workflow()
    a = wf.request("execute_post_transfer", requested_by="alice", risk_tier=RiskTier.high)
    wf.approve(a.request_id, approver="bob")
    wf.approve(a.request_id, approver="carol")
    b = wf.request("execute_close_case", requested_by="dora", risk_tier=RiskTier.low)
    wf.reject(b.request_id, actor="bob", reason="not yet")
    actions = {record.action for record in wf.audit.records()}
    assert {"approval.requested", "approval.approved", "approval.granted", "approval.rejected"} <= actions
    assert wf.audit.verify()
