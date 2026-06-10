"""Maker-checker approval workflow with dual control for high-risk actions.

The workflow turns "this needs a human" into an auditable state machine:

1. A maker calls :meth:`ApprovalWorkflow.request` for a tool, decision or
   subject string, producing a *pending* :class:`ApprovalRequest`.
2. One or more checkers call :meth:`ApprovalWorkflow.approve`. Maker-checker
   is absolute — the requester can never approve their own request — and when
   ``Settings.dual_control_required`` is on, high and critical risk tiers need
   two *distinct* approvers before anything is granted.
3. Once the required number of approvals lands, the workflow mints one
   :class:`~fintwinos.core.types.ApprovalToken` per approver, each with an
   expiry, ready to attach to a ``CallContext``.

Every transition — request, each approval signature, grant, rejection,
escalation, and every blocked attempt — is appended to the shared
:class:`~fintwinos.core.audit.AuditTrail`.

Import directly from the submodule::

    from fintwinos.policy.approvals import ApprovalWorkflow
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings, get_settings
from fintwinos.core.errors import FinTwinError, PolicyViolation
from fintwinos.core.types import (
    ApprovalStatus,
    ApprovalToken,
    Decision,
    RiskTier,
    new_id,
    utcnow,
)
from fintwinos.policy.rbac import Action, can

#: Risk tiers that require two distinct approvers when dual control is enabled.
DUAL_CONTROL_TIERS = frozenset({RiskTier.high, RiskTier.critical})


class ApprovalError(FinTwinError):
    """Raised for unknown requests or invalid state transitions (e.g. expired)."""


class MakerCheckerViolation(PolicyViolation):
    """Raised when a requester attempts to approve their own request."""


class DualControlViolation(PolicyViolation):
    """Raised when the same approver attempts to provide both dual-control signatures."""


class ApprovalRequest(BaseModel):
    """One approval request flowing through the maker-checker state machine."""

    request_id: str = Field(default_factory=lambda: new_id("req"))
    subject: str
    requested_by: str
    risk_tier: RiskTier = RiskTier.medium
    reason: str = ""
    scope: dict[str, Any] = Field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.pending
    required_approvals: int = 1
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    resolution: dict[str, Any] | None = None
    tokens: list[ApprovalToken] = Field(default_factory=list)

    def approver_names(self) -> list[str]:
        """The distinct approvers who have signed so far, in signing order."""
        return [entry["approver"] for entry in self.approvals]

    @property
    def is_open(self) -> bool:
        """True while the request can still collect approvals (pending or escalated)."""
        return self.status in {ApprovalStatus.pending, ApprovalStatus.escalated}


def _resolve_subject(subject: Any) -> tuple[str, RiskTier | None]:
    """Map a Decision, ToolSpec-like object or plain string to (subject, derived tier)."""
    if isinstance(subject, Decision):
        return subject.decision_id, subject.risk_tier
    if isinstance(subject, str):
        if not subject.strip():
            raise ValueError("approval subject must be a non-empty string")
        return subject, None
    name = getattr(subject, "name", None)
    if isinstance(name, str) and name:
        tier = getattr(subject, "risk_tier", None)
        return name, tier if isinstance(tier, RiskTier) else None
    raise TypeError(
        "approval subject must be a Decision, a tool spec with a 'name', or a string; "
        f"got {type(subject).__name__}"
    )


class ApprovalWorkflow:
    """Maker-checker approvals with dual control, expiring tokens and full audit.

    Parameters
    ----------
    audit:
        Shared audit trail; every transition is appended. A private trail is
        created when omitted, but production callers should pass
        ``runtime.audit`` / ``registry.audit``.
    settings:
        Governs ``dual_control_required``. Defaults to the process settings.
    token_ttl_minutes:
        Lifetime of minted :class:`ApprovalToken`s, from the moment of grant.
    request_ttl_minutes:
        How long a request may sit before approvals are refused as expired.
    """

    def __init__(
        self,
        audit: AuditTrail | None = None,
        settings: Settings | None = None,
        token_ttl_minutes: float = 60.0,
        request_ttl_minutes: float = 24 * 60.0,
    ):
        if token_ttl_minutes <= 0:
            raise ValueError("token_ttl_minutes must be positive")
        if request_ttl_minutes <= 0:
            raise ValueError("request_ttl_minutes must be positive")
        # NB: AuditTrail defines __len__, so an empty trail is falsy — compare to None.
        self.audit = audit if audit is not None else AuditTrail()
        self.settings = settings or get_settings()
        self.token_ttl_minutes = token_ttl_minutes
        self.request_ttl_minutes = request_ttl_minutes
        self._requests: dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()

    # -- request --------------------------------------------------------------------

    def request(
        self,
        subject: Any,
        requested_by: str,
        risk_tier: RiskTier | None = None,
        reason: str = "",
        scope: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> ApprovalRequest:
        """Open a pending approval request for a tool spec, decision or subject string.

        The risk tier is taken from the explicit argument first, then derived
        from the subject (``Decision.risk_tier`` / ``ToolSpec.risk_tier``),
        defaulting to ``medium``. High/critical tiers require two distinct
        approvers when ``settings.dual_control_required`` is on.
        """
        now = now or utcnow()
        subject_id, derived_tier = _resolve_subject(subject)
        tier = risk_tier or derived_tier or RiskTier.medium
        request = ApprovalRequest(
            subject=subject_id,
            requested_by=requested_by,
            risk_tier=tier,
            reason=reason,
            scope=dict(scope or {}),
            required_approvals=self._required_approvals(tier),
            created_at=now,
            expires_at=now + timedelta(minutes=self.request_ttl_minutes),
        )
        with self._lock:
            self._requests[request.request_id] = request
        self.audit.append(
            requested_by,
            "approval.requested",
            {
                "request_id": request.request_id,
                "subject": request.subject,
                "risk_tier": tier.value,
                "required_approvals": request.required_approvals,
                "reason": reason,
                "expires_at": request.expires_at.isoformat() if request.expires_at else None,
            },
        )
        return request

    # -- approve ----------------------------------------------------------------------

    def approve(
        self,
        request_id: str,
        approver: str,
        role: str = "approver",
        now: datetime | None = None,
    ) -> ApprovalRequest:
        """Record one approval signature, granting tokens once the quorum is met.

        Enforces, in order: the request exists and is still open; the request
        has not expired; ``role`` holds the ``approval.approve`` permission;
        maker-checker (``approver != requested_by``); and dual control (each
        signature from a distinct approver). Violations are audited and raised.
        """
        now = now or utcnow()
        request = self._get(request_id)
        if not request.is_open:
            raise ApprovalError(
                f"request '{request_id}' is {request.status.value}; only pending or "
                "escalated requests can be approved"
            )
        if request.expires_at is not None and now > request.expires_at:
            request.status = ApprovalStatus.rejected
            request.resolution = {"by": "system", "reason": "request expired", "at": now.isoformat()}
            self.audit.append(
                approver, "approval.expired", {"request_id": request_id, "subject": request.subject}
            )
            raise ApprovalError(f"request '{request_id}' expired at {request.expires_at.isoformat()}")
        if not can(role, Action.approval_approve):
            self._audit_blocked(approver, request, f"role '{role}' may not approve requests")
            raise PolicyViolation(f"role '{role}' does not hold the approval.approve permission")
        if approver == request.requested_by:
            self._audit_blocked(approver, request, "maker-checker: requester cannot self-approve")
            raise MakerCheckerViolation(
                f"maker-checker violation: '{approver}' requested '{request.subject}' "
                "and cannot approve it"
            )
        if approver in request.approver_names():
            self._audit_blocked(approver, request, "dual control: approver already signed")
            raise DualControlViolation(
                f"dual-control violation: '{approver}' has already signed request '{request_id}'"
            )

        request.approvals.append({"approver": approver, "role": role, "at": now.isoformat()})
        self.audit.append(
            approver,
            "approval.approved",
            {
                "request_id": request_id,
                "subject": request.subject,
                "signatures": len(request.approvals),
                "required": request.required_approvals,
            },
        )
        if len(request.approvals) >= request.required_approvals:
            self._grant(request, now)
        return request

    # -- reject / escalate --------------------------------------------------------------

    def reject(
        self,
        request_id: str,
        actor: str,
        reason: str = "",
        role: str = "approver",
        now: datetime | None = None,
    ) -> ApprovalRequest:
        """Terminally reject an open request. Requires the ``approval.reject`` permission."""
        now = now or utcnow()
        request = self._get(request_id)
        if not request.is_open:
            raise ApprovalError(f"request '{request_id}' is {request.status.value}; cannot reject")
        if not can(role, Action.approval_reject):
            self._audit_blocked(actor, request, f"role '{role}' may not reject requests")
            raise PolicyViolation(f"role '{role}' does not hold the approval.reject permission")
        request.status = ApprovalStatus.rejected
        request.resolution = {"by": actor, "reason": reason, "at": now.isoformat()}
        self.audit.append(
            actor,
            "approval.rejected",
            {"request_id": request_id, "subject": request.subject, "reason": reason},
        )
        return request

    def escalate(
        self,
        request_id: str,
        actor: str,
        reason: str = "",
        role: str = "approver",
        now: datetime | None = None,
    ) -> ApprovalRequest:
        """Escalate an open request to a higher authority; it can still be approved.

        Escalated requests remain open: the escalation marks them for senior
        attention without discarding signatures already collected.
        """
        now = now or utcnow()
        request = self._get(request_id)
        if not request.is_open:
            raise ApprovalError(f"request '{request_id}' is {request.status.value}; cannot escalate")
        if not can(role, Action.approval_escalate):
            self._audit_blocked(actor, request, f"role '{role}' may not escalate requests")
            raise PolicyViolation(f"role '{role}' does not hold the approval.escalate permission")
        request.status = ApprovalStatus.escalated
        self.audit.append(
            actor,
            "approval.escalated",
            {"request_id": request_id, "subject": request.subject, "reason": reason},
        )
        return request

    # -- views ----------------------------------------------------------------------------

    def get(self, request_id: str) -> ApprovalRequest:
        """Fetch a request by id, raising :class:`ApprovalError` when unknown."""
        return self._get(request_id)

    def pending(self) -> list[ApprovalRequest]:
        """Requests still collecting approvals (pending or escalated), oldest first."""
        with self._lock:
            return [r for r in self._requests.values() if r.is_open]

    def history(self) -> list[ApprovalRequest]:
        """Every request ever made, in creation order, regardless of state."""
        with self._lock:
            return list(self._requests.values())

    def tokens(self, request_id: str) -> list[ApprovalToken]:
        """The tokens minted for a granted request (empty until quorum is met)."""
        return list(self._get(request_id).tokens)

    # -- internal ----------------------------------------------------------------------------

    def _required_approvals(self, tier: RiskTier) -> int:
        if self.settings.dual_control_required and tier in DUAL_CONTROL_TIERS:
            return 2
        return 1

    def _grant(self, request: ApprovalRequest, now: datetime) -> None:
        expires_at = now + timedelta(minutes=self.token_ttl_minutes)
        request.tokens = [
            ApprovalToken(
                subject=request.subject,
                granted_by=entry["approver"],
                role=entry["role"],
                granted_at=now,
                expires_at=expires_at,
                scope={"request_id": request.request_id, "risk_tier": request.risk_tier.value},
            )
            for entry in request.approvals
        ]
        request.status = ApprovalStatus.approved
        self.audit.append(
            request.approver_names()[-1],
            "approval.granted",
            {
                "request_id": request.request_id,
                "subject": request.subject,
                "approvers": request.approver_names(),
                "token_ids": [t.token_id for t in request.tokens],
                "expires_at": expires_at.isoformat(),
            },
        )

    def _get(self, request_id: str) -> ApprovalRequest:
        with self._lock:
            request = self._requests.get(request_id)
        if request is None:
            raise ApprovalError(f"unknown approval request '{request_id}'")
        return request

    def _audit_blocked(self, actor: str, request: ApprovalRequest, reason: str) -> None:
        self.audit.append(
            actor,
            "approval.blocked",
            {"request_id": request.request_id, "subject": request.subject, "reason": reason},
        )
