"""Break-glass: time-boxed, countersigned emergency override.

When an incident demands action that normal policy would block, break-glass
issues a wildcard :class:`~fintwinos.core.types.ApprovalToken` (``subject="*"``)
— but only under three locks:

1. **Two people.** The activating actor states a reason; a *different* actor
   must countersign before any token exists.
2. **A closing window.** The countersignature must land within the activation
   window (``ttl_minutes`` from activation), and the issued token expires at
   the end of that same window — the override cannot outlive the emergency.
3. **A loud trail.** Activation, countersignature, token issuance, expiry and
   revocation are all appended to the shared audit trail with an unmissable
   banner, and every activation remains listable forever.

Break-glass deliberately does not bypass the tool registry: the wildcard token
still flows through ``CallContext.approval``, so the registry's structural
gates (execute band enabled, policy allow rules) keep applying.

Import directly from the submodule::

    from fintwinos.policy.breakglass import BreakGlass
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field

from fintwinos.core.audit import AuditTrail
from fintwinos.core.errors import FinTwinError, PolicyViolation
from fintwinos.core.types import ApprovalStatus, ApprovalToken, new_id, utcnow
from fintwinos.policy.approvals import MakerCheckerViolation
from fintwinos.policy.rbac import Action, can

#: Banner string attached to every break-glass audit record so log scrapers
#: and humans cannot miss it.
BREAKGLASS_BANNER = "*** BREAK-GLASS EMERGENCY OVERRIDE ***"


class BreakGlassError(FinTwinError):
    """Raised for unknown activations or invalid break-glass transitions."""


class BreakGlassExpired(BreakGlassError):
    """Raised when a countersignature arrives after the activation window closed."""


class BreakGlassStatus(StrEnum):
    """Lifecycle of one break-glass activation."""

    pending = "pending_countersign"
    active = "active"
    expired = "expired"
    revoked = "revoked"


class BreakGlassActivation(BaseModel):
    """One emergency-override activation and everything that happened to it."""

    activation_id: str = Field(default_factory=lambda: new_id("bgx"))
    requested_by: str
    reason: str
    requested_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    status: BreakGlassStatus = BreakGlassStatus.pending
    countersigned_by: str | None = None
    countersigned_at: datetime | None = None
    token: ApprovalToken | None = None
    resolution: str | None = None


class BreakGlass:
    """Two-person, time-boxed emergency override issuing wildcard approval tokens.

    Parameters
    ----------
    audit:
        Shared audit trail. Pass ``runtime.audit`` / ``registry.audit`` so
        break-glass events land in the same hash chain as everything else.
    default_ttl_minutes:
        Window length used when :meth:`activate` is called without an
        explicit ``ttl_minutes``.
    """

    def __init__(self, audit: AuditTrail | None = None, default_ttl_minutes: float = 15.0):
        if default_ttl_minutes <= 0:
            raise ValueError("default_ttl_minutes must be positive")
        # NB: AuditTrail defines __len__, so an empty trail is falsy — compare to None.
        self.audit = audit if audit is not None else AuditTrail()
        self.default_ttl_minutes = default_ttl_minutes
        self._activations: dict[str, BreakGlassActivation] = {}
        self._lock = threading.Lock()

    # -- activation -----------------------------------------------------------------

    def activate(
        self,
        actor: str,
        reason: str,
        ttl_minutes: float | None = None,
        role: str = "operator",
        now: datetime | None = None,
    ) -> BreakGlassActivation:
        """Open a break-glass window. No token is issued until a second actor countersigns.

        ``reason`` is mandatory and recorded verbatim; ``ttl_minutes`` bounds
        both the countersign deadline and the eventual token's lifetime.
        """
        now = now or utcnow()
        if not reason or not reason.strip():
            raise ValueError("break-glass activation requires a non-empty reason")
        ttl = self.default_ttl_minutes if ttl_minutes is None else float(ttl_minutes)
        if ttl <= 0:
            raise ValueError("ttl_minutes must be positive")
        if not can(role, Action.breakglass_activate):
            raise PolicyViolation(
                f"role '{role}' does not hold the breakglass.activate permission"
            )
        activation = BreakGlassActivation(
            requested_by=actor,
            reason=reason.strip(),
            requested_at=now,
            expires_at=now + timedelta(minutes=ttl),
        )
        with self._lock:
            self._activations[activation.activation_id] = activation
        self.audit.append(
            actor,
            "breakglass.activated",
            {
                "banner": BREAKGLASS_BANNER,
                "severity": "critical",
                "activation_id": activation.activation_id,
                "reason": activation.reason,
                "ttl_minutes": ttl,
                "expires_at": activation.expires_at.isoformat(),
                "awaiting_countersign": True,
            },
        )
        return activation

    # -- countersign -------------------------------------------------------------------

    def countersign(
        self,
        activation_id: str,
        actor: str,
        role: str = "approver",
        now: datetime | None = None,
    ) -> ApprovalToken:
        """Countersign a pending activation, issuing the wildcard token.

        The countersigner must differ from the activator (maker-checker), must
        hold the ``breakglass.countersign`` permission, and must sign before
        the window closes; a late signature flips the activation to expired
        and raises :class:`BreakGlassExpired`.
        """
        now = now or utcnow()
        activation = self._get(activation_id)
        if activation.status != BreakGlassStatus.pending:
            raise BreakGlassError(
                f"activation '{activation_id}' is {activation.status.value}; "
                "only pending activations can be countersigned"
            )
        if now > activation.expires_at:
            activation.status = BreakGlassStatus.expired
            activation.resolution = "countersign window expired"
            self.audit.append(
                actor,
                "breakglass.expired",
                {
                    "banner": BREAKGLASS_BANNER,
                    "activation_id": activation_id,
                    "expired_at": activation.expires_at.isoformat(),
                },
            )
            raise BreakGlassExpired(
                f"activation '{activation_id}' expired at {activation.expires_at.isoformat()}; "
                "activate again if the emergency persists"
            )
        if not can(role, Action.breakglass_countersign):
            raise PolicyViolation(
                f"role '{role}' does not hold the breakglass.countersign permission"
            )
        if actor == activation.requested_by:
            self.audit.append(
                actor,
                "breakglass.blocked",
                {
                    "banner": BREAKGLASS_BANNER,
                    "activation_id": activation_id,
                    "reason": "maker-checker: activator cannot countersign their own activation",
                },
            )
            raise MakerCheckerViolation(
                f"break-glass maker-checker violation: '{actor}' activated "
                f"'{activation_id}' and cannot countersign it"
            )

        token = ApprovalToken(
            subject="*",
            granted_by=f"{activation.requested_by}+{actor}",
            role="breakglass",
            granted_at=now,
            expires_at=activation.expires_at,
            scope={
                "activation_id": activation.activation_id,
                "reason": activation.reason,
                "countersigned_by": actor,
            },
        )
        activation.status = BreakGlassStatus.active
        activation.countersigned_by = actor
        activation.countersigned_at = now
        activation.token = token
        self.audit.append(
            actor,
            "breakglass.countersigned",
            {
                "banner": BREAKGLASS_BANNER,
                "severity": "critical",
                "activation_id": activation_id,
                "activated_by": activation.requested_by,
                "token_id": token.token_id,
                "token_expires_at": activation.expires_at.isoformat(),
            },
        )
        self.audit.append(
            actor,
            "breakglass.token_issued",
            {
                "banner": BREAKGLASS_BANNER,
                "activation_id": activation_id,
                "token_id": token.token_id,
                "subject": "*",
                "expires_at": activation.expires_at.isoformat(),
            },
        )
        return token

    # -- revoke -----------------------------------------------------------------------

    def revoke(
        self,
        activation_id: str,
        actor: str,
        reason: str = "",
        role: str = "operator",
        now: datetime | None = None,
    ) -> BreakGlassActivation:
        """Revoke an activation early, invalidating its token immediately.

        ``role`` must hold the ``breakglass.revoke`` permission (operator,
        approver or admin); a denied role is audited and raises
        :class:`~fintwinos.core.errors.PolicyViolation` before any state change.
        """
        now = now or utcnow()
        if not can(role, Action.breakglass_revoke):
            self.audit.append(
                actor,
                "breakglass.blocked",
                {"activation_id": activation_id, "reason": f"role '{role}' may not revoke"},
            )
            raise PolicyViolation(f"role '{role}' does not hold the breakglass.revoke permission")
        activation = self._get(activation_id)
        if activation.status in {BreakGlassStatus.revoked, BreakGlassStatus.expired}:
            raise BreakGlassError(
                f"activation '{activation_id}' is already {activation.status.value}"
            )
        activation.status = BreakGlassStatus.revoked
        activation.resolution = reason or "revoked"
        if activation.token is not None:
            activation.token.status = ApprovalStatus.rejected
        self.audit.append(
            actor,
            "breakglass.revoked",
            {
                "banner": BREAKGLASS_BANNER,
                "activation_id": activation_id,
                "reason": reason,
                "token_id": activation.token.token_id if activation.token else None,
            },
        )
        return activation

    # -- views ------------------------------------------------------------------------

    def activations(
        self, status: BreakGlassStatus | None = None, now: datetime | None = None
    ) -> list[BreakGlassActivation]:
        """Every activation ever made (loudly listable), optionally filtered by status.

        Stale pending/active activations whose window has closed are flipped to
        ``expired`` (and audited) before the listing is returned.
        """
        now = now or utcnow()
        with self._lock:
            items = list(self._activations.values())
        for activation in items:
            self._refresh(activation, now)
        if status is not None:
            items = [a for a in items if a.status == status]
        return items

    def active(self, now: datetime | None = None) -> list[BreakGlassActivation]:
        """Activations whose wildcard token is currently live."""
        return self.activations(status=BreakGlassStatus.active, now=now)

    # -- internal ------------------------------------------------------------------------

    def _refresh(self, activation: BreakGlassActivation, now: datetime) -> None:
        if (
            activation.status in {BreakGlassStatus.pending, BreakGlassStatus.active}
            and now > activation.expires_at
        ):
            activation.status = BreakGlassStatus.expired
            activation.resolution = activation.resolution or "window elapsed"
            self.audit.append(
                "breakglass",
                "breakglass.expired",
                {
                    "banner": BREAKGLASS_BANNER,
                    "activation_id": activation.activation_id,
                    "expired_at": activation.expires_at.isoformat(),
                },
            )

    def _get(self, activation_id: str) -> BreakGlassActivation:
        with self._lock:
            activation = self._activations.get(activation_id)
        if activation is None:
            raise BreakGlassError(f"unknown break-glass activation '{activation_id}'")
        return activation
