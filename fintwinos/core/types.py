"""Canonical entities, enums and envelopes shared across the federated twin.

These models are the lingua franca of FinTwinOS: connectors emit ``EventEnvelope``s,
the twin stores canonical entities, tools return ``ToolResult``s, simulators return
``SimulationResult``s, and agents produce ``Decision``s that pass through policy gates.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    """Timezone-aware UTC now, used everywhere instead of naive datetimes."""
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Generate a prefixed, sortable-enough unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskTier(StrEnum):
    """Risk tier attached to every tool and decision; drives approval policy."""

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


RISK_ORDER: dict[RiskTier, int] = {
    RiskTier.low: 0,
    RiskTier.medium: 1,
    RiskTier.high: 2,
    RiskTier.critical: 3,
}


class SideEffectClass(StrEnum):
    """What a tool does to the outside world."""

    none = "none"
    read = "read"
    reversible = "reversible"
    irreversible = "irreversible"


class ToolBand(StrEnum):
    """The four tool bands. Only ``execute`` may ever touch production systems."""

    observe = "observe"
    simulate = "simulate"
    propose = "propose"
    execute = "execute"


class ApprovalStatus(StrEnum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    escalated = "escalated"


class CaseStatus(StrEnum):
    open = "open"
    in_review = "in_review"
    awaiting_human = "awaiting_human"
    closed = "closed"
    escalated = "escalated"


class AlertStatus(StrEnum):
    new = "new"
    triaged = "triaged"
    investigating = "investigating"
    dismissed = "dismissed"
    confirmed = "confirmed"


# ---------------------------------------------------------------------------
# Provenance and events
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Where a record came from and how it can be traced back."""

    source_system: str
    ingested_at: datetime = Field(default_factory=utcnow)
    record_hash: str | None = None
    licence: str | None = None
    notes: str | None = None


class EntityRef(BaseModel):
    """A typed pointer to a canonical entity in the twin."""

    entity_type: str
    entity_id: str

    def key(self) -> str:
        return f"{self.entity_type}:{self.entity_id}"

    def __hash__(self) -> int:  # allow use in sets / dict keys
        return hash(self.key())


class EventEnvelope(BaseModel):
    """The single ingestion unit. Every connector emits these; the twin consumes them."""

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    kind: str
    occurred_at: datetime = Field(default_factory=utcnow)
    recorded_at: datetime = Field(default_factory=utcnow)
    source: str
    entities: list[EntityRef] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    episode_id: str | None = None
    provenance: Provenance | None = None

    def content_hash(self) -> str:
        body = json.dumps(
            {"kind": self.kind, "source": self.source, "payload": self.payload},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Canonical entities
# ---------------------------------------------------------------------------


class Customer(BaseModel):
    customer_id: str
    name: str
    segment: str = "retail"
    risk_rating: RiskTier = RiskTier.low
    country: str = "GB"
    attributes: dict[str, Any] = Field(default_factory=dict)


class LegalEntity(BaseModel):
    legal_entity_id: str
    name: str
    jurisdiction: str = "GB"
    lei: str | None = None
    parent_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class Account(BaseModel):
    account_id: str
    customer_id: str
    currency: str = "USD"
    account_type: str = "deposit"
    balance: float = 0.0
    opened_at: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class Instrument(BaseModel):
    instrument_id: str
    symbol: str
    asset_class: str = "equity"
    currency: str = "USD"
    attributes: dict[str, Any] = Field(default_factory=dict)


class Trade(BaseModel):
    trade_id: str
    account_id: str
    instrument_id: str
    side: str  # "buy" | "sell"
    quantity: float
    price: float
    executed_at: datetime = Field(default_factory=utcnow)
    venue: str = "XOFF"
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("side")
    @classmethod
    def _check_side(cls, v: str) -> str:
        if v not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        return v


class Position(BaseModel):
    account_id: str
    instrument_id: str
    quantity: float
    market_value: float
    as_of: datetime = Field(default_factory=utcnow)


class CaseRecord(BaseModel):
    case_id: str
    kind: str  # e.g. "aml_alert", "complaint", "limit_breach"
    status: CaseStatus = CaseStatus.open
    priority: RiskTier = RiskTier.medium
    opened_at: datetime = Field(default_factory=utcnow)
    entities: list[EntityRef] = Field(default_factory=list)
    narrative: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)


class PolicyDoc(BaseModel):
    policy_id: str
    title: str
    body: str
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0"
    effective_date: datetime | None = None


class Scenario(BaseModel):
    scenario_id: str = Field(default_factory=lambda: new_id("scn"))
    name: str
    kind: str = "stress"  # stress | replay | counterfactual | what_if
    params: dict[str, Any] = Field(default_factory=dict)
    narrative: str = ""
    assumptions_version: str = "v1"


class Alert(BaseModel):
    alert_id: str = Field(default_factory=lambda: new_id("alr"))
    kind: str
    severity: RiskTier = RiskTier.medium
    score: float = 0.0
    status: AlertStatus = AlertStatus.new
    created_at: datetime = Field(default_factory=utcnow)
    entities: list[EntityRef] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Results, decisions and approvals
# ---------------------------------------------------------------------------


class SimulationResult(BaseModel):
    """Every simulator returns this shape: point estimates AND uncertainty."""

    simulator: str
    scenario_name: str
    ok: bool = True
    metrics: dict[str, float] = Field(default_factory=dict)
    series: dict[str, list[float]] = Field(default_factory=dict)
    confidence: dict[str, list[float]] = Field(default_factory=dict)  # name -> [lo, hi]
    calibration: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    assumptions_version: str = "v1"
    seed: int | None = None


class ToolResult(BaseModel):
    """Uniform return type of every tool call routed through the registry."""

    ok: bool
    tool: str
    data: Any = None
    error: str | None = None
    provenance: list[Provenance] = Field(default_factory=list)
    audit_ref: str | None = None
    requires_approval: bool = False
    elapsed_ms: float | None = None


class ApprovalToken(BaseModel):
    """A human (or dual-control pair) granting permission for a specific action."""

    token_id: str = Field(default_factory=lambda: new_id("apr"))
    subject: str  # tool name or decision_id the approval covers
    granted_by: str
    role: str = "approver"
    status: ApprovalStatus = ApprovalStatus.approved
    granted_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    scope: dict[str, Any] = Field(default_factory=dict)

    def is_valid_for(self, subject: str, now: datetime | None = None) -> bool:
        now = now or utcnow()
        if self.status != ApprovalStatus.approved:
            return False
        if self.expires_at is not None and now > self.expires_at:
            return False
        return self.subject in {subject, "*"}


class Decision(BaseModel):
    """The unit that flows from the planner through the policy gate."""

    decision_id: str = Field(default_factory=lambda: new_id("dec"))
    objective: str
    action_type: str = "propose_only"  # propose_only | execute
    planned_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    risk_tier: RiskTier = RiskTier.medium
    owner: str = "planner"
    ticket_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("action_type")
    @classmethod
    def _check_action_type(cls, v: str) -> str:
        if v not in {"propose_only", "execute"}:
            raise ValueError("action_type must be 'propose_only' or 'execute'")
        return v


CANONICAL_MODELS: list[type[BaseModel]] = [
    Provenance,
    EntityRef,
    EventEnvelope,
    Customer,
    LegalEntity,
    Account,
    Instrument,
    Trade,
    Position,
    CaseRecord,
    PolicyDoc,
    Scenario,
    Alert,
    SimulationResult,
    ToolResult,
    ApprovalToken,
    Decision,
]
