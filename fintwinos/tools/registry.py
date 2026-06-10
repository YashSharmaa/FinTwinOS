"""The typed tool registry.

Every tool carries five governance fields beyond a normal schema — risk tier, approval
policy, side-effect class, idempotency, provenance requirement — and belongs to one of
four bands: ``observe_*``, ``simulate_*``, ``propose_*``, ``execute_*``. Only the
execute band may touch production systems, and only behind policy and approval gates.

Hard invariants enforced here, before any policy rule is even consulted:

- ``observe``/``simulate``/``propose`` tools must be side-effect free (none/read).
- ``execute`` tools must declare a real side effect and always require human approval.
- ``execute`` calls are refused outright unless ``Settings.execute_tools_enabled`` is on,
  a policy rule allows them, AND a valid ``ApprovalToken`` is attached.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from collections.abc import Callable
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import exceptions as js_exceptions
from pydantic import BaseModel, Field, model_validator

from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings, get_settings
from fintwinos.core.errors import SchemaValidationError, ToolNotFound
from fintwinos.core.types import (
    ApprovalToken,
    Provenance,
    RiskTier,
    SideEffectClass,
    ToolBand,
    ToolResult,
)

READ_ONLY_EFFECTS = {SideEffectClass.none, SideEffectClass.read}


class ToolSpec(BaseModel):
    """Declarative description of a tool, serialisable for MCP-style discovery."""

    name: str
    description: str
    input_schema: dict[str, Any]
    band: ToolBand
    risk_tier: RiskTier = RiskTier.low
    side_effect: SideEffectClass = SideEffectClass.none
    requires_human_approval: bool = False
    provenance_required: bool = False
    idempotent: bool = True
    owner: str = "platform"

    @model_validator(mode="after")
    def _enforce_band_invariants(self) -> ToolSpec:
        if self.band in {ToolBand.observe, ToolBand.simulate, ToolBand.propose}:
            if self.side_effect not in READ_ONLY_EFFECTS:
                raise ValueError(
                    f"{self.band.value} tool '{self.name}' must be side-effect free "
                    f"(got {self.side_effect.value})"
                )
        if self.band == ToolBand.execute:
            if self.side_effect in READ_ONLY_EFFECTS:
                raise ValueError(
                    f"execute tool '{self.name}' must declare a reversible or "
                    "irreversible side effect"
                )
            if not self.requires_human_approval:
                raise ValueError(
                    f"execute tool '{self.name}' must set requires_human_approval=True"
                )
        expected_prefix = f"{self.band.value}_"
        if not self.name.startswith(expected_prefix):
            raise ValueError(
                f"tool '{self.name}' must be prefixed with its band: '{expected_prefix}'"
            )
        Draft202012Validator.check_schema(self.input_schema)
        return self

    def to_public_dict(self) -> dict[str, Any]:
        """MCP-style public description with the governance extension fields."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "x_band": self.band.value,
            "x_risk_tier": self.risk_tier.value,
            "x_side_effect": self.side_effect.value,
            "x_requires_human_approval": self.requires_human_approval,
            "x_provenance_required": self.provenance_required,
            "x_idempotent": self.idempotent,
        }


class CallContext(BaseModel):
    """Who is calling, under what ticket, with what approvals."""

    caller: str = "anonymous"
    ticket_id: str | None = None
    approval: ApprovalToken | None = None
    dry_run: bool = False
    idempotency_key: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ToolRegistry:
    """Registration, discovery, validation, gating, auditing and dispatch."""

    def __init__(
        self,
        policy_gate: Any | None = None,
        audit: AuditTrail | None = None,
        settings: Settings | None = None,
    ):
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._idempotency_cache: dict[str, ToolResult] = {}
        self.policy_gate = policy_gate
        self.audit = audit if audit is not None else AuditTrail()
        self.settings = settings or get_settings()

    # -- registration / discovery ------------------------------------------------

    def register(self, spec: ToolSpec, handler: Callable[..., Any]) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool '{spec.name}' is already registered")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise ToolNotFound(f"unknown tool '{name}'")
        return self._specs[name]

    def list_specs(self, band: ToolBand | None = None, pattern: str | None = None) -> list[ToolSpec]:
        specs = list(self._specs.values())
        if band is not None:
            specs = [s for s in specs if s.band == band]
        if pattern is not None:
            specs = [s for s in specs if fnmatch.fnmatch(s.name, pattern)]
        return sorted(specs, key=lambda s: s.name)

    def public_catalog(self) -> list[dict[str, Any]]:
        return [s.to_public_dict() for s in self.list_specs()]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    # -- dispatch ------------------------------------------------------------------

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        context: CallContext | None = None,
    ) -> ToolResult:
        context = context or CallContext()
        started = time.perf_counter()

        try:
            spec = self.spec(name)
        except ToolNotFound as exc:
            self.audit.append(context.caller, "tool.unknown", {"tool": name})
            return ToolResult(ok=False, tool=name, error=str(exc))

        # 1. Schema validation
        try:
            Draft202012Validator(spec.input_schema).validate(arguments)
        except js_exceptions.ValidationError as exc:
            err = SchemaValidationError(name, exc.message)
            self.audit.append(
                context.caller, "tool.schema_rejected", {"tool": name, "error": exc.message}
            )
            return ToolResult(ok=False, tool=name, error=str(err))

        # 2. Hard execute-band gating (independent of any configurable policy)
        if spec.band == ToolBand.execute:
            refusal = self._gate_execute(spec, context)
            if refusal is not None:
                return refusal

        # 3. Configurable policy gate (skipped for dry-run execute calls: a dry run is
        #    a simulation of the action, not the action — it is always permitted)
        dry_run_execute = context.dry_run and spec.band == ToolBand.execute
        if dry_run_execute:
            self.audit.append(
                context.caller, "policy.skipped_dry_run", {"tool": name}
            )
        if self.policy_gate is not None and not dry_run_execute:
            verdict = self.policy_gate.check_tool_call(spec, arguments, context)
            self.audit.append(
                context.caller,
                "policy.checked",
                {"tool": name, "allowed": verdict.allowed,
                 "requires_human_review": verdict.requires_human_review,
                 "rules": verdict.matched_rules},
            )
            if not verdict.allowed:
                return ToolResult(
                    ok=False, tool=name,
                    error=f"denied by policy: {'; '.join(verdict.reasons) or 'no rule allows this call'}",
                )
            if verdict.requires_human_review and not _has_valid_approval(spec, context):
                return ToolResult(
                    ok=False, tool=name, requires_approval=True,
                    error="policy requires human review; attach an approval token",
                )

        # 4. Idempotency replay
        cache_key = None
        if context.idempotency_key:
            cache_key = f"{name}:{context.idempotency_key}"
            if cache_key in self._idempotency_cache:
                cached = self._idempotency_cache[cache_key]
                self.audit.append(context.caller, "tool.idempotent_replay", {"tool": name})
                return cached

        # 5. Invoke
        record = self.audit.append(
            context.caller,
            "tool.called",
            {"tool": name, "band": spec.band.value, "ticket_id": context.ticket_id,
             "dry_run": context.dry_run, "arguments": arguments},
        )
        try:
            handler = self._handlers[name]
            if context.dry_run and spec.band == ToolBand.execute:
                data: Any = {"dry_run": True, "would_execute": name, "arguments": arguments}
            else:
                data = handler(arguments, context)
                if asyncio.iscoroutine(data):
                    data = await data
        except Exception as exc:  # tool failures must never crash the orchestrator
            self.audit.append(context.caller, "tool.failed", {"tool": name, "error": str(exc)})
            return ToolResult(ok=False, tool=name, error=f"{type(exc).__name__}: {exc}")

        elapsed_ms = (time.perf_counter() - started) * 1000
        provenance: list[Provenance] = []
        if isinstance(data, dict) and "provenance" in data:
            raw = data.get("provenance") or []
            provenance = [p if isinstance(p, Provenance) else Provenance(**p) for p in raw]
        elif spec.provenance_required:
            provenance = [Provenance(source_system="twin", notes=f"tool:{name}")]

        result = ToolResult(
            ok=True, tool=name, data=data, provenance=provenance,
            audit_ref=record.hash, elapsed_ms=elapsed_ms,
        )
        self.audit.append(
            context.caller, "tool.completed",
            {"tool": name, "elapsed_ms": round(elapsed_ms, 3), "audit_ref": record.hash},
        )
        if cache_key is not None:
            self._idempotency_cache[cache_key] = result
        return result

    # -- internal -------------------------------------------------------------------

    def _gate_execute(self, spec: ToolSpec, context: CallContext) -> ToolResult | None:
        if context.dry_run:
            return None  # dry runs of execute tools are always permitted
        if not self.settings.execute_tools_enabled:
            self.audit.append(
                context.caller, "tool.execute_disabled", {"tool": spec.name}
            )
            return ToolResult(
                ok=False, tool=spec.name, requires_approval=True,
                error="execute band is disabled in this deployment "
                      "(FINTWIN_EXECUTE_TOOLS_ENABLED=0); run in dry_run or shadow mode",
            )
        if not _has_valid_approval(spec, context):
            self.audit.append(
                context.caller, "tool.approval_missing", {"tool": spec.name}
            )
            return ToolResult(
                ok=False, tool=spec.name, requires_approval=True,
                error=f"tool '{spec.name}' requires a valid human approval token",
            )
        return None


def _has_valid_approval(spec: ToolSpec, context: CallContext) -> bool:
    return context.approval is not None and context.approval.is_valid_for(spec.name)
