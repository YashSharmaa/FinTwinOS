"""Shared plumbing for the packaged demos.

:func:`build_stack` assembles the whole platform from the four contracted
entry points (``build_runtime`` → ``register_all`` → ``build_default_registry``,
plus an :class:`~fintwinos.models.llm_routing.client.LLMClient`), importing the
sibling modules lazily so a partial install fails with a clear message rather
than at import time.

The helpers here are deliberately defensive: demos discover tools by name
pattern, adapt arguments to each tool's published JSON Schema, and normalise
simulator payloads — so the demos stay working as the catalog evolves, and any
gap degrades into a recorded warning instead of a crash.
"""

from __future__ import annotations

import inspect
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import fintwinos
from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings
from fintwinos.models.llm_routing.client import LLMClient
from fintwinos.policy.gates import PolicyGate, PolicyRule, RuleEffect
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec


@dataclass
class DemoStack:
    """Everything a demo touches: the assembled twin, registry, LLM and console."""

    settings: Settings
    runtime: Any
    registry: ToolRegistry
    llm: LLMClient
    console: Console
    seed: int
    warnings: list[str] = field(default_factory=list)


def demo_settings(offline_ok: bool = True) -> Settings:
    """Build settings for a demo run, never crashing on a missing API key.

    With no OpenAI key (or ``FINTWIN_OFFLINE=1``) the stack silently switches
    to offline mode, where every agent uses its deterministic rule-based
    fallback. Pass ``offline_ok=False`` to insist on a live LLM instead.
    """
    settings = Settings()
    if settings.llm_available:
        return settings
    if not offline_ok:
        raise RuntimeError(
            "demo requires a live LLM (offline_ok=False) but no OpenAI key is configured "
            "and/or FINTWIN_OFFLINE is set; export OPENAI_API_KEY or allow offline mode"
        )
    if not settings.offline:
        settings = Settings(offline=True)
    return settings


def build_stack(
    seed: int | None = None,
    console: Console | None = None,
    offline_ok: bool = True,
    settings: Settings | None = None,
) -> DemoStack:
    """Assemble the demo twin through the contracted platform entry points.

    Imports ``fintwinos.twin_core``, ``fintwinos.twin_sim`` and
    ``fintwinos.tools.catalog`` lazily, guarantees the registry carries a
    policy gate and the demo settings, and shares one audit trail across the
    run so the closing governance summary covers everything.

    ``seed`` defaults to ``Settings.seed`` (``FINTWIN_SEED``), so demo runs are
    reproducible *and* steerable from the environment.
    """
    settings = settings or demo_settings(offline_ok=offline_ok)
    console = console or Console()
    if seed is None:
        seed = settings.seed

    from fintwinos.tools.catalog import build_default_registry
    from fintwinos.twin_core.runtime import build_runtime
    from fintwinos.twin_sim import register_all

    runtime = build_runtime(seed=seed, with_demo_data=True)
    register_all(runtime)
    registry = build_default_registry(runtime, policy_gate=None, settings=settings)
    if getattr(registry, "policy_gate", None) is None:
        registry.policy_gate = PolicyGate()
    registry.settings = settings  # demo settings drive execute-band gating
    llm = LLMClient(settings)
    return DemoStack(
        settings=settings,
        runtime=runtime,
        registry=registry,
        llm=llm,
        console=console,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tool discovery and invocation
# ---------------------------------------------------------------------------


def find_tool(registry: ToolRegistry, candidates: list[str]) -> str | None:
    """Return the first registered tool matching the candidate names/globs."""
    for candidate in candidates:
        if any(ch in candidate for ch in "*?["):
            specs = registry.list_specs(pattern=candidate)
            if specs:
                return specs[0].name
        elif candidate in registry:
            return candidate
    return None


#: Sensible fill-ins for required schema properties the demo did not supply.
_NAMED_DEFAULTS: dict[str, Any] = {
    "seed": 7,
    "horizon_days": 30,
    "currency": "USD",
    "currencies": ["USD"],
    "k": 5,
    "limit": 5,
    "query": "risk factors",
    "ticket_id": "DEMO-0001",
    "assumptions_version": "v1",
}

_TYPE_DEFAULTS: dict[str, Any] = {
    "string": "",
    "number": 0.0,
    "integer": 0,
    "boolean": False,
    "array": [],
    "object": {},
}


def schema_args(spec: ToolSpec, preferred: dict[str, Any]) -> dict[str, Any]:
    """Adapt preferred arguments to a tool's published input schema.

    Keeps only keys the schema declares (when it declares any), then fills
    required properties the caller did not provide — by conventional name
    first, then by JSON type — so schema validation never rejects a demo call
    over an argument-name mismatch.
    """
    schema = spec.input_schema or {}
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return {k: v for k, v in preferred.items() if v is not None}
    args = {k: v for k, v in preferred.items() if k in properties and v is not None}
    for required in schema.get("required", []):
        if required in args or required not in properties:
            continue
        if required in _NAMED_DEFAULTS:
            args[required] = _NAMED_DEFAULTS[required]
        else:
            prop_type = properties[required].get("type", "string")
            if isinstance(prop_type, list):
                prop_type = prop_type[0] if prop_type else "string"
            args[required] = _TYPE_DEFAULTS.get(prop_type, "")
    return args


async def call_tool(
    stack: DemoStack,
    name: str,
    arguments: dict[str, Any],
    caller: str,
    approval: Any = None,
    ticket_id: str | None = None,
    role: str | None = None,
) -> Any:
    """Dispatch one audited tool call through the registry.

    ``role`` is asserted in ``CallContext.extra["role"]`` for the RBAC gate;
    calls without one run under ``Settings.rbac_default_role`` (analyst), which
    cannot reach the execute band.
    """
    extra = {"role": role} if role else {}
    context = CallContext(caller=caller, ticket_id=ticket_id, approval=approval, extra=extra)
    return await stack.registry.call(name, arguments, context)


async def maybe_await(value: Any) -> Any:
    """Await ``value`` if a sibling entry point turned out to be async."""
    if inspect.isawaitable(value):
        return await value
    return value


def ensure_allow_rule(gate: Any, tool_name: str, rule_name: str) -> None:
    """Idempotently add an explicit policy ``allow`` rule for one execute tool.

    The execute band is default-deny; a demo that legitimately exercises an
    execute tool (behind an approval token) must first put an allow rule on
    the record, exactly as an adopting institution would.
    """
    # The default gate chain is RbacGate(PolicyGate()): unwrap to the rule-bearing gate.
    while gate is not None and not hasattr(gate, "add_rule"):
        gate = getattr(gate, "inner", None)
    if gate is None:
        return
    existing = {getattr(rule, "name", None) for rule in getattr(gate, "rules", [])}
    if rule_name in existing:
        return
    gate.add_rule(
        PolicyRule(
            name=rule_name,
            description=f"Demo policy pack: '{tool_name}' is permitted with human approval.",
            effect=RuleEffect.allow,
            tools=[tool_name],
        )
    )


# ---------------------------------------------------------------------------
# Result normalisation
# ---------------------------------------------------------------------------


def as_plain(value: Any) -> Any:
    """Recursively convert pydantic models / nested containers to plain data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: as_plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [as_plain(v) for v in value]
    return value


def extract_simulation(data: Any) -> dict[str, Any]:
    """Normalise a tool payload into a ``SimulationResult``-shaped dict."""
    plain = as_plain(data)
    base: dict[str, Any] = {}
    if isinstance(plain, dict):
        if isinstance(plain.get("metrics"), dict):
            base = plain
        else:
            for key in ("result", "simulation", "sim", "data"):
                nested = plain.get(key)
                if isinstance(nested, dict) and isinstance(nested.get("metrics"), dict):
                    base = nested
                    break
    return {
        "metrics": dict(base.get("metrics", {})),
        "confidence": dict(base.get("confidence", {})),
        "series": dict(base.get("series", {})),
        "calibration": dict(base.get("calibration", {})),
        "warnings": list(base.get("warnings", [])),
        "scenario_name": base.get("scenario_name", ""),
        "simulator": base.get("simulator", ""),
    }


def metric_named(sim: dict[str, Any], *keywords: str) -> float | None:
    """First metric whose name contains a keyword (checked in keyword order)."""
    metrics = sim.get("metrics", {})
    for keyword in keywords:
        for name, value in metrics.items():
            if keyword in name and isinstance(value, int | float):
                return float(value)
    return None


def ci_named(sim: dict[str, Any], *keywords: str) -> list[float] | None:
    """First confidence interval whose name contains a keyword."""
    confidence = sim.get("confidence", {})
    for keyword in keywords:
        for name, bounds in confidence.items():
            if keyword in name and isinstance(bounds, list | tuple) and len(bounds) == 2:
                return [float(bounds[0]), float(bounds[1])]
    return None


def normalize_case(raw: Any) -> dict[str, Any]:
    """Coerce a ``handle_case`` return into the contracted shape."""
    plain = as_plain(raw)
    if not isinstance(plain, dict):
        plain = {"decision": plain}
    plain.setdefault("status", "complete")
    plain.setdefault("decision", None)
    plain.setdefault("policy", None)
    return plain


# ---------------------------------------------------------------------------
# Audit views
# ---------------------------------------------------------------------------


def audit_excerpt(audit: AuditTrail, n: int = 8) -> list[dict[str, Any]]:
    """The last ``n`` audit records, summarised for display and assertions."""
    return [
        {
            "seq": record.seq,
            "actor": record.actor,
            "action": record.action,
            "hash": record.hash[:12],
        }
        for record in audit.records()[-n:]
    ]


def audit_summary(audit: AuditTrail, n_excerpt: int = 8) -> dict[str, Any]:
    """Length, integrity verdict and a tail excerpt of the audit chain."""
    return {
        "records": len(audit),
        "verified": audit.verify(),
        "excerpt": audit_excerpt(audit, n_excerpt),
    }


def tool_calls_by_band(audit: AuditTrail) -> dict[str, int]:
    """Count audited ``tool.called`` records grouped by tool band."""
    counter: Counter[str] = Counter()
    for record in audit.records(action="tool.called"):
        counter[str(record.payload.get("band", "unknown"))] += 1
    return dict(counter)


# ---------------------------------------------------------------------------
# Rich rendering helpers
# ---------------------------------------------------------------------------


def mode_line(settings: Settings) -> str:
    """Human-readable runtime mode for panel subtitles."""
    if settings.offline:
        return "offline — deterministic rule-based fallbacks, no network"
    return f"online — OpenAI ({settings.llm_model_primary} / {settings.llm_model_fast})"


def render_header(console: Console, title: str, tagline: str, settings: Settings) -> None:
    """Print the standard demo banner with runtime mode and attribution."""
    console.print(
        Panel(
            f"[bold]{tagline}[/bold]\n[dim]mode: {mode_line(settings)}[/dim]",
            title=f"[bold cyan]FinTwinOS · {title}[/bold cyan]",
            subtitle=f"[dim]{fintwinos.CREDIT}[/dim]",
            border_style="cyan",
        )
    )


def render_case(console: Console, case: dict[str, Any], title: str = "Agent decision") -> None:
    """Render a decision + policy verdict from a (normalised) case result."""
    decision = case.get("decision") or {}
    policy = case.get("policy") or {}
    status = str(case.get("status", "unknown"))
    status_style = {
        "complete": "green",
        "awaiting_human": "yellow",
        "blocked": "red",
    }.get(status, "white")

    lines = [f"status: [{status_style}]{status}[/{status_style}]"]
    if isinstance(decision, dict) and decision:
        lines.append(f"objective: {decision.get('objective', '—')}")
        lines.append(
            f"action_type: [bold]{decision.get('action_type', '—')}[/bold]   "
            f"risk_tier: {decision.get('risk_tier', '—')}   "
            f"owner: {decision.get('owner', '—')}"
        )
        rationale = str(decision.get("rationale", "")).strip()
        if rationale:
            lines.append(f"rationale: {rationale}")
        planned = decision.get("planned_tool_calls") or []
        if planned:
            names = ", ".join(str(c.get("tool", c.get("name", "?"))) for c in planned if isinstance(c, dict))
            lines.append(f"planned tool calls: {names}")
    if isinstance(policy, dict) and policy:
        allowed = policy.get("allowed")
        review = policy.get("requires_human_review")
        verdict_style = "green" if allowed else "red"
        lines.append(
            f"policy verdict: [{verdict_style}]{'allowed' if allowed else 'denied'}[/{verdict_style}]"
            f"{' · human review required' if review else ''}"
        )
        reasons = policy.get("reasons") or []
        if reasons:
            lines.append(f"policy reasons: {'; '.join(str(r) for r in reasons)}")
    console.print(Panel("\n".join(lines), title=f"[bold]{title}[/bold]", border_style="magenta"))


def render_audit_excerpt(
    console: Console, excerpt: list[dict[str, Any]], title: str = "Audit trail (tail)"
) -> None:
    """Render the tail of the hash-chained audit trail."""
    table = Table(title=title, title_justify="left", border_style="dim")
    table.add_column("seq", justify="right", style="dim")
    table.add_column("actor")
    table.add_column("action", style="cyan")
    table.add_column("hash", style="dim")
    for row in excerpt:
        table.add_row(str(row["seq"]), row["actor"], row["action"], row["hash"])
    console.print(table)


def fmt_num(value: Any, digits: int = 2) -> str:
    """Format a possibly-missing number for table cells."""
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    """Format a possibly-missing ratio as a percentage."""
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_ci(bounds: list[float] | None, digits: int = 1) -> str:
    """Format a two-sided confidence interval."""
    if not bounds or len(bounds) != 2:
        return "—"
    return f"[{float(bounds[0]):.{digits}f}, {float(bounds[1]):.{digits}f}]"
