"""Demo: rehearsing a USD liquidity squeeze inside the twin.

Flow: stage a deteriorating USD cash ladder in the twin (the squeeze is in
today's data, not in the prompt) → observe it through ``observe_cash_ladder``
→ run ``simulate_liquidity_stress`` through the governed tool registry for two
stress presets, showing survival days with confidence intervals and the
funding-cost impact → hand the case to the agent runtime
(``handle_case("case_liq", ...)``) and show the resulting decision, policy
verdict and an excerpt of the hash-chained audit trail.

Run it with ``fintwinos demo liquidity`` or call :func:`run` directly. Works
fully offline: with no OpenAI key the stack falls back to deterministic
rule-based behaviour, and if the platform simulator is unavailable a bundled
Monte Carlo stress model keeps the rehearsal honest.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from fintwinos.core.config import Settings
from fintwinos.core.types import Decision, EntityRef, RiskTier, utcnow
from fintwinos.demos import _fallbacks
from fintwinos.demos.stack import (
    DemoStack,
    as_plain,
    audit_summary,
    build_stack,
    call_tool,
    ci_named,
    extract_simulation,
    find_tool,
    fmt_ci,
    fmt_num,
    maybe_await,
    metric_named,
    normalize_case,
    render_audit_excerpt,
    render_case,
    render_header,
    schema_args,
)
from fintwinos.policy.gates import PolicyGate

CASE_ID = "case_liq"
OBJECTIVE = "rehearse a usd liquidity squeeze and propose contingency funding"
LADDER_SERIES = "treasury.cash_ladder.USD"
OPENING_BALANCE = 1000.0  # USD mm, matching the treasury simulator's default book

#: The two stress presets contrasted in the demo. ``shock_bps`` is the funding
#: spread shock applied on top of the observed squeeze; the bundled fallback
#: model maps it onto an outflow severity.
STRESS_PRESETS: list[dict[str, Any]] = [
    {
        "name": "moderate_outflow",
        "shock_bps": 75.0,
        "description": "Deposit attrition persists; funding spreads widen moderately",
    },
    {
        "name": "severe_squeeze",
        "shock_bps": 300.0,
        "description": "Wholesale funding seizes; spreads gap out across the curve",
    },
]

_LADDER_TOOLS = [
    "observe_cash_ladder",
    "observe_liquidity_ladder",
    "observe_*ladder*",
    "observe_*liquidity*",
]
_STRESS_TOOLS = ["simulate_liquidity_stress", "simulate_*liquidity*", "simulate_*funding*"]


async def arun(
    offline_ok: bool = True,
    console: Console | None = None,
    seed: int = 7,
    settings: Settings | None = None,
    stack: DemoStack | None = None,
) -> dict[str, Any]:
    """Async core of the demo; ``day_in_the_life`` composes demos through this."""
    stack = stack or build_stack(seed=seed, console=console, offline_ok=offline_ok, settings=settings)
    console = console or stack.console
    warnings: list[str] = []

    render_header(
        console,
        "USD liquidity squeeze",
        "A squeeze is unfolding in the twin's cash ladder. Observe it, stress it under "
        "two presets, and rehearse the contingency-funding decision — all audited.",
        stack.settings,
    )

    _stage_squeeze(stack, seed)
    ladder = await _observe_cash_ladder(stack, console, seed, warnings)
    stress = await _run_stress_presets(stack, console, seed, warnings)
    case = await _handle_liquidity_case(stack, warnings)

    render_case(console, case, title=f"Treasury decision · {CASE_ID}")
    audit = audit_summary(stack.registry.audit)
    render_audit_excerpt(console, audit["excerpt"])

    return {
        "demo": "liquidity",
        "offline": stack.llm.offline,
        "cash_ladder": ladder,
        "stress": stress,
        "case": case,
        "audit": audit,
        "warnings": warnings,
    }


def _stage_squeeze(stack: DemoStack, seed: int) -> None:
    """Stage the squeeze in the twin: a deteriorating USD net-flow ladder.

    Writes 30 days of escalating net outflows to the ``treasury.cash_ladder.USD``
    timeseries (the treasury simulator uses observed ladders as its flow
    baseline when present) and mirrors them as a ``cash_ladder`` graph entity
    so ``observe_cash_ladder`` reports the same picture. Idempotent per stack.
    """
    if stack.runtime.timeseries.latest(LADDER_SERIES) is not None:
        return
    rng = np.random.default_rng(seed + 3)
    days = np.arange(1, 31)
    net = -(35.0 + 4.5 * days) * (1.0 + 0.05 * rng.standard_normal(30))
    inflows = np.clip(net, 0.0, None) + 45.0 + 4.0 * rng.random(30)
    outflows = inflows - net

    now = utcnow()
    for value in net:
        stack.runtime.timeseries.append(LADDER_SERIES, now, float(value))
    stack.runtime.graph.upsert_entity(
        EntityRef(entity_type="cash_ladder", entity_id="USD"),
        {
            "currency": "USD",
            "opening_balance": OPENING_BALANCE,
            "buckets": [
                {
                    "day": int(day),
                    "inflow": round(float(inflows[i]), 2),
                    "outflow": round(float(outflows[i]), 2),
                }
                for i, day in enumerate(days)
            ],
        },
    )
    stack.registry.audit.append(
        "demo.liquidity",
        "scenario.staged",
        {
            "series": LADDER_SERIES,
            "days": len(days),
            "cumulative_net_30d": round(float(net.sum()), 2),
        },
    )


async def _observe_cash_ladder(
    stack: DemoStack, console: Console, seed: int, warnings: list[str]
) -> dict[str, Any]:
    """Observe the USD cash ladder through the registry, with a local fallback."""
    source = "fintwinos.demos._fallbacks.cash_ladder"
    rows: list[dict[str, Any]] | None = None
    tool = find_tool(stack.registry, _LADDER_TOOLS)
    if tool is not None:
        spec = stack.registry.spec(tool)
        result = await call_tool(
            stack,
            tool,
            schema_args(
                spec,
                {"currencies": ["USD"], "currency": "USD", "horizon_days": 14, "seed": seed},
            ),
            caller="demo.liquidity",
        )
        if result.ok:
            rows = _coerce_ladder(result.data)
            source = tool if rows else f"{tool} (unparsed payload; fallback rows)"
        else:
            warnings.append(f"{tool} failed: {result.error}")
    else:
        warnings.append("no observe_* cash-ladder tool registered; using bundled ladder")

    if rows is None:
        rows = _fallbacks.cash_ladder(seed=seed, horizon_days=14)

    table = Table(title="USD cash ladder (mm)", title_justify="left", border_style="blue")
    for column in ("day", "inflows", "outflows", "net", "closing cash"):
        table.add_column(column, justify="right")
    for row in rows[:7]:
        net = row.get("net")
        net_style = "red" if isinstance(net, int | float) and net < 0 else "green"
        table.add_row(
            str(row.get("day", "—")),
            fmt_num(row.get("inflows")),
            fmt_num(row.get("outflows")),
            f"[{net_style}]{fmt_num(net)}[/{net_style}]",
            fmt_num(row.get("closing_cash")),
        )
    if len(rows) > 7:
        table.add_row("…", "…", "…", "…", "…")
    console.print(table)
    return {"source": source, "rows": rows}


def _coerce_ladder(data: Any) -> list[dict[str, Any]] | None:
    """Normalise an observe payload into canonical ladder rows.

    Accepts both the platform shape (``{"ladders": [{"ladder": rows}]}`` with
    ``inflow``/``outflow``/``closing_balance`` keys) and flat ``rows``/``ladder``
    lists; output rows always carry ``day``/``inflows``/``outflows``/``net``/
    ``closing_cash``.
    """
    plain = as_plain(data)
    candidate: Any = plain
    if isinstance(plain, dict):
        ladders = plain.get("ladders")
        if isinstance(ladders, list) and ladders and isinstance(ladders[0], dict):
            candidate = ladders[0].get("ladder")
        else:
            for key in ("ladder", "rows", "buckets", "days"):
                if isinstance(plain.get(key), list):
                    candidate = plain[key]
                    break
    if not (isinstance(candidate, list) and candidate and all(isinstance(r, dict) for r in candidate)):
        return None
    normalised: list[dict[str, Any]] = []
    for row in candidate:
        normalised.append(
            {
                "day": row.get("day"),
                "inflows": row.get("inflows", row.get("inflow")),
                "outflows": row.get("outflows", row.get("outflow")),
                "net": row.get("net"),
                "closing_cash": row.get("closing_cash", row.get("closing_balance")),
            }
        )
    return normalised


async def _run_stress_presets(
    stack: DemoStack, console: Console, seed: int, warnings: list[str]
) -> dict[str, Any]:
    """Run both stress presets via the registry (or the bundled Monte Carlo)."""
    tool = find_tool(stack.registry, _STRESS_TOOLS)
    if tool is None:
        warnings.append(
            "no simulate_liquidity_stress tool registered; using bundled Monte Carlo model"
        )

    stress: dict[str, Any] = {}
    for preset in STRESS_PRESETS:
        sim: dict[str, Any] | None = None
        source = "fintwinos.demos._fallbacks.liquidity_stress_model"
        if tool is not None:
            spec = stack.registry.spec(tool)
            result = await call_tool(
                stack,
                tool,
                schema_args(
                    spec,
                    {
                        "stress_name": preset["name"],
                        "scenario": preset["name"],
                        "shock_bps": preset["shock_bps"],
                        "severity": min(preset["shock_bps"] / 400.0, 1.0),
                        "horizon_days": 30,
                        "currencies": ["USD"],
                        "currency": "USD",
                        "assumptions_version": "v1",
                        "ticket_id": "TREAS-2031",
                        "seed": seed,
                    },
                ),
                caller="demo.liquidity",
                ticket_id="TREAS-2031",
            )
            if result.ok:
                candidate = extract_simulation(result.data)
                if candidate["metrics"]:
                    sim, source = candidate, tool
            else:
                warnings.append(f"{tool}[{preset['name']}] failed: {result.error}")
        if sim is None:
            sim = extract_simulation(
                _fallbacks.liquidity_stress_model(
                    severity=min(preset["shock_bps"] / 400.0, 1.0),
                    seed=seed,
                    scenario_name=preset["name"],
                )
            )
        stress[preset["name"]] = {
            "source": source,
            "description": preset["description"],
            "shock_bps": preset["shock_bps"],
            "survival_days": metric_named(sim, "survival"),
            "ci": ci_named(sim, "survival"),
            "breach_probability": metric_named(sim, "breach", "shortfall"),
            "peak_funding_cost_bps": metric_named(sim, "funding_cost", "peak_funding"),
            "metrics": sim["metrics"],
            "calibration": sim["calibration"],
        }

    table = Table(
        title="Liquidity stress — survival horizon", title_justify="left", border_style="red"
    )
    table.add_column("preset", style="bold")
    table.add_column("survival days (p50)", justify="right")
    table.add_column("90% CI", justify="right")
    table.add_column("peak funding cost", justify="right")
    table.add_column("source", style="dim")
    for name, row in stress.items():
        cost = row["peak_funding_cost_bps"]
        cost_cell = f"{cost:,.0f} bps" if cost is not None else (
            f"breach p={row['breach_probability']:.0%}" if row["breach_probability"] is not None else "—"
        )
        table.add_row(
            name,
            fmt_num(row["survival_days"], 1),
            fmt_ci(row["ci"]),
            cost_cell,
            str(row["source"]),
        )
    console.print(table)
    return stress


async def _handle_liquidity_case(stack: DemoStack, warnings: list[str]) -> dict[str, Any]:
    """Route the rehearsal through ``agents.runtime.handle_case`` (with fallback)."""
    try:
        from fintwinos.agents.runtime import handle_case

        raw = await maybe_await(
            handle_case(CASE_ID, OBJECTIVE, stack.runtime, stack.registry, stack.llm)
        )
        return normalize_case(raw)
    except Exception as exc:  # noqa: BLE001 — a demo must degrade, never crash
        warnings.append(f"agents.runtime.handle_case unavailable ({exc}); using fallback planner")
        return _fallback_case(stack)


def _fallback_case(stack: DemoStack) -> dict[str, Any]:
    """Deterministic propose-only contingency-funding decision."""
    decision = Decision(
        objective=OBJECTIVE,
        action_type="propose_only",
        planned_tool_calls=[
            {
                "tool": "propose_funding_plan",
                "arguments": {"currency": "USD", "horizon_days": 14, "buffer_ratio": 0.1},
            }
        ],
        rationale=(
            "Severe-squeeze rehearsal shows a finite survival horizon; staging the CFP "
            "tranches now keeps every option reversible while the treasurer reviews."
        ),
        risk_tier=RiskTier.high,
        owner="demo.liquidity.fallback_planner",
    )
    gate = stack.registry.policy_gate or PolicyGate()
    verdict = gate.check_decision(decision)
    stack.registry.audit.append(
        "demo.liquidity",
        "decision.proposed",
        {
            "case_id": CASE_ID,
            "decision_id": decision.decision_id,
            "action_type": decision.action_type,
            "allowed": verdict.allowed,
            "requires_human_review": verdict.requires_human_review,
        },
    )
    return {
        "status": "awaiting_human" if verdict.requires_human_review else "complete",
        "decision": decision.model_dump(mode="json"),
        "policy": verdict.model_dump(),
        "case_id": CASE_ID,
    }


def run(
    offline_ok: bool = True,
    console: Console | None = None,
    seed: int = 7,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the liquidity demo end-to-end and return structured results.

    Returns a dict with keys ``demo``, ``offline``, ``cash_ladder``, ``stress``
    (one entry per preset with ``survival_days``, ``ci`` and funding-cost /
    breach severity indicators), ``case`` (``status`` / ``decision`` /
    ``policy``), ``audit`` and ``warnings``.
    """
    return asyncio.run(arun(offline_ok=offline_ok, console=console, seed=seed, settings=settings))


def main() -> None:
    """Contracted CLI entry point: ``fintwinos demo liquidity``."""
    run()
