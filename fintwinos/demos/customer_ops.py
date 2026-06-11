"""Demo: complaints spike — staffing counterfactual and a propose-only fix.

Flow: observe the live queue state through the registry → rehearse the spike
on the calibrated customer-operations simulator (a contracted ``twin_sim``
entry point) at current staffing and with two extra handlers, using the same
master seed so the comparison is fair → lay out the SLA breach table →
package the recommended routing change as a **propose-only**
:class:`~fintwinos.core.types.Decision` and show the policy gate's verdict.
Nothing executes: the twin rehearses, humans decide.

The counterfactual drives ``runtime.simulators`` directly (demos are a
contracted consumer of ``twin_sim``) because the staffing scenario needs the
arrival-rate spike lever, which the current ``simulate_*`` tool schemas do not
expose. Every comparison is still appended to the shared audit trail, and a
bundled fluid-queue Monte Carlo stands in when the simulator is absent.

Run with ``fintwinos demo customer_ops``. Fully offline-capable.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.table import Table

from fintwinos.core.config import Settings
from fintwinos.core.types import Decision, RiskTier, Scenario
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
    fmt_num,
    fmt_pct,
    metric_named,
    render_audit_excerpt,
    render_case,
    render_header,
    schema_args,
)
from fintwinos.models.llm_routing.router import TaskClass
from fintwinos.policy.gates import PolicyGate

BASELINE_AGENTS = 8
EXTRA_AGENTS = 2
ARRIVAL_RATE_PER_HR = 96.0  # the spike: +60% over the 60/hr business-as-usual rate
HORIZON_HOURS = 8
SLA_MINUTES = 15.0

_QUEUE_STATE_TOOLS = ["observe_queue_state", "observe_*queue*"]
_ROUTING_TOOLS = ["propose_routing_change", "propose_*routing*"]
_SIMULATOR_NAMES = ["customer_ops", "ops_queue", "service_queue", "queue"]


async def arun(
    offline_ok: bool = True,
    console: Console | None = None,
    seed: int = 7,
    settings: Settings | None = None,
    stack: DemoStack | None = None,
) -> dict[str, Any]:
    """Async core of the customer operations demo."""
    stack = stack or build_stack(seed=seed, console=console, offline_ok=offline_ok, settings=settings)
    console = console or stack.console
    warnings: list[str] = []

    render_header(
        console,
        "Complaints spike",
        f"Complaint volume is running at {ARRIVAL_RATE_PER_HR:.0f}/hr (+60%). Rehearse the "
        f"queue at current staffing and with +{EXTRA_AGENTS} handlers, then propose — never "
        "execute — the fix.",
        stack.settings,
    )

    queue_state = await _observe_queue_state(stack, warnings)
    baseline = _simulate_queue(stack, BASELINE_AGENTS, seed, warnings)
    counterfactual = _simulate_queue(stack, BASELINE_AGENTS + EXTRA_AGENTS, seed, warnings)
    stack.registry.audit.append(
        "demo.customer_ops",
        "counterfactual.compared",
        {
            "baseline_agents": BASELINE_AGENTS,
            "counterfactual_agents": BASELINE_AGENTS + EXTRA_AGENTS,
            "seed": seed,
            "source": baseline["source"],
        },
    )
    sla_table = _sla_breach_table(baseline, counterfactual)
    _render_sla_table(console, sla_table)

    decision, policy, proposal = await _propose_routing_change(
        stack, baseline, counterfactual, warnings
    )
    render_case(
        console,
        {"status": "awaiting_human" if policy["requires_human_review"] else "complete",
         "decision": decision, "policy": policy},
        title="Routing change · propose-only",
    )

    customer_note = await _draft_customer_note(stack, baseline, counterfactual, warnings)
    console.print(f"[bold]Draft customer holding note[/bold] "
                  f"([dim]{customer_note['source']}[/dim]): {customer_note['text']}")

    audit = audit_summary(stack.registry.audit)
    render_audit_excerpt(console, audit["excerpt"])

    return {
        "demo": "customer_ops",
        "offline": stack.llm.offline,
        "queue_state": queue_state,
        "queue": {
            "source": baseline["source"],
            "baseline": baseline,
            "plus_two_staff": counterfactual,
        },
        "sla_table": sla_table,
        "decision": decision,
        "policy": policy,
        "proposal_tool": proposal,
        "customer_note": customer_note,
        "audit": audit,
        "warnings": warnings,
    }


async def _draft_customer_note(
    stack: DemoStack,
    baseline: dict[str, Any],
    counterfactual: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Draft a compliant customer holding note — deterministic text, LLM-polished online.

    The LLM is enrichment only: any API failure degrades to the rule-based
    draft with a recorded warning, never a crash.
    """
    base_wait = baseline.get("metrics", {}).get("avg_wait_hours")
    cf_wait = counterfactual.get("metrics", {}).get("avg_wait_hours")
    draft = (
        "We are handling an unusually high volume of complaints and current responses are "
        f"taking around {base_wait:.0f} hours on average. We have proposed additional staffing "
        f"that our simulations indicate would bring this down to roughly {cf_wait:.0f} hours. "
        "Your complaint remains in the queue and you do not need to contact us again."
        if isinstance(base_wait, int | float) and isinstance(cf_wait, int | float)
        else "We are handling an unusually high volume of complaints; your case remains in the "
             "queue and you do not need to contact us again."
    )
    if stack.llm.offline:
        return {"text": draft, "source": "rule-based"}
    try:
        response = await stack.llm.complete(
            [
                {
                    "role": "system",
                    "content": "You polish customer communications for a regulated financial "
                    "firm. Rewrite the note in a warm, plain-English tone. Keep every figure "
                    "unchanged, make no promises beyond the draft, two sentences maximum.",
                },
                {"role": "user", "content": draft},
            ],
            task=TaskClass.drafting,
        )
    except Exception as exc:
        warnings.append(f"LLM customer-note polish unavailable: {type(exc).__name__}: {exc}")
        return {"text": draft, "source": "rule-based (LLM unavailable)"}
    if response.text.strip() and not response.offline:
        return {"text": response.text.strip(), "source": f"llm ({response.model})"}
    return {"text": draft, "source": "rule-based"}


async def _observe_queue_state(stack: DemoStack, warnings: list[str]) -> dict[str, Any]:
    """Observe the live queue board through the registry, if the tool exists."""
    tool = find_tool(stack.registry, _QUEUE_STATE_TOOLS)
    if tool is None:
        return {"source": None, "queues": []}
    spec = stack.registry.spec(tool)
    result = await call_tool(stack, tool, schema_args(spec, {}), caller="demo.customer_ops")
    if not result.ok:
        warnings.append(f"{tool} failed: {result.error}")
        return {"source": tool, "queues": []}
    data = as_plain(result.data) or {}
    queues = data.get("queues", []) if isinstance(data, dict) else []
    return {"source": tool, "queues": queues}


def _simulate_queue(
    stack: DemoStack, agents: int, seed: int, warnings: list[str]
) -> dict[str, Any]:
    """One staffing scenario on the platform queue simulator (or bundled model).

    The same master ``seed`` is used for every staffing level, so scenarios are
    compared under common random numbers.
    """
    sim: dict[str, Any] | None = None
    source = "fintwinos.demos._fallbacks.queue_simulation"
    simulator = _find_simulator(stack)
    if simulator is not None:
        scenario = Scenario(
            name=f"complaints_spike_agents_{agents}",
            kind="stress",
            params={
                "arrival_rate_per_hr": ARRIVAL_RATE_PER_HR,
                "n_agents": agents,
                "horizon_hours": HORIZON_HOURS,
                "sla_minutes": SLA_MINUTES,
            },
        )
        try:
            result = simulator.run(scenario, seed=seed)
            sim = extract_simulation(result)
            source = f"runtime.simulators['{simulator.name}']"
        except Exception as exc:  # noqa: BLE001 — degrade to the bundled model
            warnings.append(f"queue simulator failed ({exc}); using bundled fluid model")
    else:
        warnings.append("no queue simulator registered; using bundled fluid-queue model")

    if sim is None:
        sim = extract_simulation(
            _fallbacks.queue_simulation(
                staff=agents,
                arrival_multiplier=ARRIVAL_RATE_PER_HR / 60.0,
                horizon_days=10,
                sla_hours=24.0,
                seed=seed,
            )
        )
    return {
        "source": source,
        "staff": agents,
        "sla_breach_rate": metric_named(sim, "sla_breach", "breach", "sla"),
        "avg_wait": metric_named(sim, "p90_wait", "avg_wait", "wait"),
        "ci": {
            "sla_breach_rate": ci_named(sim, "sla_breach", "breach", "sla"),
            "avg_wait": ci_named(sim, "p90_wait", "avg_wait", "wait"),
        },
        "metrics": sim["metrics"],
        "confidence": sim["confidence"],
    }


def _find_simulator(stack: DemoStack) -> Any | None:
    """Locate the customer-operations simulator on the runtime by name."""
    simulators = getattr(stack.runtime, "simulators", {}) or {}
    for name in _SIMULATOR_NAMES:
        if name in simulators:
            return simulators[name]
    for name, simulator in sorted(simulators.items()):
        if "queue" in name or "ops" in name:
            return simulator
    return None


def _sla_breach_table(
    baseline: dict[str, Any], counterfactual: dict[str, Any]
) -> list[dict[str, Any]]:
    """Structured baseline vs counterfactual rows over the shared metric set."""
    shared = [
        name
        for name in baseline["metrics"]
        if name in counterfactual["metrics"]
        and isinstance(baseline["metrics"][name], int | float)
    ]
    rows: list[dict[str, Any]] = []
    for name in sorted(shared):
        base, cf = float(baseline["metrics"][name]), float(counterfactual["metrics"][name])
        rows.append(
            {
                "metric": name,
                "label": name.replace("_", " "),
                "baseline": base,
                "plus_two": cf,
                "delta": round(cf - base, 6),
            }
        )
    return rows


def _render_sla_table(console: Console, rows: list[dict[str, Any]]) -> None:
    table = Table(
        title=f"Complaints queue under the spike — baseline ({BASELINE_AGENTS} agents) vs "
        f"+{EXTRA_AGENTS}",
        title_justify="left",
        border_style="blue",
    )
    table.add_column("metric", style="bold")
    table.add_column(f"baseline ({BASELINE_AGENTS})", justify="right")
    table.add_column(f"+{EXTRA_AGENTS} agents ({BASELINE_AGENTS + EXTRA_AGENTS})", justify="right")
    table.add_column("delta", justify="right")
    for row in rows:
        is_rate = "rate" in row["metric"]
        fmt = fmt_pct if is_rate else fmt_num
        delta = row["delta"]
        if delta is None:
            delta_cell = "—"
        else:
            style = "green" if delta <= 0 else "red"
            delta_cell = f"[{style}]{fmt(delta)}[/{style}]"
        table.add_row(row["label"], fmt(row["baseline"]), fmt(row["plus_two"]), delta_cell)
    console.print(table)


async def _propose_routing_change(
    stack: DemoStack,
    baseline: dict[str, Any],
    counterfactual: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Package the staffing/routing fix as a propose-only decision."""
    base_breach = baseline.get("sla_breach_rate")
    cf_breach = counterfactual.get("sla_breach_rate")
    breach_phrase = (
        f"SLA breaches {base_breach:.0%} → {cf_breach:.0%}"
        if isinstance(base_breach, int | float) and isinstance(cf_breach, int | float)
        else "SLA breach rate improves"
    )

    proposal_args = {
        "queue": "complaints",
        "add_staff": EXTRA_AGENTS,
        "reroute_overflow_to": "digital_servicing",
        "effective": "next_rota",
        "sla_minutes": SLA_MINUTES,
    }
    decision = Decision(
        objective="restore the complaints SLA during the volume spike",
        action_type="propose_only",
        planned_tool_calls=[{"tool": "propose_routing_change", "arguments": proposal_args}],
        rationale=(
            f"Counterfactual with +{EXTRA_AGENTS} handlers (common random numbers): "
            f"{breach_phrase}. Routing overflow to digital servicing absorbs residual peaks. "
            "No customer-facing change executes without operations sign-off."
        ),
        risk_tier=RiskTier.medium,
        owner="demo.customer_ops.planner",
    )

    proposal_result: dict[str, Any] | None = None
    tool = find_tool(stack.registry, _ROUTING_TOOLS)
    if tool is not None:
        spec = stack.registry.spec(tool)
        result = await call_tool(
            stack, tool, schema_args(spec, proposal_args), caller="demo.customer_ops"
        )
        if result.ok:
            proposal_result = {"tool": tool, "data": as_plain(result.data)}
        else:
            warnings.append(f"{tool} failed: {result.error}")

    gate = stack.registry.policy_gate or PolicyGate()
    verdict = gate.check_decision(decision)
    stack.registry.audit.append(
        "demo.customer_ops",
        "decision.proposed",
        {
            "decision_id": decision.decision_id,
            "action_type": decision.action_type,
            "allowed": verdict.allowed,
            "requires_human_review": verdict.requires_human_review,
        },
    )
    return decision.model_dump(mode="json"), verdict.model_dump(), proposal_result


def run(
    offline_ok: bool = True,
    console: Console | None = None,
    seed: int = 7,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the customer ops demo end-to-end and return structured results.

    Returns a dict with keys ``demo``, ``queue_state``, ``queue`` (``baseline``
    and ``plus_two_staff`` scenarios with metrics and CIs), ``sla_table``,
    ``decision`` (propose-only), ``policy`` (gate verdict), ``proposal_tool``,
    ``audit`` and ``warnings``.
    """
    return asyncio.run(arun(offline_ok=offline_ok, console=console, seed=seed, settings=settings))


def main() -> None:
    """Contracted CLI entry point: ``fintwinos demo customer_ops``."""
    run()
