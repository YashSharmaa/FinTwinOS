"""Demo: AML ring surge, triage, narrative, and a governed case closure.

Flow: run the compliance simulator for a laundering-ring surge → score the
alert queue with the AML scorer (preferring ``fintwinos.models.graph``,
trained on synthetic labelled data) → show the precision/recall trade-off of
different triage depths → draft a case narrative for the top alert via
``propose_case_narrative`` → then demonstrate *both* sides of the execute
band: ``execute_close_case`` is refused without a human approval, and goes
through, writing to the local outbox, once an :class:`ApprovalToken` is
granted, the policy pack carries an explicit allow rule, and
``execute_tools_enabled`` is switched on.

Run with ``fintwinos demo aml_triage``. Fully offline-capable.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fintwinos.core.config import Settings
from fintwinos.core.types import (
    ApprovalStatus,
    EntityRef,
    RiskTier,
    SideEffectClass,
    ToolBand,
    utcnow,
)
from fintwinos.demos import _fallbacks
from fintwinos.demos.stack import (
    DemoStack,
    as_plain,
    audit_summary,
    build_stack,
    call_tool,
    ensure_allow_rule,
    extract_simulation,
    find_tool,
    fmt_pct,
    render_audit_excerpt,
    render_header,
    schema_args,
)
from fintwinos.models.llm_routing.router import TaskClass
from fintwinos.policy.approvals import ApprovalWorkflow
from fintwinos.tools.registry import CallContext, ToolSpec

CASE_ID = "case_0001"  # the open aml_alert case shipped with the demo twin
TRIAGE_DEPTHS = [5, 10, 15, 20, 30]

_SURGE_TOOLS = [
    "simulate_alert_threshold",
    "simulate_aml_ring",
    "simulate_compliance_alerts",
    "simulate_*aml*",
    "simulate_*ring*",
    "simulate_*compliance*",
    "simulate_*alert*",
]
_ALERT_TOOLS = ["observe_alerts", "observe_*alert*"]
_NARRATIVE_TOOLS = ["propose_case_narrative", "propose_*narrative*"]
CLOSE_TOOL = "execute_close_case"


async def arun(
    offline_ok: bool = True,
    console: Console | None = None,
    seed: int = 7,
    settings: Settings | None = None,
    stack: DemoStack | None = None,
) -> dict[str, Any]:
    """Async core of the AML triage demo."""
    stack = stack or build_stack(seed=seed, console=console, offline_ok=offline_ok, settings=settings)
    console = console or stack.console
    warnings: list[str] = []

    render_header(
        console,
        "AML ring surge",
        "A laundering ring lights up the alert queue. Triage it with a trained scorer, "
        "draft the case narrative, then close the case, but only with a human approval.",
        stack.settings,
    )

    _ensure_case(stack)
    twin_alerts = await _observe_twin_alerts(stack, warnings)
    surge, queue = await _run_surge(stack, seed, warnings)
    alerts, X_queue, labels = queue

    # Train the scorer on a synthetic labelled population, score the live queue.
    _, X_train, y_train = _fallbacks.make_alert_population(
        400, 0.22, np.random.default_rng(seed + 12)
    )
    scores, scorer_source = _fallbacks.train_and_score(X_train, y_train, X_queue)
    auc = _fallbacks.auc_score(scores, labels)
    triage = _fallbacks.precision_recall_at_k(scores, labels, TRIAGE_DEPTHS)
    _render_queue(console, alerts, scores, labels)
    _render_triage(console, triage, auc, scorer_source)

    top_idx = int(np.argmax(scores))
    top_alert = {**alerts[top_idx], "score": round(float(scores[top_idx]), 4)}
    narrative = await _case_narrative(stack, console, top_alert, surge, warnings)
    close_case = await _close_case_flow(stack, console, narrative, warnings)

    audit = audit_summary(stack.registry.audit)
    render_audit_excerpt(console, audit["excerpt"])

    return {
        "demo": "aml_triage",
        "offline": stack.llm.offline,
        "surge": surge,
        "twin_alerts": twin_alerts,
        "queue_size": len(alerts),
        "scorer": {"source": scorer_source, "auc": round(float(auc), 4)},
        "triage": triage,
        "top_alert": top_alert,
        "narrative": narrative,
        "close_case": close_case,
        "audit": audit,
        "warnings": warnings,
    }


def _ensure_case(stack: DemoStack) -> None:
    """Make sure the demo case exists in the twin graph (idempotent staging)."""
    case_ref = EntityRef(entity_type="case", entity_id=CASE_ID)
    if stack.runtime.graph.get_entity(case_ref) is not None:
        return
    stack.runtime.graph.upsert_entity(
        case_ref,
        {
            "case_id": CASE_ID,
            "kind": "aml_alert",
            "status": "open",
            "priority": "high",
            "opened_at": utcnow().isoformat(),
        },
    )
    stack.registry.audit.append("demo.aml_triage", "case.staged", {"case_id": CASE_ID})


async def _observe_twin_alerts(stack: DemoStack, warnings: list[str]) -> dict[str, Any]:
    """Pull the twin's live alert backlog through the observe band, if available."""
    tool = find_tool(stack.registry, _ALERT_TOOLS)
    if tool is None:
        return {"source": None, "count": 0, "alerts": []}
    spec = stack.registry.spec(tool)
    result = await call_tool(
        stack, tool, schema_args(spec, {"limit": 10}), caller="demo.aml_triage"
    )
    if not result.ok:
        warnings.append(f"{tool} failed: {result.error}")
        return {"source": tool, "count": 0, "alerts": []}
    data = as_plain(result.data) or {}
    rows = data.get("alerts", []) if isinstance(data, dict) else []
    return {
        "source": tool,
        "count": int(data.get("count", len(rows))) if isinstance(data, dict) else len(rows),
        "alerts": rows[:10],
    }


async def _run_surge(
    stack: DemoStack, seed: int, warnings: list[str]
) -> tuple[dict[str, Any], tuple[list[dict[str, Any]], np.ndarray, np.ndarray]]:
    """Run the compliance simulator and obtain a labelled alert queue."""
    surge: dict[str, Any] = {
        "source": "fintwinos.demos._fallbacks.make_alert_population",
        "metrics": {},
        "confidence": {},
    }
    queue: tuple[list[dict[str, Any]], np.ndarray, np.ndarray] | None = None

    tool = find_tool(stack.registry, _SURGE_TOOLS)
    if tool is not None:
        spec = stack.registry.spec(tool)
        result = await call_tool(
            stack,
            tool,
            schema_args(
                spec,
                {
                    "scenario": "ring_surge",
                    "threshold": 0.72,
                    "n_alerts": 48,
                    "horizon_days": 7,
                    "lookback_days": 90,
                    "seed": seed,
                },
            ),
            caller="demo.aml_triage",
        )
        if result.ok:
            sim = extract_simulation(result.data)
            surge = {
                "source": tool,
                "metrics": sim["metrics"],
                "confidence": sim["confidence"],
            }
            queue = _fallbacks.coerce_alert_queue(result.data)
            if queue is None:
                warnings.append(
                    f"{tool} returned no labelled alert queue; generating a deterministic one"
                )
        else:
            warnings.append(f"{tool} failed: {result.error}")
    else:
        warnings.append("no compliance simulator tool registered; using bundled surge generator")

    if queue is None:
        queue = _fallbacks.make_alert_population(48, 0.25, np.random.default_rng(seed + 11))
        if not surge["metrics"]:
            labels = queue[2]
            surge["metrics"] = {
                "n_alerts": float(len(labels)),
                "ring_alerts": float(int(labels.sum())),
                "ring_rate": round(float(labels.mean()), 4),
            }
    return surge, queue


def _render_queue(
    console: Console, alerts: list[dict[str, Any]], scores: np.ndarray, labels: np.ndarray
) -> None:
    """Show the top of the scored queue (ground truth included, it's a rehearsal)."""
    order = np.argsort(-scores, kind="stable")
    table = Table(
        title=f"Scored alert queue, top 8 of {len(alerts)}",
        title_justify="left",
        border_style="yellow",
    )
    table.add_column("rank", justify="right", style="dim")
    table.add_column("alert", style="bold")
    table.add_column("score", justify="right")
    table.add_column("fan-in", justify="right")
    table.add_column("structuring", justify="right")
    table.add_column("cycle", justify="center")
    table.add_column("ground truth", justify="center")
    for rank, idx in enumerate(order[:8], start=1):
        alert = alerts[int(idx)]
        feats = alert["features"]
        truth = "[red]ring[/red]" if labels[int(idx)] == 1 else "[dim]benign[/dim]"
        table.add_row(
            str(rank),
            alert["alert_id"],
            f"{float(scores[int(idx)]):.3f}",
            f"{feats['fan_in']:.0f}",
            f"{feats['structuring_score']:.2f}",
            "✓" if feats["cycle_participation"] >= 0.5 else "·",
            truth,
        )
    console.print(table)


def _render_triage(
    console: Console, triage: list[dict[str, float]], auc: float, scorer_source: str
) -> None:
    """Precision/recall trade-off table for the triage depth decision."""
    table = Table(
        title=f"Triage depth trade-off, scorer: {scorer_source} (AUC {auc:.3f})",
        title_justify="left",
        border_style="yellow",
    )
    table.add_column("investigate top-k", justify="right", style="bold")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("reading", style="dim")
    for row in triage:
        k = int(row["k"])
        reading = (
            "tight queue, misses ring members" if row["recall"] < 0.6
            else "balanced" if row["precision"] >= 0.5
            else "wide net, analyst hours burn"
        )
        table.add_row(str(k), fmt_pct(row["precision"]), fmt_pct(row["recall"]), reading)
    console.print(table)


async def _case_narrative(
    stack: DemoStack,
    console: Console,
    top_alert: dict[str, Any],
    surge: dict[str, Any],
    warnings: list[str],
) -> str:
    """Draft the case narrative for the top alert via the propose band."""
    narrative: str | None = None
    tool = find_tool(stack.registry, _NARRATIVE_TOOLS)
    if tool is not None:
        spec = stack.registry.spec(tool)
        result = await call_tool(
            stack,
            tool,
            schema_args(
                spec,
                {
                    "case_id": CASE_ID,
                    "alert_id": top_alert["alert_id"],
                    "kind": "aml_alert",
                    "summary": _feature_summary(top_alert),
                },
            ),
            caller="demo.aml_triage",
        )
        if result.ok:
            data = result.data
            if isinstance(data, dict) and isinstance(data.get("narrative"), str):
                narrative = data["narrative"]
            elif isinstance(data, str):
                narrative = data
        else:
            warnings.append(f"{tool} failed: {result.error}")
    else:
        warnings.append("no propose_case_narrative tool registered; drafting locally")

    if not narrative:
        narrative = _fallback_narrative(top_alert, surge)

    if not stack.llm.offline:
        try:
            response = await stack.llm.complete(
                [
                    {
                        "role": "system",
                        "content": "You are an AML investigator. Tighten this draft case narrative "
                        "into 3 crisp sentences. Keep every figure unchanged.",
                    },
                    {"role": "user", "content": narrative},
                ],
                task=TaskClass.drafting,
            )
        except Exception as exc:  # LLM polish is enrichment, never a dependency
            warnings.append(f"LLM narrative polish unavailable: {type(exc).__name__}: {exc}")
        else:
            if response.text.strip() and not response.offline:
                narrative = response.text.strip()

    console.print(
        Panel(
            narrative,
            title=f"[bold]Proposed case narrative · {CASE_ID} "
            f"(top alert {top_alert['alert_id']})[/bold]",
            border_style="yellow",
        )
    )
    return narrative


def _feature_summary(alert: dict[str, Any]) -> str:
    feats = alert["features"]
    return (
        f"fan_in={feats['fan_in']:.0f}, fan_out={feats['fan_out']:.0f}, "
        f"structuring={feats['structuring_score']:.2f}, "
        f"cycle={'yes' if feats['cycle_participation'] >= 0.5 else 'no'}, "
        f"velocity={feats['velocity']:.2f}"
    )


def _fallback_narrative(alert: dict[str, Any], surge: dict[str, Any]) -> str:
    """Deterministic template narrative grounded in the alert's features."""
    feats = alert["features"]
    ring_rate = surge.get("metrics", {}).get("ring_rate")
    context = (
        f" The alert arrived during a surge in which an estimated {ring_rate:.0%} of new alerts "
        "share ring characteristics." if isinstance(ring_rate, int | float) else ""
    )
    return (
        f"Alert {alert['alert_id']} (customer {alert['customer_id']}) sits at the centre of a "
        f"suspected layering ring: {feats['fan_in']:.0f} inbound and {feats['fan_out']:.0f} "
        f"outbound counterparties transacted within 72 hours, with a structuring score of "
        f"{feats['structuring_score']:.2f} (transfers clustered just below the reporting "
        f"threshold) and {'confirmed' if feats['cycle_participation'] >= 0.5 else 'no'} "
        f"participation in a closed transaction cycle. Funds velocity is "
        f"{feats['velocity']:.1f}x the segment baseline"
        f"{', following a dormancy break' if feats['dormancy_break'] >= 0.5 else ''}"
        f"{', with cross-border legs' if feats['cross_border'] >= 0.5 else ''}.{context} "
        f"Model score {alert.get('score', 0):.2f}. Recommended action: escalate to a case, "
        "restrain outbound payments pending review, and prepare a suspicious activity report "
        "for MLRO sign-off."
    )


async def _close_case_flow(
    stack: DemoStack, console: Console, narrative: str, warnings: list[str]
) -> dict[str, Any]:
    """Demonstrate the execute band: refusal without approval, success with it."""
    if CLOSE_TOOL not in stack.registry:
        warnings.append(f"{CLOSE_TOOL} not in catalog; registering the demo reference tool")
        _register_demo_close_tool(stack)
    spec = stack.registry.spec(CLOSE_TOOL)
    closure_note = (
        f"Ring-pattern indicators corroborated by the triage scorer; SAR preparation "
        f"recommended. Narrative excerpt: {narrative[:240]}"
    )
    close_args = schema_args(
        spec,
        {
            "case_id": CASE_ID,
            "disposition": "escalated",
            "closure_note": closure_note,
            "ticket_id": "FCC-2031",
            "resolution": "escalated",
            "narrative": narrative[:280],
        },
    )

    previous_enabled = stack.registry.settings.execute_tools_enabled
    previous_shadow = stack.registry.settings.shadow_mode
    try:
        # --- Branch 1: no approval, execute band disabled → refusal. -------------
        stack.registry.settings.execute_tools_enabled = False
        refused = await stack.registry.call(
            CLOSE_TOOL,
            close_args,
            CallContext(caller="demo.aml_triage", extra={"role": "operator"}),
        )
        console.print(
            Panel(
                f"[red]refused[/red], {refused.error}\n"
                f"requires_approval={refused.requires_approval}",
                title="[bold]execute_close_case · attempt without approval[/bold]",
                border_style="red",
            )
        )

        # --- Branch 2: allow rule + maker-checker approvals + execute switch. -----
        # The token is minted through the real ApprovalWorkflow: the requester can
        # never self-approve, and the high-risk tier needs two distinct approvers
        # under dual control, exactly what an adopting institution would run.
        ensure_allow_rule(stack.registry.policy_gate, CLOSE_TOOL, "demo-allow-close-case")
        stack.registry.settings.execute_tools_enabled = True
        stack.registry.settings.shadow_mode = False
        console.print(
            "[dim]demo: FINTWIN_SHADOW_MODE disabled for this branch to demonstrate the "
            "real (outbox-only) write path; production deployments keep it on.[/dim]"
        )
        workflow = ApprovalWorkflow(audit=stack.registry.audit, settings=stack.settings)
        request = workflow.request(
            spec, requested_by="demo.aml_triage", reason=f"close case {CASE_ID}",
            scope={"case_id": CASE_ID, "ticket": "FCC-2031"},
        )
        workflow.approve(request.request_id, approver="mlro.on.duty", role="approver")
        if request.status != ApprovalStatus.approved:
            workflow.approve(request.request_id, approver="deputy.mlro", role="approver")
        token = workflow.tokens(request.request_id)[0]
        approved = await stack.registry.call(
            CLOSE_TOOL,
            close_args,
            CallContext(
                caller="demo.aml_triage", ticket_id="FCC-2031", approval=token,
                extra={"role": "operator"},
            ),
        )
    finally:
        stack.registry.settings.execute_tools_enabled = previous_enabled
        stack.registry.settings.shadow_mode = previous_shadow

    outbox_dir = Path(stack.settings.data_dir) / "outbox"
    outbox_files = sorted(str(p) for p in outbox_dir.glob("*")) if outbox_dir.exists() else []
    status = "[green]executed[/green]" if approved.ok else f"[red]failed[/red], {approved.error}"
    console.print(
        Panel(
            f"{status}\napproval: {token.token_id} granted by {token.granted_by} ({token.role})\n"
            f"outbox: {outbox_dir} ({len(outbox_files)} file(s))",
            title="[bold]execute_close_case · with granted ApprovalToken[/bold]",
            border_style="green" if approved.ok else "red",
        )
    )
    if not approved.ok:
        warnings.append(f"approved execute path failed: {approved.error}")

    return {
        "blocked": {
            "ok": refused.ok,
            "requires_approval": refused.requires_approval,
            "error": refused.error,
        },
        "approved": {
            "ok": approved.ok,
            "error": approved.error,
            "approval_token": token.token_id,
            "data": as_plain(approved.data),
        },
        "outbox_dir": str(outbox_dir),
        "outbox_files": outbox_files,
    }


def _register_demo_close_tool(stack: DemoStack) -> None:
    """Reference implementation of ``execute_close_case`` writing to the outbox.

    Registered only when the platform catalog does not already provide the
    tool, and obeys every execute-band invariant: declared side effect, human
    approval required, and writes land in ``settings.data_dir / "outbox"`` -
    never in a real system.
    """
    settings = stack.settings

    def handler(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        outbox = Path(settings.ensure_data_dir()) / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        path = outbox / "case_closures.jsonl"
        record = {
            "case_id": arguments.get("case_id"),
            "resolution": arguments.get("resolution"),
            "narrative": arguments.get("narrative", ""),
            "closed_by": context.caller,
            "ticket_id": context.ticket_id,
            "approval_token": context.approval.token_id if context.approval else None,
            "ts": utcnow().isoformat(),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return {"status": "closed", "outbox_path": str(path), "case_id": record["case_id"]}

    stack.registry.register(
        ToolSpec(
            name=CLOSE_TOOL,
            description="Close an AML case with a resolution code. Writes a closure record to "
            "the local outbox for downstream (human-operated) case management.",
            input_schema={
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "resolution": {"type": "string"},
                    "narrative": {"type": "string"},
                },
                "required": ["case_id", "resolution"],
                "additionalProperties": False,
            },
            band=ToolBand.execute,
            risk_tier=RiskTier.high,
            side_effect=SideEffectClass.reversible,
            requires_human_approval=True,
            provenance_required=True,
            idempotent=False,
            owner="demos",
        ),
        handler,
    )


def run(
    offline_ok: bool = True,
    console: Console | None = None,
    seed: int = 7,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the AML triage demo end-to-end and return structured results.

    Returns a dict with keys ``demo``, ``surge``, ``queue_size``, ``scorer``
    (``source``/``auc``), ``triage`` (precision/recall per depth),
    ``top_alert``, ``narrative``, ``close_case`` (with ``blocked`` and
    ``approved`` branches plus outbox paths), ``audit`` and ``warnings``.
    """
    return asyncio.run(arun(offline_ok=offline_ok, console=console, seed=seed, settings=settings))


def main() -> None:
    """Contracted CLI entry point: ``fintwinos demo aml_triage``."""
    run()
