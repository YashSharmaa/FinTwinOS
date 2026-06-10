"""Demo: a full day in the life of the institution, on one shared twin.

Composes the four scenario demos morning-to-evening on a **single** runtime,
registry, LLM client and hash-chained audit trail:

* 09:10 — Treasury rehearses a USD liquidity squeeze.
* 11:25 — Financial crime triages an AML ring surge and closes a case under
  dual control.
* 14:00 — Research ships a fully cited filings brief past the critic.
* 16:40 — Customer operations rehearses the complaints spike counterfactual.
* 18:05 — Close of day: governance roll-up — tool calls by band, audit chain
  length and integrity verdict, LLM usage and indicative cost.

Run with ``fintwinos demo day_in_the_life``. Fully offline-capable.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import fintwinos
from fintwinos.core.config import Settings
from fintwinos.demos import aml_triage, analyst_research, customer_ops, liquidity
from fintwinos.demos.stack import (
    DemoStack,
    build_stack,
    mode_line,
    tool_calls_by_band,
)

#: The day's schedule: (clock, section key, headline, demo module).
SCHEDULE = [
    ("09:10", "liquidity", "Treasury — USD liquidity squeeze", liquidity),
    ("11:25", "aml_triage", "Financial crime — AML ring surge", aml_triage),
    ("14:00", "analyst_research", "Research — filings brief with citations", analyst_research),
    ("16:40", "customer_ops", "Customer operations — complaints spike", customer_ops),
]


async def arun(
    offline_ok: bool = True,
    console: Console | None = None,
    seed: int = 7,
    settings: Settings | None = None,
    stack: DemoStack | None = None,
) -> dict[str, Any]:
    """Async core: run all four scenarios on one shared stack, then roll up."""
    stack = stack or build_stack(seed=seed, console=console, offline_ok=offline_ok, settings=settings)
    console = console or stack.console

    console.rule("[bold cyan]FinTwinOS — a day in the life[/bold cyan]")
    console.print(
        Panel(
            "One institution, one twin, one tamper-evident audit trail. Four desks "
            "rehearse their hardest hour of the day before anything touches production.\n"
            f"[dim]mode: {mode_line(stack.settings)}[/dim]",
            border_style="cyan",
            subtitle=f"[dim]{fintwinos.CREDIT}[/dim]",
        )
    )

    sections: dict[str, Any] = {}
    section_warnings: list[str] = []
    for clock, key, headline, module in SCHEDULE:
        console.rule(f"[bold]{clock} · {headline}[/bold]")
        result = await module.arun(console=console, seed=seed, stack=stack)
        sections[key] = result
        section_warnings.extend(f"{key}: {w}" for w in result.get("warnings", []))

    console.rule("[bold]18:05 · Close of day — governance[/bold]")
    governance = _governance_summary(stack)
    _render_governance(console, stack, governance)

    return {
        "demo": "day_in_the_life",
        "offline": stack.llm.offline,
        "sections": sections,
        "governance": governance,
        "warnings": section_warnings,
    }


def _governance_summary(stack: DemoStack) -> dict[str, Any]:
    """Roll up the day: tool bands, audit integrity, LLM spend."""
    registry_audit = stack.registry.audit
    verified = registry_audit.verify()
    records = len(registry_audit)
    runtime_audit = getattr(stack.runtime, "audit", None)
    if runtime_audit is not None and runtime_audit is not registry_audit:
        verified = verified and runtime_audit.verify()
        records += len(runtime_audit)
    return {
        "tool_calls_by_band": tool_calls_by_band(registry_audit),
        "audit_records": records,
        "audit_verified": verified,
        "llm_usage": stack.llm.usage_summary(),
    }


def _render_governance(console: Console, stack: DemoStack, governance: dict[str, Any]) -> None:
    bands = governance["tool_calls_by_band"]
    table = Table(title="Tool calls by band", title_justify="left", border_style="cyan")
    table.add_column("band", style="bold")
    table.add_column("calls", justify="right")
    table.add_column("governance posture", style="dim")
    posture = {
        "observe": "side-effect free, audited",
        "simulate": "side-effect free, calibrated, audited",
        "propose": "side-effect free; high-risk flagged for review",
        "execute": "approval token + allow rule + kill switch; outbox only",
    }
    for band in ("observe", "simulate", "propose", "execute"):
        if band in bands:
            table.add_row(band, str(bands[band]), posture[band])
    for band, count in bands.items():
        if band not in posture:
            table.add_row(band, str(count), "—")
    console.print(table)

    verified = governance["audit_verified"]
    chain_line = (
        "[green]✓ intact[/green]" if verified else "[red]✗ BROKEN — investigate[/red]"
    )
    usage = governance["llm_usage"]
    usage_line = (
        f"{usage.get('calls', 0)} call(s) · {usage.get('input_tokens', 0)} in / "
        f"{usage.get('output_tokens', 0)} out tokens · "
        f"est. ${usage.get('estimated_cost_usd', 0.0):.4f}"
        + (" · offline (no network)" if usage.get("offline") else "")
    )
    console.print(
        Panel(
            f"audit chain: {governance['audit_records']} hash-linked records — {chain_line}\n"
            f"LLM usage: {usage_line}",
            title="[bold]Governance close of day[/bold]",
            border_style="green" if verified else "red",
            subtitle=f"[dim]{fintwinos.CREDIT}[/dim]",
        )
    )


def run(
    offline_ok: bool = True,
    console: Console | None = None,
    seed: int = 7,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the full day end-to-end and return structured results.

    Returns a dict with keys ``demo``, ``sections`` (the four scenario demo
    results keyed by name) and ``governance`` (``tool_calls_by_band``,
    ``audit_records``, ``audit_verified``, ``llm_usage``).
    """
    return asyncio.run(arun(offline_ok=offline_ok, console=console, seed=seed, settings=settings))


def main() -> None:
    """Contracted CLI entry point: ``fintwinos demo day_in_the_life``."""
    run()
