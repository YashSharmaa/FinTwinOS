"""Quickstart: build the demo twin and list the governed tool catalog.

Run: ``python examples/01_quickstart.py`` (works fully offline, no API key).
"""

from rich.table import Table

from fintwinos.demos.stack import build_stack

stack = build_stack(seed=7)

table = Table(title=f"FinTwinOS tool catalog — {len(stack.registry)} governed tools")
table.add_column("tool", style="bold")
table.add_column("band", style="cyan")
table.add_column("risk", justify="center")
table.add_column("side effect")
table.add_column("human approval", justify="center")
for spec in stack.registry.list_specs():
    table.add_row(
        spec.name,
        spec.band.value,
        spec.risk_tier.value,
        spec.side_effect.value,
        "required" if spec.requires_human_approval else "—",
    )
stack.console.print(table)
stack.console.print(f"twin assembled · audit chain verified: {stack.registry.audit.verify()}")
