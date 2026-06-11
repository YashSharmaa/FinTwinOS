"""FinTwinOS command-line interface.

Subcommands lazy-import their modules so a partial install still gives a working CLI.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

import fintwinos
from fintwinos.core.config import get_settings

app = typer.Typer(
    name="fintwinos",
    help="FinTwinOS — auditable digital-twin operating system for financial organisations.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print version and attribution."""
    console.print(fintwinos.CREDIT)


@app.command()
def info() -> None:
    """Show effective settings (secrets redacted) and runtime mode."""
    settings = get_settings()
    table = Table(title="FinTwinOS configuration")
    table.add_column("setting")
    table.add_column("value")
    table.add_row("version", fintwinos.__version__)
    table.add_row("llm_provider", settings.llm_provider)
    table.add_row("llm_model_primary", settings.llm_model_primary)
    table.add_row("llm_model_fast", settings.llm_model_fast)
    table.add_row("llm_model_cheap", settings.llm_model_cheap)
    table.add_row("openai_api_key", "set" if settings.openai_api_key else "missing")
    table.add_row("llm_budget_usd", str(settings.llm_budget_usd))
    table.add_row("llm_cache_enabled", str(settings.llm_cache_enabled))
    table.add_row("offline", str(settings.offline))
    table.add_row("llm_available", str(settings.llm_available))
    table.add_row("environment", settings.environment)
    table.add_row("seed", str(settings.seed))
    table.add_row("execute_tools_enabled", str(settings.execute_tools_enabled))
    table.add_row("dual_control_required", str(settings.dual_control_required))
    table.add_row("shadow_mode", str(settings.shadow_mode))
    table.add_row("approval_secret", "set" if settings.approval_secret else "missing")
    table.add_row("rbac_default_role", settings.rbac_default_role)
    table.add_row("data_dir", str(settings.data_dir))
    table.add_row(
        "audit_path", str(settings.audit_path) if settings.audit_path else "in-memory"
    )
    console.print(table)


@app.command("export-schemas")
def export_schemas(out: Path = typer.Option(Path("schemas"), help="Output directory")) -> None:
    """Write canonical entity JSON Schemas to disk."""
    from fintwinos.core.schema_export import export_all

    written = export_all(out)
    console.print(f"wrote {len(written)} schema files to {out}/")


@app.command()
def demo(
    name: str = typer.Argument("liquidity", help="liquidity | aml_triage | analyst_research | customer_ops | day_in_the_life"),
) -> None:
    """Run a packaged demo against the public-data demo twin (works fully offline)."""
    try:
        module = importlib.import_module(f"fintwinos.demos.{name}")
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(f"unknown demo '{name}': {exc}") from exc
    entry = getattr(module, "main", None)
    if not callable(entry):
        raise typer.BadParameter(
            f"'{name}' is not a runnable demo (fintwinos.demos.{name} has no main())"
        )
    entry()


@app.command("serve-tools")
def serve_tools(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8341),
    with_ingest: bool = typer.Option(
        False, "--with-ingest",
        help="Also mount POST /ingest/{kind} (webhook ingestion into the twin)",
    ),
) -> None:
    r"""Serve the typed tool catalog over MCP-style JSON-RPC (requires the \[server] extra)."""
    try:
        import uvicorn

        from fintwinos.tools.server import create_app
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "server extra not installed — pip install 'fintwinos[server]'"
        ) from exc
    uvicorn.run(create_app(with_ingest=with_ingest), host=host, port=port)


@app.command()
def ingest(
    path: Path | None = typer.Argument(None, help="CSV or JSONL-CDC file to ingest"),
    edgar: str | None = typer.Option(None, "--edgar", help="Ingest SEC EDGAR filings for a CIK instead of a file"),
    kind: str | None = typer.Option(None, help="Event kind override for CSV rows"),
    entity_type: str | None = typer.Option(None, help="Entity type for CSV rows / CDC tables"),
    limit: int = typer.Option(5, help="Max filings to ingest with --edgar"),
) -> None:
    """Ingest a file (CSV / JSONL CDC) or EDGAR filings into the demo twin and report routing."""
    if (path is None) == (edgar is None):
        raise typer.BadParameter("provide exactly one of PATH or --edgar CIK")
    from fintwinos.connectors.ingest import ingest_edgar, ingest_path
    from fintwinos.twin_core.runtime import build_runtime

    runtime = build_runtime()
    if path is not None:
        summary = ingest_path(path, runtime, kind=kind, entity_type=entity_type)
    else:
        summary = ingest_edgar(edgar, runtime, limit=limit)
    console.print(summary)


@app.command()
def datasets(
    action: str = typer.Argument("list", help="list | load"),
    name: str | None = typer.Argument(None, help="Dataset name (for load)"),
) -> None:
    """List the registered datasets or load one and print its summary."""
    from fintwinos.datasets.registry import list_datasets, load_dataset

    if action == "list":
        table = Table(title="FinTwinOS datasets")
        table.add_column("name")
        table.add_column("kind")
        table.add_column("offline")
        table.add_column("description")
        for entry in list_datasets():
            table.add_row(
                str(entry.get("name")), str(entry.get("kind")),
                str(entry.get("offline_available")), str(entry.get("description"))[:70],
            )
        console.print(table)
    elif action == "load":
        if not name:
            raise typer.BadParameter("datasets load requires a dataset name")
        data = load_dataset(name)
        size = len(data) if hasattr(data, "__len__") else "n/a"
        console.print({"dataset": name, "loaded": True, "size": size})
    else:
        raise typer.BadParameter("action must be 'list' or 'load'")


@app.command()
def rl(
    seed: int | None = typer.Option(None, help="Master seed (default: FINTWIN_SEED)"),
    report_dir: Path | None = typer.Option(None, help="Write report.json here"),
) -> None:
    """Run the bounded offline RL pipeline: twin episodes → train → OPE → shadow gate."""
    from fintwinos.rl.pipeline import run_rl_pipeline

    report = run_rl_pipeline(seed=seed, report_dir=report_dir)
    console.print(report)


@app.command("replay-verify")
def replay_verify(
    seed: int = typer.Option(7, help="Demo twin seed to rebuild and replay"),
) -> None:
    """Replay the recorded demo episode into a fresh twin and verify state-hash equality."""
    from fintwinos.twin_core.replay import verify_replay

    result = verify_replay(seed=seed)
    console.print(result)
    if not result.get("match"):
        raise typer.Exit(code=1)


@app.command("eval")
def run_eval(
    suite: str = typer.Argument("all", help="Suite name or 'all'"),
    report: Path | None = typer.Option(
        None, help="Report output prefix (default: <FINTWIN_DATA_DIR>/eval-report)"
    ),
) -> None:
    """Run the evaluation stack (function-calls, agent runs, domain metrics)."""
    from fintwinos.evals.runner import run_suites

    if report is None:
        report = Path(get_settings().data_dir) / "eval-report"
    summary = run_suites(suite, report_prefix=report)
    console.print(summary)


@app.command()
def killswitch(
    action: str = typer.Argument(..., help="status | engage | release"),
    band: str | None = typer.Option(None, help="Tool band (observe|simulate|propose|execute); omit for global"),
    reason: str = typer.Option("", help="Mandatory for engage/release"),
    actor: str = typer.Option("cli", help="Who is operating the switch"),
    role: str | None = typer.Option(
        None,
        help="Optional RBAC role to enforce (operator engages; admin releases). Omit to skip the check.",
    ),
) -> None:
    """Operate the persisted platform kill switch (state in <FINTWIN_DATA_DIR>/killswitch.json)."""
    from fintwinos.policy.killswitch import KillSwitch

    switch = KillSwitch()
    if action == "status":
        console.print(switch.status())
    elif action == "engage":
        console.print(switch.engage(actor=actor, reason=reason, band=band, role=role))
    elif action == "release":
        console.print(switch.release(actor=actor, reason=reason, band=band, role=role))
    else:
        raise typer.BadParameter("action must be one of: status, engage, release")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
