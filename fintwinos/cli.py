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
    table.add_row("offline", str(settings.offline))
    table.add_row("llm_available", str(settings.llm_available))
    table.add_row("environment", settings.environment)
    table.add_row("execute_tools_enabled", str(settings.execute_tools_enabled))
    table.add_row("shadow_mode", str(settings.shadow_mode))
    table.add_row("data_dir", str(settings.data_dir))
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
    module.main()


@app.command("serve-tools")
def serve_tools(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8341),
) -> None:
    """Serve the typed tool catalog over MCP-style JSON-RPC (requires [server] extra)."""
    try:
        import uvicorn

        from fintwinos.tools.server import create_app
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "server extra not installed — pip install 'fintwinos[server]'"
        ) from exc
    uvicorn.run(create_app(), host=host, port=port)


@app.command("eval")
def run_eval(
    suite: str = typer.Argument("all", help="Suite name or 'all'"),
    report: Path = typer.Option(Path(".fintwinos/eval-report"), help="Report output prefix"),
) -> None:
    """Run the evaluation stack (function-calls, agent runs, domain metrics)."""
    from fintwinos.evals.runner import run_suites

    summary = run_suites(suite, report_prefix=report)
    console.print(summary)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
