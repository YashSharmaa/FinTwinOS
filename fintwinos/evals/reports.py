"""Evaluation report rendering: Markdown for humans, JSON for machines.

Both renderings carry the same payload: per-suite tables, suite-specific metrics,
release-gate verdicts (PASS/FAIL per founding-brief gate), and the configuration
fingerprint — a stable hash over the settings and gate thresholds that produced the
run, so two reports are comparable exactly when their fingerprints match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import fintwinos
from fintwinos.core.config import Settings, get_settings
from fintwinos.core.types import utcnow
from fintwinos.evals.harness import GateResult, SuiteSummary


def config_fingerprint(settings: Settings | None = None) -> str:
    """Stable 16-hex fingerprint of everything that shapes an eval run's meaning.

    Covers the package version, seed, offline flag, model routing tiers, environment
    and the full release-gate threshold dict — so a threshold tweak or model remap
    visibly changes the fingerprint on the next report.
    """
    from fintwinos.evals.domain_metrics import GATES

    settings = settings or get_settings()
    body = json.dumps(
        {
            "version": fintwinos.__version__,
            "seed": settings.seed,
            "offline": settings.offline,
            "llm_provider": settings.llm_provider,
            "llm_model_primary": settings.llm_model_primary,
            "llm_model_fast": settings.llm_model_fast,
            "llm_model_cheap": settings.llm_model_cheap,
            "environment": settings.environment,
            "gates": GATES,
        },
        sort_keys=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _credit_footer() -> str:
    return (
        f"{fintwinos.CREDIT}\n"
        "FinTwinOS — auditable digital-twin operating system for financial organisations."
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def render_json(
    summaries: dict[str, SuiteSummary],
    gates: list[GateResult] | None = None,
    fingerprint: str = "",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable report payload (already JSON-serialisable)."""
    gates = gates or []
    return {
        "report": "fintwinos-eval",
        "version": fintwinos.__version__,
        "generated_at": utcnow().isoformat(),
        "config_fingerprint": fingerprint,
        "suites": {name: summary.model_dump() for name, summary in summaries.items()},
        "gates": [gate.model_dump() for gate in gates],
        "gates_passed": sum(1 for g in gates if g.passed),
        "gates_total": len(gates),
        "notes": list(notes or []),
        "credit": fintwinos.CREDIT,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(
    summaries: dict[str, SuiteSummary],
    gates: list[GateResult] | None = None,
    fingerprint: str = "",
    notes: list[str] | None = None,
) -> str:
    """Render the human-readable Markdown report with per-suite and gate tables."""
    gates = gates or []
    parts: list[str] = [
        "# FinTwinOS evaluation report",
        "",
        f"- Generated: `{utcnow().isoformat()}`",
        f"- Config fingerprint: `{fingerprint}`",
        f"- FinTwinOS version: `{fintwinos.__version__}`",
        "",
        "## Suite results",
        "",
    ]
    suite_rows = []
    for name in sorted(summaries):
        s = summaries[name]
        suite_rows.append([
            f"`{name}`",
            s.status,
            str(s.n_cases),
            str(s.n_passed),
            f"{s.pass_rate:.3f}",
            f"{s.mean_score:.3f}",
            f"{s.mean_latency_ms:.1f}",
        ])
    parts.append(_md_table(
        ["suite", "status", "cases", "passed", "pass rate", "mean score", "mean latency (ms)"],
        suite_rows,
    ))

    metric_rows = [
        [f"`{name}`", f"`{key}`", f"{value:.6g}"]
        for name in sorted(summaries)
        for key, value in sorted(summaries[name].metrics.items())
    ]
    if metric_rows:
        parts += ["", "## Suite metrics", "", _md_table(["suite", "metric", "value"], metric_rows)]

    skipped = [s for s in summaries.values() if s.status != "ok"]
    if skipped:
        parts += ["", "## Suites not evaluated", ""]
        parts += [f"- `{s.suite}` ({s.status}): {s.error or 'no detail'}" for s in skipped]

    parts += ["", "## Release gates", ""]
    gate_rows = []
    for gate in gates:
        symbol = ">=" if gate.op == "ge" else "<="
        gate_rows.append([
            f"`{gate.name}`",
            f"`{gate.suite}`",
            f"`{gate.metric}` {symbol} {gate.threshold:g}",
            "n/a" if gate.value is None else f"{gate.value:.6g}",
            "**PASS**" if gate.passed else "**FAIL**",
            gate.blocks,
        ])
    parts.append(_md_table(
        ["gate", "suite", "requirement", "value", "verdict", "blocks while failing"],
        gate_rows,
    ))
    passed = sum(1 for g in gates if g.passed)
    parts += ["", f"**{passed}/{len(gates)} release gates passing.**"]

    if notes:
        parts += ["", "## Run notes", ""]
        parts += [f"- {note}" for note in notes]

    parts += ["", "---", "", _credit_footer(), ""]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_reports(
    report_prefix: Path,
    summaries: dict[str, SuiteSummary],
    gates: list[GateResult] | None = None,
    fingerprint: str = "",
    notes: list[str] | None = None,
) -> dict[str, str]:
    """Write ``<prefix>.md`` and ``<prefix>.json``; returns both paths."""
    report_prefix = Path(report_prefix)
    report_prefix.parent.mkdir(parents=True, exist_ok=True)
    # plain string concatenation: a prefix like "eval-report.v2" must not lose ".v2"
    md_path = Path(str(report_prefix) + ".md")
    json_path = Path(str(report_prefix) + ".json")
    md_path.write_text(
        render_markdown(summaries, gates, fingerprint, notes), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(render_json(summaries, gates, fingerprint, notes), indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    return {"markdown": str(md_path), "json": str(json_path)}
