"""The contracted eval entry point: ``run_suites(suite, report_prefix) -> dict``.

Wired to ``fintwinos eval`` in the CLI. The runner:

1. builds the twin runtime, simulators and tool registry through the canonical entry
   points (all imported lazily — a partial install degrades to skipped suites, never
   a crash);
2. assembles the requested suites (``"all"``, one name, or a comma-separated list);
3. runs them through the harness :class:`~fintwinos.evals.harness.Runner`;
4. evaluates the founding-brief release gates against the summaries;
5. writes ``<report_prefix>.md`` and ``<report_prefix>.json``; and
6. returns the whole summary as one JSON-serialisable dict.

Everything works fully offline (``FINTWIN_OFFLINE=1``): the function-call suites then
run their deterministic rule-based subjects instead of OpenAI function calling.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from fintwinos.core.config import Settings, get_settings
from fintwinos.evals.adapters import (
    FilingsQASuite,
    MultiTurnToolSuite,
    load_bfcl_style,
)
from fintwinos.evals.consistency import ConsistencySuite
from fintwinos.evals.domain_metrics import (
    ComplianceRecallSuite,
    CustomerOpsSlaSuite,
    RiskScenarioSuite,
    TreasurySurvivalSuite,
)
from fintwinos.evals.function_calls import LlmSubject, ToolCallExactnessSuite
from fintwinos.evals.harness import Runner, Suite, release_gates
from fintwinos.evals.reports import config_fingerprint, write_reports
from fintwinos.models.llm_routing.client import LLMClient

DEFAULT_REPORT_PREFIX = Path(".fintwinos/eval-report")

SUITE_NAMES: tuple[str, ...] = (
    "function_calls",
    "bfcl_style",
    "finmcp_style",
    "secque_style",
    "consistency",
    "domain_risk",
    "domain_treasury",
    "domain_compliance",
    "domain_customer_ops",
)


def _run_coro(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine to completion whether or not an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # called from inside a running loop (e.g. a notebook): isolate in a worker thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _build_environment(settings: Settings) -> tuple[Any, Any, LLMClient, list[str]]:
    """Assemble runtime + simulators + registry via the canonical (lazy) entry points.

    Each step that fails leaves a human-readable note in the report instead of
    aborting: the eval stack must always produce evidence, even on a partial build.
    """
    notes: list[str] = []
    runtime: Any = None
    registry: Any = None

    try:
        from fintwinos.twin_core.runtime import build_runtime

        runtime = build_runtime(seed=settings.seed, with_demo_data=True)
    except Exception as exc:  # module absent or broken — degrade, don't die
        notes.append(f"twin runtime unavailable: {type(exc).__name__}: {exc}")

    if runtime is not None:
        try:
            from fintwinos.twin_sim import register_all

            register_all(runtime)
        except Exception as exc:
            notes.append(f"simulators unavailable: {type(exc).__name__}: {exc}")

    if runtime is not None:
        try:
            from fintwinos.tools.catalog import build_default_registry

            registry = build_default_registry(runtime, settings=settings)
        except Exception as exc:
            notes.append(f"tool registry unavailable: {type(exc).__name__}: {exc}")

    llm = LLMClient(settings=settings)
    if llm.offline:
        notes.append("offline mode: LLM-backed subjects fell back to rule-based baselines")
    return runtime, registry, llm, notes


def assemble_suites(
    suite: str,
    runtime: Any,
    registry: Any,
    llm: LLMClient,
    seed: int = 7,
) -> list[Suite]:
    """Build the requested suites; ``suite`` is ``"all"``, a name, or a comma list."""
    factories: dict[str, Any] = {
        "function_calls": lambda: ToolCallExactnessSuite(subject=LlmSubject(llm)),
        "bfcl_style": lambda: ToolCallExactnessSuite(
            cases=load_bfcl_style(), subject=LlmSubject(llm), name="bfcl_style"
        ),
        "finmcp_style": lambda: MultiTurnToolSuite(),
        "secque_style": lambda: FilingsQASuite(),
        "consistency": lambda: ConsistencySuite(runtime, registry, llm),
        "domain_risk": lambda: RiskScenarioSuite(runtime, seed),
        "domain_treasury": lambda: TreasurySurvivalSuite(runtime, seed),
        "domain_compliance": lambda: ComplianceRecallSuite(runtime, seed),
        "domain_customer_ops": lambda: CustomerOpsSlaSuite(runtime, seed),
    }
    requested = (
        list(SUITE_NAMES)
        if suite.strip().lower() == "all"
        else [name.strip() for name in suite.split(",") if name.strip()]
    )
    unknown = [name for name in requested if name not in factories]
    if unknown:
        raise ValueError(
            f"unknown suite(s) {unknown}; valid names: {', '.join(SUITE_NAMES)} or 'all'"
        )
    return [factories[name]() for name in requested]


def run_suites(
    suite: str = "all",
    report_prefix: Path = DEFAULT_REPORT_PREFIX,
    offline: bool | None = None,
) -> dict[str, Any]:
    """Run the evaluation stack and write Markdown + JSON reports.

    Args:
        suite: ``"all"``, one suite name, or a comma-separated list of
            :data:`SUITE_NAMES`.
        report_prefix: reports are written to ``<report_prefix>.md`` and
            ``<report_prefix>.json`` (parent directories are created).
        offline: force offline (``True``) or online (``False``) mode for this run;
            ``None`` keeps the environment's setting (``FINTWIN_OFFLINE``).

    Returns:
        A JSON-serialisable dict with per-suite summaries, gate verdicts, the config
        fingerprint and the written report paths.
    """
    settings = get_settings()
    if offline is not None:
        settings = settings.model_copy(update={"offline": bool(offline)})

    runtime, registry, llm, notes = _build_environment(settings)
    suites = assemble_suites(suite, runtime, registry, llm, seed=settings.seed)

    runner = Runner(suites)
    summaries = _run_coro(runner.run_all())
    gates = release_gates(summaries)
    fingerprint = config_fingerprint(settings)
    paths = write_reports(Path(report_prefix), summaries, gates, fingerprint, notes)

    if runtime is not None and getattr(runtime, "audit", None) is not None:
        runtime.audit.append(
            "evals",
            "eval.completed",
            {
                "suite": suite,
                "fingerprint": fingerprint,
                "gates_passed": sum(1 for g in gates if g.passed),
                "gates_total": len(gates),
                "reports": paths,
            },
        )

    return {
        "suite": suite,
        "offline": llm.offline,
        "config_fingerprint": fingerprint,
        "suites": {name: summary.model_dump() for name, summary in summaries.items()},
        "gates": [gate.model_dump() for gate in gates],
        "gates_passed": sum(1 for g in gates if g.passed),
        "gates_total": len(gates),
        "reports": paths,
        "notes": notes,
        "llm_usage": llm.usage_summary(),
    }
