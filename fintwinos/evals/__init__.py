"""FinTwinOS evaluation stack: suites, scoring, release gates and reports.

The eval stack is the promotion machinery of the platform: nothing graduates from
shadow mode to assisted (let alone write-enabled) operation without the release gates
defined here passing on a fresh, reproducible run.

Layout:

- :mod:`fintwinos.evals.harness` — ``EvalCase`` / ``EvalResult`` / ``Suite`` / ``Runner``
  primitives plus the ``release_gates`` check.
- :mod:`fintwinos.evals.function_calls` — tool-call exactness benchmark with a
  deterministic rule-based subject and an LLM function-calling subject.
- :mod:`fintwinos.evals.consistency` — multi-run agreement of the agent runtime.
- :mod:`fintwinos.evals.domain_metrics` — risk / treasury / compliance / customer-ops
  gates computed from simulator outputs, thresholds in one ``GATES`` dict.
- :mod:`fintwinos.evals.adapters` — local-fixture adapters in the style of public
  benchmarks (BFCL, FinMCP, SECQUE) — no downloads, ever.
- :mod:`fintwinos.evals.reports` — Markdown + JSON report rendering.
- :mod:`fintwinos.evals.runner` — the contracted ``run_suites`` CLI entry point.

Created by Yash Sharma (https://www.linkedin.com/in/yashsharmaa/). MIT License.
"""

from __future__ import annotations

from fintwinos.evals.harness import (
    EvalCase,
    EvalResult,
    GateResult,
    ReleaseGate,
    Runner,
    Suite,
    SuiteSummary,
    SuiteUnavailable,
    release_gates,
)

__all__ = [
    "EvalCase",
    "EvalResult",
    "GateResult",
    "ReleaseGate",
    "Runner",
    "Suite",
    "SuiteSummary",
    "SuiteUnavailable",
    "release_gates",
]
