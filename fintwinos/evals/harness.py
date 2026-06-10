"""Core evaluation harness: cases, results, suites, the runner, and release gates.

Vocabulary
----------

- An :class:`EvalCase` is one frozen test item (input + expected output).
- An :class:`EvalResult` is the scored outcome of one case: ``passed`` plus a
  continuous ``score`` in ``[0, 1]`` so partial credit is visible in reports.
- A :class:`Suite` owns a list of cases and knows how to evaluate one case against a
  *subject* (the thing under test: a routing function, an LLM, an agent runtime, or a
  whole twin). ``Suite.run(subject)`` returns ``list[EvalResult]``.
- The :class:`Runner` orchestrates many suites, producing one :class:`SuiteSummary`
  per suite (pass rate, mean score, latency, suite-specific metrics).
- :func:`release_gates` maps suite summaries onto the founding-brief release gates
  (e.g. *function-call exactness pass rate >= 0.95 before any write-enabled tool is
  switched on*). A gate over a missing or skipped suite **fails** — absence of
  evidence is never treated as evidence of safety.

Suites that depend on sibling modules (the agent runtime, the simulators) raise
:class:`SuiteUnavailable` from ``preflight`` when those modules are not importable;
the runner records them as ``skipped`` rather than crashing, so the eval stack always
completes end-to-end, fully offline.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class SuiteUnavailable(RuntimeError):
    """A suite cannot run because a dependency (module, runtime, simulator) is absent."""


# ---------------------------------------------------------------------------
# Case / result / summary models
# ---------------------------------------------------------------------------


class EvalCase(BaseModel):
    """One frozen evaluation item.

    ``input`` and ``expected`` are free-form dicts whose shape is owned by the suite
    that evaluates the case; ``metadata`` carries provenance (fixture line, source
    benchmark style, author notes) and never affects scoring.
    """

    id: str
    suite: str
    input: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """The scored outcome of a single case.

    ``score`` is continuous in ``[0, 1]`` (partial credit); ``passed`` is the binary
    verdict used for pass rates and release gates. ``details`` holds suite-specific
    diagnostics (per-component scores, predictions, latency).
    """

    case_id: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)


class SuiteSummary(BaseModel):
    """Per-suite roll-up rendered into reports and consumed by release gates."""

    suite: str
    status: Literal["ok", "skipped", "error"] = "ok"
    n_cases: int = 0
    n_passed: int = 0
    pass_rate: float = 0.0
    mean_score: float = 0.0
    mean_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class Suite(ABC):
    """Base class for evaluation suites.

    Subclasses implement :meth:`evaluate_case` and may override:

    - :meth:`default_subject` — the subject used when ``run()`` is called bare;
    - :meth:`preflight` — raise :class:`SuiteUnavailable` if dependencies are missing;
    - :meth:`extra_metrics` — suite-specific scalars merged into the summary
      (e.g. ``hallucinated_tool_rate``, ``min_survival_days``).
    """

    def __init__(self, name: str, cases: list[EvalCase]):
        self.name = name
        self.cases: list[EvalCase] = list(cases)

    # -- hooks -----------------------------------------------------------------

    def default_subject(self) -> Any | None:
        """Subject to evaluate when none is supplied; ``None`` means 'required'."""
        return None

    def preflight(self, subject: Any) -> None:
        """Raise :class:`SuiteUnavailable` when the suite cannot run."""
        return None

    @abstractmethod
    async def evaluate_case(self, case: EvalCase, subject: Any) -> EvalResult:
        """Score one case against the subject."""

    def extra_metrics(self, results: list[EvalResult]) -> dict[str, float]:
        """Suite-specific scalar metrics derived from the full result list."""
        return {}

    # -- execution ---------------------------------------------------------------

    async def run(self, subject: Any | None = None) -> list[EvalResult]:
        """Evaluate every case, timing each one; per-case failures never abort the run."""
        subject = subject if subject is not None else self.default_subject()
        self.preflight(subject)
        results: list[EvalResult] = []
        for case in self.cases:
            started = time.perf_counter()
            try:
                result = await self.evaluate_case(case, subject)
            except SuiteUnavailable:
                raise
            except Exception as exc:  # one bad case must not sink the suite
                result = EvalResult(
                    case_id=case.id,
                    passed=False,
                    score=0.0,
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result.details.setdefault("latency_ms", round(elapsed_ms, 3))
            results.append(result)
        return results


def summarise(suite: Suite, results: list[EvalResult]) -> SuiteSummary:
    """Roll a suite's results up into a :class:`SuiteSummary`."""
    n = len(results)
    n_passed = sum(1 for r in results if r.passed)
    latencies = [float(r.details.get("latency_ms", 0.0)) for r in results]
    metrics = {key: round(float(val), 6) for key, val in suite.extra_metrics(results).items()}
    return SuiteSummary(
        suite=suite.name,
        status="ok",
        n_cases=n,
        n_passed=n_passed,
        pass_rate=round(n_passed / n, 6) if n else 0.0,
        mean_score=round(sum(r.score for r in results) / n, 6) if n else 0.0,
        mean_latency_ms=round(sum(latencies) / n, 3) if n else 0.0,
        total_latency_ms=round(sum(latencies), 3),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Runner:
    """Orchestrates suites: runs each one, summarises, and survives unavailable suites.

    ``last_results`` keeps the raw per-case results of the most recent ``run_all``
    keyed by suite name, for callers that want case-level drill-down.
    """

    def __init__(self, suites: list[Suite]):
        names = [s.name for s in suites]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate suite names: {sorted(names)}")
        self.suites = list(suites)
        self.last_results: dict[str, list[EvalResult]] = {}

    async def run_all(self, subjects: dict[str, Any] | None = None) -> dict[str, SuiteSummary]:
        """Run every suite; ``subjects`` optionally overrides a suite's default subject.

        A suite raising :class:`SuiteUnavailable` is recorded as ``skipped``; any other
        suite-level exception is recorded as ``error``. Both leave the run alive.
        """
        subjects = subjects or {}
        summaries: dict[str, SuiteSummary] = {}
        for suite in self.suites:
            try:
                results = await suite.run(subjects.get(suite.name))
            except SuiteUnavailable as exc:
                summaries[suite.name] = SuiteSummary(
                    suite=suite.name, status="skipped", error=str(exc)
                )
                continue
            except Exception as exc:  # defensive: a broken suite must not kill the run
                summaries[suite.name] = SuiteSummary(
                    suite=suite.name, status="error", error=f"{type(exc).__name__}: {exc}"
                )
                continue
            self.last_results[suite.name] = results
            summaries[suite.name] = summarise(suite, results)
        return summaries


# ---------------------------------------------------------------------------
# Release gates
# ---------------------------------------------------------------------------


class ReleaseGate(BaseModel):
    """One founding-brief release gate: a threshold on a suite-level metric."""

    name: str
    description: str
    suite: str
    metric: str  # "pass_rate" | "mean_score" | "mean_latency_ms" | a key in summary.metrics
    op: Literal["ge", "le"]
    threshold: float
    blocks: str  # what stays locked while this gate fails


class GateResult(BaseModel):
    """The evaluated verdict of one release gate against a set of summaries."""

    name: str
    description: str
    suite: str
    metric: str
    op: Literal["ge", "le"]
    threshold: float
    blocks: str
    value: float | None = None
    passed: bool = False
    reason: str = ""


def _resolve_metric(summary: SuiteSummary, metric: str) -> float | None:
    if metric in {"pass_rate", "mean_score", "mean_latency_ms", "total_latency_ms"}:
        return float(getattr(summary, metric))
    if metric in summary.metrics:
        return float(summary.metrics[metric])
    return None


def builtin_gates() -> list[ReleaseGate]:
    """The founding-brief release gates, thresholds sourced from the single ``GATES`` dict.

    Imported lazily from :mod:`fintwinos.evals.domain_metrics` (which imports this
    module at top level) to keep the dependency acyclic.
    """
    from fintwinos.evals.domain_metrics import GATES

    return [
        ReleaseGate(
            name="function-call-exactness",
            description=(
                "Tool-call exactness pass rate on the frozen function-call benchmark "
                "must reach the brief's floor before any write-enabled (execute-band) "
                "tool is switched on."
            ),
            suite="function_calls",
            metric="pass_rate",
            op="ge",
            threshold=GATES["function_calls.pass_rate_min"],
            blocks="write-enabled (execute-band) tools",
        ),
        ReleaseGate(
            name="tool-hallucination-rate",
            description="Rate of calls to tools that do not exist in the catalog.",
            suite="function_calls",
            metric="hallucinated_tool_rate",
            op="le",
            threshold=GATES["function_calls.hallucinated_tool_rate_max"],
            blocks="write-enabled (execute-band) tools",
        ),
        ReleaseGate(
            name="multi-run-consistency",
            description=(
                "Agent runtime must produce the same status and decision action_type "
                "across repeated runs of fixed objectives."
            ),
            suite="consistency",
            metric="mean_score",
            op="ge",
            threshold=GATES["consistency.mean_score_min"],
            blocks="agent autonomy promotion (shadow -> assisted)",
        ),
        ReleaseGate(
            name="risk-scenario-coverage",
            description="Fraction of preset stress scenarios the risk simulator covers cleanly.",
            suite="domain_risk",
            metric="scenario_coverage",
            op="ge",
            threshold=GATES["risk.scenario_coverage_min"],
            blocks="risk simulator promotion to decision support",
        ),
        ReleaseGate(
            name="risk-expected-shortfall-present",
            description="Every covered stress scenario must report an Expected Shortfall metric.",
            suite="domain_risk",
            metric="es_present_rate",
            op="ge",
            threshold=GATES["risk.es_present_rate_min"],
            blocks="risk simulator promotion to decision support",
        ),
        ReleaseGate(
            name="treasury-survival-floor",
            description="Minimum survival days across preset deposit-runoff stresses.",
            suite="domain_treasury",
            metric="min_survival_days",
            op="ge",
            threshold=GATES["treasury.survival_days_floor"],
            blocks="treasury recommendations beyond shadow mode",
        ),
        ReleaseGate(
            name="compliance-recall-floor",
            description=(
                "Detection recall at the fixed alert threshold on synthetic AML typologies."
            ),
            suite="domain_compliance",
            metric="recall",
            op="ge",
            threshold=GATES["compliance.recall_floor"],
            blocks="automated alert triage suggestions",
        ),
        ReleaseGate(
            name="ops-sla-breach-ceiling",
            description="SLA breach rate under baseline staffing in the customer-ops simulator.",
            suite="domain_customer_ops",
            metric="sla_breach_rate",
            op="le",
            threshold=GATES["customer_ops.sla_breach_ceiling"],
            blocks="customer-ops staffing proposals beyond shadow mode",
        ),
    ]


def release_gates(
    summaries: dict[str, SuiteSummary],
    gates: list[ReleaseGate] | None = None,
) -> list[GateResult]:
    """Evaluate release gates against suite summaries.

    Conservative semantics: a gate whose suite is missing, skipped, errored, or whose
    metric is absent **fails** with an explanatory reason. Gates only pass on positive
    evidence from a suite that actually ran.
    """
    gates = gates if gates is not None else builtin_gates()
    results: list[GateResult] = []
    for gate in gates:
        base = gate.model_dump()
        summary = summaries.get(gate.suite)
        if summary is None:
            results.append(
                GateResult(
                    **base, value=None, passed=False,
                    reason=f"no evidence: suite '{gate.suite}' was not run",
                )
            )
            continue
        if summary.status != "ok":
            results.append(
                GateResult(
                    **base, value=None, passed=False,
                    reason=(
                        f"no evidence: suite '{gate.suite}' {summary.status}"
                        + (f" ({summary.error})" if summary.error else "")
                    ),
                )
            )
            continue
        value = _resolve_metric(summary, gate.metric)
        if value is None:
            results.append(
                GateResult(
                    **base, value=None, passed=False,
                    reason=f"metric '{gate.metric}' missing from suite '{gate.suite}'",
                )
            )
            continue
        passed = value >= gate.threshold if gate.op == "ge" else value <= gate.threshold
        symbol = ">=" if gate.op == "ge" else "<="
        results.append(
            GateResult(
                **base,
                value=round(value, 6),
                passed=passed,
                reason=f"{gate.metric}={round(value, 6)} {symbol} {gate.threshold} is {passed}",
            )
        )
    return results
