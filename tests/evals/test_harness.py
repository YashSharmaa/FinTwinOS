"""Tests for the eval harness primitives: cases, suites, runner, summaries, gates."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fintwinos.evals.harness import (
    EvalCase,
    EvalResult,
    ReleaseGate,
    Runner,
    Suite,
    SuiteSummary,
    SuiteUnavailable,
    builtin_gates,
    release_gates,
    summarise,
)


class _ConstantSuite(Suite):
    """Scores every case with a fixed (passed, score) pair."""

    def __init__(self, name: str, n_cases: int, passed: bool, score: float):
        cases = [EvalCase(id=f"{name}-{i}", suite=name) for i in range(n_cases)]
        super().__init__(name, cases)
        self._passed = passed
        self._score = score

    async def evaluate_case(self, case, subject):
        return EvalResult(case_id=case.id, passed=self._passed, score=self._score)

    def extra_metrics(self, results):
        return {"constant_metric": self._score}


class _ExplodingSuite(Suite):
    def __init__(self):
        super().__init__("exploding", [EvalCase(id="boom-1", suite="exploding")])

    async def evaluate_case(self, case, subject):
        raise RuntimeError("kaboom")


class _UnavailableSuite(Suite):
    def __init__(self):
        super().__init__("unavailable", [EvalCase(id="u-1", suite="unavailable")])

    def preflight(self, subject):
        raise SuiteUnavailable("sibling module not built yet")

    async def evaluate_case(self, case, subject):  # pragma: no cover - never reached
        raise AssertionError("should not run")


# --- models ----------------------------------------------------------------------


def test_eval_result_score_bounds():
    with pytest.raises(ValidationError):
        EvalResult(case_id="x", passed=True, score=1.5)
    with pytest.raises(ValidationError):
        EvalResult(case_id="x", passed=False, score=-0.1)


def test_eval_case_defaults():
    case = EvalCase(id="c1", suite="s")
    assert case.input == {} and case.expected == {} and case.metadata == {}


# --- suite execution ----------------------------------------------------------------


async def test_suite_run_records_latency_and_survives_case_errors():
    suite = _ExplodingSuite()
    results = await suite.run(subject=object())
    assert len(results) == 1
    assert results[0].passed is False and results[0].score == 0.0
    assert "RuntimeError" in results[0].details["error"]
    assert results[0].details["latency_ms"] >= 0.0


async def test_summarise_math():
    suite = _ConstantSuite("half", 4, passed=False, score=0.5)
    results = await suite.run(subject=object())
    # flip two to passing to get a 0.5 pass rate
    results[0] = EvalResult(case_id="half-0", passed=True, score=1.0,
                            details={"latency_ms": 2.0})
    results[1] = EvalResult(case_id="half-1", passed=True, score=1.0,
                            details={"latency_ms": 4.0})
    summary = summarise(suite, results)
    assert summary.n_cases == 4 and summary.n_passed == 2
    assert summary.pass_rate == 0.5
    assert summary.mean_score == pytest.approx((1.0 + 1.0 + 0.5 + 0.5) / 4)
    assert summary.metrics["constant_metric"] == 0.5
    assert summary.total_latency_ms >= 6.0


# --- runner ----------------------------------------------------------------------


async def test_runner_handles_ok_skipped_and_error_suites():
    class _BrokenSuite(Suite):
        def __init__(self):
            super().__init__("broken", [])

        def preflight(self, subject):
            raise ZeroDivisionError("suite-level explosion")

        async def evaluate_case(self, case, subject):  # pragma: no cover
            raise AssertionError

    runner = Runner([_ConstantSuite("good", 3, True, 1.0), _UnavailableSuite(), _BrokenSuite()])
    summaries = await runner.run_all()
    assert summaries["good"].status == "ok" and summaries["good"].pass_rate == 1.0
    assert summaries["unavailable"].status == "skipped"
    assert "sibling module" in summaries["unavailable"].error
    assert summaries["broken"].status == "error"
    assert "ZeroDivisionError" in summaries["broken"].error
    assert runner.last_results["good"][0].passed is True


def test_runner_rejects_duplicate_suite_names():
    with pytest.raises(ValueError):
        Runner([_ConstantSuite("dup", 1, True, 1.0), _ConstantSuite("dup", 1, True, 1.0)])


async def test_runner_subject_override():
    class _EchoSuite(Suite):
        def __init__(self):
            super().__init__("echo", [EvalCase(id="e-1", suite="echo")])

        def default_subject(self):
            return "default"

        async def evaluate_case(self, case, subject):
            return EvalResult(case_id=case.id, passed=subject == "override", score=1.0)

    runner = Runner([_EchoSuite()])
    summaries = await runner.run_all(subjects={"echo": "override"})
    assert summaries["echo"].pass_rate == 1.0


# --- release gates ----------------------------------------------------------------


def _summary(name: str, **kwargs) -> SuiteSummary:
    return SuiteSummary(suite=name, **kwargs)


def test_release_gates_pass_and_fail_on_synthetic_summaries():
    gates = [
        ReleaseGate(name="g-ge", description="", suite="alpha", metric="pass_rate",
                    op="ge", threshold=0.95, blocks="execute tools"),
        ReleaseGate(name="g-le", description="", suite="alpha", metric="bad_rate",
                    op="le", threshold=0.02, blocks="execute tools"),
    ]
    summaries = {
        "alpha": _summary("alpha", status="ok", n_cases=20, n_passed=19,
                          pass_rate=0.95, mean_score=0.97, metrics={"bad_rate": 0.05}),
    }
    results = release_gates(summaries, gates)
    by_name = {r.name: r for r in results}
    assert by_name["g-ge"].passed is True and by_name["g-ge"].value == 0.95
    assert by_name["g-le"].passed is False and by_name["g-le"].value == 0.05


def test_release_gates_fail_conservatively_without_evidence():
    gates = [
        ReleaseGate(name="missing-suite", description="", suite="ghost",
                    metric="pass_rate", op="ge", threshold=0.5, blocks="x"),
        ReleaseGate(name="skipped-suite", description="", suite="skipped",
                    metric="pass_rate", op="ge", threshold=0.5, blocks="x"),
        ReleaseGate(name="missing-metric", description="", suite="ok",
                    metric="not_a_metric", op="ge", threshold=0.5, blocks="x"),
    ]
    summaries = {
        "skipped": _summary("skipped", status="skipped", error="not built"),
        "ok": _summary("ok", status="ok", n_cases=1, n_passed=1, pass_rate=1.0),
    }
    results = {r.name: r for r in release_gates(summaries, gates)}
    assert all(not r.passed for r in results.values())
    assert "not run" in results["missing-suite"].reason
    assert "skipped" in results["skipped-suite"].reason
    assert "missing" in results["missing-metric"].reason


def test_builtin_gates_cover_every_brief_domain():
    gates = builtin_gates()
    suites_covered = {g.suite for g in gates}
    assert {"function_calls", "consistency", "domain_risk", "domain_treasury",
            "domain_compliance", "domain_customer_ops"} <= suites_covered
    fc = next(g for g in gates if g.name == "function-call-exactness")
    assert fc.threshold == 0.95 and fc.op == "ge" and fc.metric == "pass_rate"
    assert "execute" in fc.blocks
