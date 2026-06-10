"""Tests for the domain release-gate suites.

Most tests drive the suites with deterministic stub simulators; the compliance
gate additionally gets an end-to-end test against the real fraud-ring simulator,
because that gate ships in ``fintwinos eval all`` and must pass out of the box.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fintwinos.core.types import Scenario, SimulationResult
from fintwinos.evals.domain_metrics import (
    GATES,
    ComplianceRecallSuite,
    CustomerOpsSlaSuite,
    RiskScenarioSuite,
    TreasurySurvivalSuite,
    domain_suites,
)
from fintwinos.evals.harness import SuiteUnavailable, summarise


class StubSimulator:
    """Deterministic stand-in implementing the Simulator protocol surface we use."""

    def __init__(self, name: str, metrics: dict[str, float],
                 confidence: dict[str, list[float]] | None = None, ok: bool = True):
        self.name = name
        self._metrics = metrics
        self._confidence = confidence if confidence is not None else {"pnl": [-2.0, -0.5]}
        self._ok = ok
        self.seen_scenarios: list[Scenario] = []

    def run(self, scenario: Scenario, *, seed: int | None = None) -> SimulationResult:
        self.seen_scenarios.append(scenario)
        return SimulationResult(
            simulator=self.name,
            scenario_name=scenario.name,
            ok=self._ok,
            metrics=dict(self._metrics),
            confidence=dict(self._confidence),
            calibration={"method": "stub"},
            seed=seed,
        )

    def calibration_report(self) -> dict[str, Any]:
        return {"method": "stub"}


def _runtime(**simulators: StubSimulator) -> SimpleNamespace:
    return SimpleNamespace(simulators=dict(simulators))


# --- GATES dict ----------------------------------------------------------------


def test_gates_dict_contains_every_brief_threshold():
    expected_keys = {
        "function_calls.pass_rate_min",
        "function_calls.hallucinated_tool_rate_max",
        "consistency.mean_score_min",
        "risk.scenario_coverage_min",
        "risk.es_present_rate_min",
        "treasury.survival_days_floor",
        "compliance.recall_floor",
        "compliance.alert_threshold",
        "customer_ops.sla_breach_ceiling",
    }
    assert expected_keys <= set(GATES)
    assert GATES["function_calls.pass_rate_min"] == 0.95


# --- risk ----------------------------------------------------------------


async def test_risk_suite_full_coverage_with_es_and_confidence():
    sim = StubSimulator("market_risk", {"es_97_5": -3.1, "var_99": -2.4, "pnl_mean": -0.6})
    suite = RiskScenarioSuite(_runtime(market_risk=sim))
    results = await suite.run()
    assert all(r.passed for r in results)
    summary = summarise(suite, results)
    assert summary.metrics["scenario_coverage"] == 1.0
    assert summary.metrics["es_present_rate"] == 1.0
    # every preset scenario was actually exercised, with the suite's seed
    assert len(sim.seen_scenarios) == len(suite.cases)


async def test_risk_suite_flags_missing_expected_shortfall():
    sim = StubSimulator("market_risk", {"var_99": -2.4})  # no ES anywhere
    suite = RiskScenarioSuite(_runtime(market_risk=sim))
    results = await suite.run()
    assert all(not r.passed for r in results)
    summary = summarise(suite, results)
    assert summary.metrics["scenario_coverage"] == 1.0
    assert summary.metrics["es_present_rate"] == 0.0


# --- treasury ----------------------------------------------------------------


async def test_treasury_floor_passes_and_fails():
    floor = GATES["treasury.survival_days_floor"]
    healthy = TreasurySurvivalSuite(
        _runtime(liquidity=StubSimulator("liquidity_runoff", {"survival_days": floor + 30}))
    )
    results = await healthy.run()
    assert all(r.passed for r in results)
    assert summarise(healthy, results).metrics["min_survival_days"] == floor + 30

    starved = TreasurySurvivalSuite(
        _runtime(liquidity=StubSimulator("liquidity_runoff", {"survival_days": floor - 10}))
    )
    results = await starved.run()
    assert all(not r.passed for r in results)
    assert summarise(starved, results).metrics["min_survival_days"] == floor - 10


# --- compliance ----------------------------------------------------------------


async def test_compliance_recall_direct_metric_and_fixed_threshold():
    sim = StubSimulator("aml_detection", {"recall": 0.85, "precision": 0.4})
    suite = ComplianceRecallSuite(_runtime(aml=sim))
    results = await suite.run()
    assert all(r.passed for r in results)
    assert all(
        r.details["threshold"] == GATES["compliance.alert_threshold"] for r in results
    )
    # the fixed threshold is passed into the scenario params
    assert all(
        s.params["threshold"] == GATES["compliance.alert_threshold"]
        for s in sim.seen_scenarios
    )
    assert summarise(suite, results).metrics["recall"] == pytest.approx(0.85)


async def test_compliance_recall_computed_from_tp_fn_fallback():
    sim = StubSimulator("aml_detection", {"true_positives": 9, "false_negatives": 3})
    suite = ComplianceRecallSuite(_runtime(aml=sim))
    results = await suite.run()
    assert all(r.passed for r in results)  # 9 / 12 = 0.75 meets the floor
    assert summarise(suite, results).metrics["recall"] == pytest.approx(0.75)


async def test_compliance_gate_passes_with_shipped_fraud_ring_simulator():
    """End to end: the ``compliance-recall-floor`` gate must pass out of the box.

    Runs the real fraud-ring simulator through the suite exactly as
    ``fintwinos eval all`` does (same presets, fixed threshold, seed 7) and
    requires every ring-density preset — including the worst one the gate is
    scored on — to clear the recall floor.
    """
    from fintwinos.twin_sim.compliance import FraudRingSimulator

    suite = ComplianceRecallSuite(_runtime(compliance=FraudRingSimulator()))
    results = await suite.run()
    assert len(results) == 3
    for r in results:
        assert r.passed, f"{r.case_id}: recall={r.details.get('recall')}"
        assert r.details["threshold"] == GATES["compliance.alert_threshold"]
    # the gate metric is the worst preset's recall, and it must clear the floor
    summary = summarise(suite, results)
    assert summary.metrics["recall"] >= GATES["compliance.recall_floor"]


# --- customer ops ----------------------------------------------------------------


async def test_ops_sla_ceiling_pass_and_fail():
    good = CustomerOpsSlaSuite(
        _runtime(customer_ops=StubSimulator("ops_backlog", {"sla_breach_rate": 0.04}))
    )
    results = await good.run()
    assert all(r.passed for r in results)
    assert summarise(good, results).metrics["sla_breach_rate"] == pytest.approx(0.04)

    bad = CustomerOpsSlaSuite(
        _runtime(customer_ops=StubSimulator("ops_backlog", {"sla_breach_rate": 0.31}))
    )
    results = await bad.run()
    assert all(not r.passed for r in results)


# --- availability ----------------------------------------------------------------


async def test_suites_raise_unavailable_without_runtime_or_simulator():
    with pytest.raises(SuiteUnavailable):
        await RiskScenarioSuite(runtime=None).run()
    wrong_sims = _runtime(weather=StubSimulator("weather", {"rain": 1.0}))
    with pytest.raises(SuiteUnavailable):
        await TreasurySurvivalSuite(wrong_sims).run()


def test_domain_suites_builder_returns_all_four():
    suites = domain_suites(None, seed=11)
    assert [s.name for s in suites] == [
        "domain_risk", "domain_treasury", "domain_compliance", "domain_customer_ops",
    ]
    assert all(s.seed == 11 for s in suites)
