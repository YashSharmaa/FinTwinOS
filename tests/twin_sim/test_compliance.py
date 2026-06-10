"""Tests for the fraud-ring transaction-graph simulator."""

from __future__ import annotations

import pytest

from fintwinos.core.types import Alert, Scenario, SimulationResult
from fintwinos.twin_sim.compliance import FraudRingSimulator

HEADLINE = {
    "true_ring_edges",
    "ring_accounts",
    "n_alerts",
    "alert_precision",
    "alert_recall",
    "analyst_minutes_expected",
}

# Release-gate operating point, mirroring fintwinos.evals.domain_metrics
# (GATES["compliance.recall_floor"] / GATES["compliance.alert_threshold"] and
# COMPLIANCE_PRESETS). Kept as literals: twin_sim tests code against core only
# (CONTRACTS.md), so they must not import evals internals.
RELEASE_GATE_RECALL_FLOOR = 0.75
RELEASE_GATE_ALERT_THRESHOLD = 0.5
RELEASE_GATE_RING_RATES = (0.08, 0.12, 0.20)  # sparse / baseline / dense presets


def make_scenario(**params) -> Scenario:
    params.setdefault("n_steps", 40)
    params.setdefault("n_accounts", 120)
    params.setdefault("n_reps", 4)
    return Scenario(name="compliance_test", params=params)


@pytest.fixture()
def sim() -> FraudRingSimulator:
    return FraudRingSimulator()


def test_run_is_deterministic_per_seed(sim):
    scenario = make_scenario(ring_rate=0.3)
    r1 = sim.run(scenario, seed=31)
    r2 = sim.run(scenario, seed=31)
    assert r1.metrics == r2.metrics
    assert r1.series == r2.series
    assert sim.run(scenario, seed=32).metrics != r1.metrics


def test_result_shape_and_confidence_for_every_metric(sim):
    result = sim.run(make_scenario(ring_rate=0.3), seed=1)
    assert isinstance(result, SimulationResult)
    assert result.simulator == "compliance"
    assert HEADLINE == set(result.metrics)
    for name in result.metrics:
        ci = result.confidence[name]
        assert len(ci) == 2 and ci[0] <= ci[1], name
    assert "ks_stat" in result.calibration
    assert "calibrated" in result.calibration


def test_rings_are_embedded_and_counted(sim):
    result = sim.run(make_scenario(ring_rate=0.4), seed=2)
    assert result.metrics["true_ring_edges"] > 0.0
    assert result.metrics["ring_accounts"] > 0.0
    assert result.metrics["analyst_minutes_expected"] > 0.0


def test_precision_and_recall_are_sane(sim):
    result = sim.run(make_scenario(ring_rate=0.4), seed=3)
    assert 0.0 <= result.metrics["alert_precision"] <= 1.0
    assert 0.0 <= result.metrics["alert_recall"] <= 1.0
    for series in ("precision_at_threshold", "recall_at_threshold"):
        assert all(0.0 <= v <= 1.0 for v in result.series[series])


def test_recall_rises_as_threshold_falls(sim):
    result = sim.run(make_scenario(ring_rate=0.4), seed=4)
    thresholds = result.series["thresholds"]
    recall = result.series["recall_at_threshold"]
    assert thresholds == sorted(thresholds)
    # Monotone non-increasing in the threshold == rising as the threshold falls.
    for lo_idx in range(len(recall) - 1):
        assert recall[lo_idx] >= recall[lo_idx + 1] - 1e-12
    assert recall[0] >= 0.8  # nearly every ring member is caught wide-open
    assert recall[0] >= recall[-1]


def test_alert_count_grows_as_threshold_falls(sim):
    result = sim.run(make_scenario(ring_rate=0.3), seed=5)
    alerts = result.series["alerts_at_threshold"]
    for i in range(len(alerts) - 1):
        assert alerts[i] >= alerts[i + 1]


def test_last_alerts_are_alert_objects_above_threshold(sim):
    threshold = 0.6
    sim.run(make_scenario(ring_rate=0.4, alert_threshold=threshold), seed=6)
    assert sim.last_alerts, "expected alerts at the default surge rate"
    for alert in sim.last_alerts:
        assert isinstance(alert, Alert)
        assert alert.score >= threshold
        assert alert.kind == "aml_ring_suspicion"
        assert alert.entities and alert.entities[0].entity_type == "account"


def test_zero_ring_rate_yields_vacuous_recall_and_warning(sim):
    result = sim.run(make_scenario(ring_rate=0.0), seed=7)
    assert result.metrics["ring_accounts"] == 0.0
    assert result.metrics["true_ring_edges"] == 0.0
    assert result.metrics["alert_recall"] == 0.0
    assert any("ring" in w for w in result.warnings)


@pytest.mark.parametrize("ring_rate", RELEASE_GATE_RING_RATES)
def test_recall_at_default_threshold_clears_release_gate_floor(sim, ring_rate):
    """Recall at the documented default threshold must clear the 0.75 gate floor.

    Runs the simulator exactly as the ``compliance-recall-floor`` release gate does:
    default sizes (n_steps=60, n_accounts=200, n_reps=8), the documented default
    ``alert_threshold`` of 0.5, the eval runner's seed 7, and every ring-density
    preset — the gate is conjunctive over the worst preset. A margin above the
    floor keeps the gate off the knife edge.
    """
    scenario = Scenario(
        name=f"gate_ring_rate_{ring_rate:g}",
        params={"ring_rate": ring_rate, "alert_threshold": RELEASE_GATE_ALERT_THRESHOLD},
    )
    result = sim.run(scenario, seed=7)
    assert result.metrics["alert_recall"] >= RELEASE_GATE_RECALL_FLOOR + 0.05
    # The CI lower bound must clear the floor too, not just the point estimate.
    assert result.confidence["alert_recall"][0] >= RELEASE_GATE_RECALL_FLOOR


@pytest.mark.parametrize("seed", [11, 31, 99])
def test_release_gate_recall_is_not_seed_dependent(sim, seed):
    """The gate floor must hold for seeds other than the eval runner's default."""
    for ring_rate in RELEASE_GATE_RING_RATES:
        scenario = Scenario(
            name=f"gate_seed_{seed}_rr_{ring_rate:g}",
            params={"ring_rate": ring_rate, "alert_threshold": RELEASE_GATE_ALERT_THRESHOLD},
        )
        result = sim.run(scenario, seed=seed)
        assert result.metrics["alert_recall"] >= RELEASE_GATE_RECALL_FLOOR, (
            f"seed={seed} ring_rate={ring_rate}"
        )


def test_structured_evidence_outranks_background_traffic(sim):
    """Ring members (structured amounts) must score above pure-background accounts."""
    result = sim.run(make_scenario(ring_rate=0.3, alert_threshold=0.5), seed=12)
    assert sim.last_alerts, "expected alerts at ring_rate 0.3"
    ring_hits = [a for a in sim.last_alerts if a.payload["is_true_ring_member"]]
    assert len(ring_hits) / len(sim.last_alerts) >= 0.9
    assert result.metrics["alert_precision"] >= 0.9


def test_ring_surge_increases_workload(sim):
    quiet = sim.run(make_scenario(ring_rate=0.05), seed=8)
    surge = sim.run(make_scenario(ring_rate=0.6), seed=8)
    assert surge.metrics["true_ring_edges"] > quiet.metrics["true_ring_edges"]
    assert (
        surge.metrics["analyst_minutes_expected"] > quiet.metrics["analyst_minutes_expected"]
    )


def test_invalid_params_are_clamped_with_warnings(sim):
    result = sim.run(make_scenario(ring_rate=7.0, alert_threshold=2.0), seed=9)
    assert any("ring_rate" in w for w in result.warnings)
    assert any("alert_threshold" in w for w in result.warnings)


def test_calibration_report_shape(sim):
    sim.run(make_scenario(), seed=10)
    report = sim.calibration_report()
    assert report["simulator"] == "compliance"
    assert "ks_stat" in report["last_run"]
