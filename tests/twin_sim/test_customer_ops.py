"""Tests for the multi-class service-queue simulator."""

from __future__ import annotations

import pytest

from fintwinos.core.types import Scenario, SimulationResult
from fintwinos.twin_sim.customer_ops import ServiceQueueSimulator

HEADLINE = {"avg_handling_time", "sla_breach_rate", "abandonment_rate", "p90_wait"}


def make_scenario(**params) -> Scenario:
    params.setdefault("horizon_hours", 4.0)
    params.setdefault("n_reps", 6)
    return Scenario(name="queue_test", params=params)


@pytest.fixture()
def sim() -> ServiceQueueSimulator:
    return ServiceQueueSimulator()


def test_run_is_deterministic_per_seed(sim):
    scenario = make_scenario(arrival_rate_per_hr=80.0, n_agents=6)
    r1 = sim.run(scenario, seed=41)
    r2 = sim.run(scenario, seed=41)
    assert r1.metrics == r2.metrics
    assert r1.series == r2.series
    assert sim.run(scenario, seed=42).metrics != r1.metrics


def test_result_shape_and_confidence_for_every_metric(sim):
    result = sim.run(make_scenario(arrival_rate_per_hr=60.0, n_agents=8), seed=1)
    assert isinstance(result, SimulationResult)
    assert result.simulator == "customer_ops"
    assert HEADLINE == set(result.metrics)
    for name in result.metrics:
        ci = result.confidence[name]
        assert len(ci) == 2 and ci[0] <= ci[1], name
    assert "ks_stat" in result.calibration
    assert "calibrated" in result.calibration
    assert "queue_length" in result.series
    assert len(result.series["queue_length"]) >= 2


def test_rates_are_valid_fractions(sim):
    result = sim.run(make_scenario(arrival_rate_per_hr=100.0, n_agents=5), seed=2)
    assert 0.0 <= result.metrics["sla_breach_rate"] <= 1.0
    assert 0.0 <= result.metrics["abandonment_rate"] <= 1.0
    assert result.metrics["avg_handling_time"] > 0.0
    assert result.metrics["p90_wait"] >= 0.0


def test_sla_breach_rate_increases_with_arrival_rate(sim):
    quiet = sim.run(make_scenario(arrival_rate_per_hr=40.0, n_agents=6), seed=3)
    slammed = sim.run(make_scenario(arrival_rate_per_hr=120.0, n_agents=6), seed=3)
    assert (
        slammed.metrics["sla_breach_rate"] > quiet.metrics["sla_breach_rate"] + 0.1
    )
    assert slammed.metrics["abandonment_rate"] > quiet.metrics["abandonment_rate"]
    assert slammed.metrics["p90_wait"] > quiet.metrics["p90_wait"]


def test_staffing_lever_reduces_breaches(sim):
    understaffed = sim.run(make_scenario(arrival_rate_per_hr=100.0, n_agents=4), seed=4)
    well_staffed = sim.run(make_scenario(arrival_rate_per_hr=100.0, n_agents=14), seed=4)
    assert (
        well_staffed.metrics["sla_breach_rate"]
        < understaffed.metrics["sla_breach_rate"]
    )


def test_routing_lever_accepts_both_policies(sim):
    for routing in ("priority", "fifo"):
        result = sim.run(
            make_scenario(arrival_rate_per_hr=90.0, n_agents=6, routing=routing), seed=5
        )
        assert 0.0 <= result.metrics["sla_breach_rate"] <= 1.0
    bad = sim.run(make_scenario(routing="psychic", n_agents=6), seed=5)
    assert any("routing" in w for w in bad.warnings)


def test_overload_warning_emitted(sim):
    result = sim.run(make_scenario(arrival_rate_per_hr=200.0, n_agents=4), seed=6)
    assert any("offered load" in w for w in result.warnings)


def test_escalations_extend_handling_time(sim):
    calm = sim.run(
        make_scenario(arrival_rate_per_hr=50.0, n_agents=10, escalation_prob=0.0), seed=7
    )
    heavy = sim.run(
        make_scenario(arrival_rate_per_hr=50.0, n_agents=10, escalation_prob=0.9), seed=7
    )
    assert heavy.metrics["avg_handling_time"] > calm.metrics["avg_handling_time"]


def test_custom_class_mix_accepted(sim):
    classes = {
        "gold": {"weight": 0.5, "priority": 0, "handle_mean": 5.0, "patience_mean": 20.0},
        "basic": {"weight": 0.5, "priority": 1, "handle_mean": 4.0, "patience_mean": 10.0},
    }
    result = sim.run(
        make_scenario(arrival_rate_per_hr=70.0, n_agents=6, classes=classes), seed=8
    )
    assert result.metrics["avg_handling_time"] > 0.0


def test_calibration_report_shape(sim):
    sim.run(make_scenario(n_agents=8), seed=9)
    report = sim.calibration_report()
    assert report["simulator"] == "customer_ops"
    assert "ks_stat" in report["last_run"]
