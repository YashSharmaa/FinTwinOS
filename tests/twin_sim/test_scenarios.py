"""Tests for the scenario catalog and the shipped stress presets."""

from __future__ import annotations

import pytest

from fintwinos.core.types import Scenario
from fintwinos.twin_sim import (
    AgentBasedMarketSimulator,
    FraudRingSimulator,
    LiquiditySimulator,
    ServiceQueueSimulator,
    default_catalog,
)
from fintwinos.twin_sim.scenarios import ScenarioCatalog

REQUIRED_PRESETS = {
    "usd_liquidity_squeeze",
    "flash_crash",
    "ring_surge",
    "complaints_spike",
    "rates_shock_200bp",
}


def test_required_presets_are_shipped():
    catalog = default_catalog()
    assert REQUIRED_PRESETS.issubset(set(catalog.list()))
    assert len(catalog) >= len(REQUIRED_PRESETS)


def test_get_returns_fresh_copies():
    catalog = default_catalog()
    a = catalog.get("flash_crash")
    b = catalog.get("flash_crash")
    assert a.scenario_id != b.scenario_id  # distinguishable in the audit trail
    assert a.params == b.params
    a.params["n_steps"] = 1  # mutating a copy must not poison the catalog
    assert catalog.get("flash_crash").params["n_steps"] == 240


def test_get_unknown_scenario_raises_with_listing():
    catalog = default_catalog()
    with pytest.raises(KeyError, match="flash_crash"):
        catalog.get("not_a_scenario")


def test_register_and_overwrite_semantics():
    catalog = ScenarioCatalog(include_defaults=False)
    assert len(catalog) == 0
    scenario = Scenario(name="custom_stress", params={"simulator": "market"})
    catalog.register(scenario)
    assert "custom_stress" in catalog
    with pytest.raises(ValueError):
        catalog.register(Scenario(name="custom_stress"))
    catalog.register(Scenario(name="custom_stress", params={"x": 1}), overwrite=True)
    assert catalog.get("custom_stress").params["x"] == 1


def test_for_simulator_filter():
    catalog = default_catalog()
    treasury = catalog.for_simulator("treasury")
    assert "usd_liquidity_squeeze" in treasury
    assert "rates_shock_200bp" in treasury
    assert "flash_crash" in catalog.for_simulator("market")
    assert "ring_surge" in catalog.for_simulator("compliance")
    assert "complaints_spike" in catalog.for_simulator("customer_ops")


def test_every_preset_runs_on_its_target_simulator():
    catalog = default_catalog()
    simulators = {
        "market": AgentBasedMarketSimulator(),
        "treasury": LiquiditySimulator(),
        "compliance": FraudRingSimulator(),
        "customer_ops": ServiceQueueSimulator(),
    }
    for name in catalog.list():
        scenario = catalog.get(name)
        # Trim the Monte-Carlo budget so the full catalog stays fast in CI.
        scenario.params.setdefault("n_reps", 4)
        scenario.params.setdefault("n_paths", 4)
        target = scenario.params["simulator"]
        result = simulators[target].run(scenario, seed=99)
        assert result.ok is True
        assert result.scenario_name == name
        assert result.metrics, name
        assert set(result.confidence) == set(result.metrics), name
        assert "ks_stat" in result.calibration, name


def test_usd_squeeze_is_worse_than_baseline():
    catalog = default_catalog()
    sim = LiquiditySimulator()
    squeeze = catalog.get("usd_liquidity_squeeze")
    baseline = catalog.get("treasury_baseline")
    squeeze.params["n_reps"] = baseline.params["n_reps"] = 6
    r_squeeze = sim.run(squeeze, seed=12)
    r_base = sim.run(baseline, seed=12)
    assert r_squeeze.metrics["survival_days"] < r_base.metrics["survival_days"]
    assert (
        r_squeeze.metrics["peak_funding_cost_bps"] > r_base.metrics["peak_funding_cost_bps"]
    )


def test_complaints_spike_breaches_more_than_baseline():
    catalog = default_catalog()
    sim = ServiceQueueSimulator()
    spike = catalog.get("complaints_spike")
    base = catalog.get("ops_baseline")
    for s in (spike, base):
        s.params["n_reps"] = 6
        s.params["horizon_hours"] = 4.0
    r_spike = sim.run(spike, seed=13)
    r_base = sim.run(base, seed=13)
    assert r_spike.metrics["sla_breach_rate"] > r_base.metrics["sla_breach_rate"]
