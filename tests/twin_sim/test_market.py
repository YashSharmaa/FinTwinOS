"""Tests for the agent-based market simulator."""

from __future__ import annotations

import pytest

from fintwinos.core.types import Scenario, SimulationResult
from fintwinos.twin_sim.market import AgentBasedMarketSimulator


def make_scenario(**params) -> Scenario:
    return Scenario(name="market_test", params=params)


@pytest.fixture()
def sim() -> AgentBasedMarketSimulator:
    return AgentBasedMarketSimulator()


def test_run_is_deterministic_per_seed(sim):
    scenario = make_scenario(regime="stressed", n_steps=120, shock_bps=200.0, n_paths=6)
    r1 = sim.run(scenario, seed=21)
    r2 = sim.run(scenario, seed=21)
    assert r1.metrics == r2.metrics
    assert r1.series == r2.series
    assert r1.confidence == r2.confidence
    r3 = sim.run(scenario, seed=22)
    assert r3.metrics != r1.metrics


def test_result_shape_and_confidence_for_every_metric(sim):
    result = sim.run(make_scenario(n_steps=100, n_paths=6), seed=3)
    assert isinstance(result, SimulationResult)
    assert result.simulator == "market"
    assert result.seed == 3
    expected = {
        "realised_vol",
        "max_drawdown",
        "expected_shortfall_97_5",
        "kurtosis",
        "autocorr_abs_lag1",
        "shock_half_life",
        "terminal_price",
    }
    assert expected == set(result.metrics)
    for name in result.metrics:
        ci = result.confidence[name]
        assert len(ci) == 2 and ci[0] <= ci[1], name
    assert "ks_stat" in result.calibration
    assert "calibrated" in result.calibration
    assert result.calibration["calibrated"] is False
    assert len(result.series["mid_price"]) == 101
    assert len(result.series["returns"]) == 100


def test_stylised_facts_hold_in_stressed_regime(sim):
    result = sim.run(make_scenario(regime="stressed", n_steps=250, n_paths=12), seed=7)
    facts = sim.stylised_facts(result)
    assert facts["heavy_tails"] is True  # kurtosis > 3
    assert facts["volatility_clustering"] is True  # |r| lag-1 autocorr > 0
    assert facts["all_pass"] is True
    assert result.metrics["kurtosis"] > 3.0
    assert result.metrics["autocorr_abs_lag1"] > 0.0


def test_stressed_regime_is_wilder_than_calm(sim):
    calm = sim.run(make_scenario(regime="calm", n_steps=250, n_paths=10), seed=13)
    stressed = sim.run(make_scenario(regime="stressed", n_steps=250, n_paths=10), seed=13)
    assert stressed.metrics["realised_vol"] > calm.metrics["realised_vol"]
    assert stressed.metrics["kurtosis"] > calm.metrics["kurtosis"]


def test_shock_half_life_behaviour(sim):
    no_shock = sim.run(make_scenario(n_steps=150, shock_bps=0.0, n_paths=6), seed=5)
    assert no_shock.metrics["shock_half_life"] == 0.0
    shocked = sim.run(
        make_scenario(n_steps=150, shock_bps=400.0, shock_step=50, n_paths=6), seed=5
    )
    assert shocked.metrics["shock_half_life"] > 0.0
    assert shocked.metrics["shock_half_life"] <= 150 - 50


def test_integer_agent_mix_is_accepted(sim):
    result = sim.run(make_scenario(n_agents=40, n_steps=64, n_paths=4), seed=1)
    assert result.metrics["realised_vol"] > 0.0


def test_unknown_regime_falls_back_with_warning(sim):
    result = sim.run(make_scenario(regime="apocalyptic", n_steps=64, n_paths=4), seed=1)
    assert any("regime" in w for w in result.warnings)


def test_out_of_range_shock_step_is_relocated(sim):
    result = sim.run(
        make_scenario(n_steps=64, shock_bps=100.0, shock_step=999, n_paths=4), seed=2
    )
    assert any("shock_step" in w for w in result.warnings)
    assert result.metrics["shock_half_life"] >= 0.0


def test_seed_falls_back_to_scenario_params(sim):
    scenario = make_scenario(n_steps=64, n_paths=4, seed=123)
    r1 = sim.run(scenario)
    r2 = sim.run(scenario, seed=123)
    assert r1.seed == 123
    assert r1.metrics == r2.metrics


def test_calibration_report_shape(sim):
    sim.run(make_scenario(n_steps=64, n_paths=4), seed=9)
    report = sim.calibration_report()
    assert report["simulator"] == "market"
    assert "ks_stat" in report["last_run"]
    assert report["calibrator"] is None
    assert report["reference_overridden"] is False


def test_fit_calibration_marks_results_calibrated(sim):
    baseline = sim.run(make_scenario(n_steps=80, n_paths=6), seed=4)
    # Pretend the institution's observed return history is the simulator's own
    # output: after fitting, the calibration block flips to calibrated=True and the
    # mapped KS cannot exceed sensible bounds.
    ref = baseline.series["returns"] * 3  # >= 8 observations
    sim.fit_calibration(ref, baseline.series["returns"])
    result = sim.run(make_scenario(n_steps=80, n_paths=6), seed=4)
    assert result.calibration["calibrated"] is True
    assert "ks_stat_raw" in result.calibration
    assert 0.0 <= result.calibration["ks_stat"] <= 1.0
