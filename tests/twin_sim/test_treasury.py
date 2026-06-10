"""Tests for the multi-currency liquidity simulator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fintwinos.core.types import Scenario, SimulationResult
from fintwinos.twin_sim.treasury import LiquiditySimulator

HEADLINE = {
    "min_cumulative_net_position",
    "survival_days",
    "lcr_proxy",
    "peak_funding_cost_bps",
}


def make_scenario(**params) -> Scenario:
    return Scenario(name="treasury_test", params=params)


@pytest.fixture()
def sim() -> LiquiditySimulator:
    return LiquiditySimulator()


def test_run_is_deterministic_per_seed(sim):
    scenario = make_scenario(stress_outflow_multiplier=2.0, n_reps=6)
    r1 = sim.run(scenario, seed=17)
    r2 = sim.run(scenario, seed=17)
    assert r1.metrics == r2.metrics
    assert r1.series == r2.series
    assert r1.confidence == r2.confidence
    assert sim.run(scenario, seed=18).metrics != r1.metrics


def test_result_shape_and_confidence_for_every_metric(sim):
    result = sim.run(make_scenario(horizon_days=20, n_reps=6), seed=2)
    assert isinstance(result, SimulationResult)
    assert result.simulator == "treasury"
    assert HEADLINE == set(result.metrics)
    for name in result.metrics:
        ci = result.confidence[name]
        assert len(ci) == 2 and ci[0] <= ci[1], name
    assert "ks_stat" in result.calibration
    assert "calibrated" in result.calibration


def test_per_currency_series_present(sim):
    result = sim.run(
        make_scenario(currencies=["USD", "EUR", "GBP"], horizon_days=15, n_reps=4), seed=3
    )
    for ccy in ("usd", "eur", "gbp"):
        assert f"cum_net_{ccy}" in result.series
        assert len(result.series[f"cum_net_{ccy}"]) == 15
    for key in ("total_cum_net", "cbc_remaining", "funding_cost_bps"):
        assert key in result.series


def test_survival_days_decrease_with_stress_severity(sim):
    survivals = []
    for mult in (1.0, 2.5, 4.0):
        result = sim.run(
            make_scenario(stress_outflow_multiplier=mult, horizon_days=30, n_reps=8), seed=5
        )
        survivals.append(result.metrics["survival_days"])
    assert survivals[0] == 30.0  # business-as-usual survives the horizon
    assert survivals[0] > survivals[1] > survivals[2]  # severity bites monotonically


def test_breach_produces_warning(sim):
    result = sim.run(make_scenario(stress_outflow_multiplier=4.0, n_reps=4), seed=5)
    assert any("counterbalancing" in w for w in result.warnings)


def test_funding_spread_shock_raises_peak_cost(sim):
    base = sim.run(make_scenario(shock_bps=0.0, n_reps=4), seed=9)
    shocked = sim.run(make_scenario(shock_bps=200.0, n_reps=4), seed=9)
    assert (
        shocked.metrics["peak_funding_cost_bps"]
        >= base.metrics["peak_funding_cost_bps"] + 200.0 - 1e-9
    )


def test_lcr_proxy_degrades_under_stress(sim):
    calm = sim.run(make_scenario(stress_outflow_multiplier=1.0, n_reps=4), seed=4)
    squeezed = sim.run(make_scenario(stress_outflow_multiplier=3.0, n_reps=4), seed=4)
    assert squeezed.metrics["lcr_proxy"] < calm.metrics["lcr_proxy"]


def test_reads_cash_ladder_series_from_runtime(runtime):
    # An observed USD ladder of heavy net outflows must dominate the synthetic
    # baseline: the wired simulator sees a far worse cumulative position.
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for day in range(10):
        runtime.timeseries.append(
            "treasury.cash_ladder.USD", start + timedelta(days=day), -200.0
        )
    scenario = make_scenario(currencies=["USD"], horizon_days=20, n_reps=4)
    wired = LiquiditySimulator(runtime=runtime).run(scenario, seed=6)
    unwired = LiquiditySimulator().run(scenario, seed=6)
    assert (
        wired.metrics["min_cumulative_net_position"]
        < unwired.metrics["min_cumulative_net_position"] - 1000.0
    )


def test_custom_cbc_tranches_change_survival(sim):
    thin = sim.run(
        make_scenario(
            stress_outflow_multiplier=3.0,
            cbc_tranches=[{"name": "cash", "amount": 50.0, "haircut": 0.0}],
            n_reps=4,
        ),
        seed=8,
    )
    thick = sim.run(
        make_scenario(
            stress_outflow_multiplier=3.0,
            cbc_tranches=[{"name": "cash", "amount": 5000.0, "haircut": 0.0}],
            n_reps=4,
        ),
        seed=8,
    )
    assert thick.metrics["survival_days"] > thin.metrics["survival_days"]


def test_per_currency_multiplier_dict_accepted(sim):
    result = sim.run(
        make_scenario(
            stress_outflow_multiplier={"USD": 3.0, "EUR": 1.0, "GBP": 1.0}, n_reps=4
        ),
        seed=10,
    )
    assert result.metrics["survival_days"] <= 30.0
    assert result.metrics["min_cumulative_net_position"] < 0.0


def test_calibration_report_shape(sim):
    sim.run(make_scenario(n_reps=4), seed=1)
    report = sim.calibration_report()
    assert report["simulator"] == "treasury"
    assert "ks_stat" in report["last_run"]
