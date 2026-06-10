"""Risk domain tool tests: positions, exposures, limits, shocks, hedge ranking."""

from __future__ import annotations

from fintwinos.tools.registry import CallContext

CTX = CallContext(caller="test-risk")


async def test_observe_positions_returns_enriched_book(registry):
    result = await registry.call("observe_positions", {}, CTX)
    assert result.ok, result.error
    data = result.data
    assert data["count"] == 5
    # Sorted by absolute market value descending.
    values = [abs(p["market_value"]) for p in data["positions"]]
    assert values == sorted(values, reverse=True)
    top = data["positions"][0]
    assert top["position_id"] == "POS_1"
    assert top["asset_class"] == "equity"  # enriched from the instrument entity
    assert top["currency"] == "USD"
    assert data["total_market_value"] == 6_750_000.0


async def test_observe_positions_filters_by_account(registry):
    result = await registry.call("observe_positions", {"account_id": "ACC_TRD_1"}, CTX)
    assert result.ok
    assert result.data["count"] == 2
    assert {p["account_id"] for p in result.data["positions"]} == {"ACC_TRD_1"}


async def test_observe_positions_rejects_extra_properties(registry):
    result = await registry.call("observe_positions", {"acount_id": "typo"}, CTX)
    assert not result.ok
    assert "failed validation" in result.error


async def test_observe_exposures_aggregates_by_asset_class_and_currency(registry):
    result = await registry.call("observe_exposures", {}, CTX)
    assert result.ok
    by_asset_class = result.data["by_asset_class"]
    assert by_asset_class["equity"]["gross"] == 6_800_000.0
    assert by_asset_class["equity"]["net"] == 6_800_000.0
    assert by_asset_class["rates"]["net"] == -1_200_000.0
    assert by_asset_class["rates"]["short"] == -1_200_000.0
    by_currency = result.data["by_currency"]
    assert by_currency["GBP"]["gross"] == 2_300_000.0
    assert result.data["totals"]["positions"] == 5


async def test_observe_limit_utilisation_flags_breach(registry):
    result = await registry.call("observe_limit_utilisation", {}, CTX)
    assert result.ok
    limits = {row["limit_id"]: row for row in result.data["limits"]}
    assert limits["LIM_CR"]["status"] == "breach"
    assert limits["LIM_CR"]["utilisation"] > 1.0
    assert limits["LIM_EQ"]["status"] in {"ok", "warning"}
    assert result.data["breaches"] == 1
    # Highest utilisation first.
    utilisations = [row["utilisation"] for row in result.data["limits"]]
    assert utilisations == sorted(utilisations, reverse=True)


async def test_simulate_market_shock_returns_cis_and_is_deterministic(registry):
    arguments = {"scenario_name": "equity_down_10", "shock_pct": -10}
    first = await registry.call("simulate_market_shock", arguments, CTX)
    second = await registry.call("simulate_market_shock", arguments, CTX)
    assert first.ok, first.error
    assert first.data["metrics"] == second.data["metrics"]
    assert first.data["confidence"]
    for low, high in first.data["confidence"].values():
        assert low <= high
    assert first.data["calibration"]
    assert first.provenance  # provenance_required tools surface provenance


async def test_simulate_market_shock_rejects_out_of_range_shock(registry):
    result = await registry.call(
        "simulate_market_shock", {"scenario_name": "melt", "shock_pct": -95}, CTX
    )
    assert not result.ok
    assert "failed validation" in result.error


async def test_propose_hedge_candidates_ranks_by_reduction_per_cost(registry):
    result = await registry.call("propose_hedge_candidates", {"max_candidates": 10}, CTX)
    assert result.ok, result.error
    candidates = result.data["candidates"]
    assert candidates, "exposed asset classes must yield candidates"
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)
    for candidate in candidates:
        assert candidate["notional"] > 0
        assert candidate["estimated_cost"] > 0
        assert candidate["expected_exposure_reduction"] > 0
        assert candidate["direction"] in {"buy", "sell"}
    # Long equity book must be hedged by selling.
    equity = [c for c in candidates if c["asset_class"] == "equity"]
    assert equity and all(c["direction"] == "sell" for c in equity)


async def test_propose_hedge_candidates_is_deterministic_and_filterable(registry):
    arguments = {"asset_class": "equity", "max_candidates": 2}
    first = await registry.call("propose_hedge_candidates", arguments, CTX)
    second = await registry.call("propose_hedge_candidates", arguments, CTX)
    assert first.ok
    assert first.data == second.data
    assert len(first.data["candidates"]) <= 2
    assert {c["asset_class"] for c in first.data["candidates"]} == {"equity"}


async def test_propose_hedge_candidates_respects_cost_ceiling(registry):
    result = await registry.call(
        "propose_hedge_candidates", {"max_cost_bps": 1.5, "max_candidates": 10}, CTX
    )
    assert result.ok
    assert all(c["cost_bps"] <= 1.5 for c in result.data["candidates"])
