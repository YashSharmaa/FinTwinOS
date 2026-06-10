"""Market & credit risk tool pack.

Tools registered here:

- ``observe_positions`` — positions enriched with instrument asset class / currency.
- ``observe_exposures`` — gross/net/long/short aggregates by asset class and currency.
- ``observe_limit_utilisation`` — limit entities measured against live exposures.
- ``simulate_market_shock`` — runs the ``market`` simulator for a stress scenario.
- ``propose_hedge_candidates`` — deterministic ranking of hedge instruments by
  exposure reduction per unit of cost.

All handlers read twin state exclusively through the :class:`TwinRuntime` stores and
are side-effect free.
"""

from __future__ import annotations

from typing import Any

from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import EntityRef, RiskTier, Scenario, SideEffectClass, ToolBand
from fintwinos.tools.domains.common import (
    derive_seed,
    entities_of_type,
    object_schema,
    provenance_entry,
    run_simulator,
    simulation_payload,
)
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec

OWNER = "risk"

#: Static hedge universe used by ``propose_hedge_candidates``. Costs are indicative
#: round-trip transaction costs in basis points of notional; effectiveness is the
#: fraction of the targeted exposure the instrument is assumed to offset.
HEDGE_UNIVERSE: list[dict[str, Any]] = [
    {
        "instrument_id": "FUT_ES",
        "name": "S&P 500 e-mini future",
        "asset_class": "equity",
        "cost_bps": 1.6,
        "effectiveness": 0.92,
    },
    {
        "instrument_id": "FUT_SX5E",
        "name": "EURO STOXX 50 future",
        "asset_class": "equity",
        "cost_bps": 2.1,
        "effectiveness": 0.85,
    },
    {
        "instrument_id": "FUT_UST10Y",
        "name": "US 10Y Treasury note future",
        "asset_class": "rates",
        "cost_bps": 1.1,
        "effectiveness": 0.90,
    },
    {
        "instrument_id": "FUT_BUND",
        "name": "Euro-Bund future",
        "asset_class": "rates",
        "cost_bps": 1.3,
        "effectiveness": 0.84,
    },
    {
        "instrument_id": "CDX_IG_5Y",
        "name": "CDX investment-grade 5Y index",
        "asset_class": "credit",
        "cost_bps": 3.2,
        "effectiveness": 0.80,
    },
    {
        "instrument_id": "ITRX_MAIN_5Y",
        "name": "iTraxx Europe Main 5Y index",
        "asset_class": "credit",
        "cost_bps": 3.6,
        "effectiveness": 0.78,
    },
    {
        "instrument_id": "FX_FWD_EURUSD",
        "name": "EUR/USD 1M forward",
        "asset_class": "fx",
        "cost_bps": 0.9,
        "effectiveness": 0.95,
    },
    {
        "instrument_id": "FX_FWD_GBPUSD",
        "name": "GBP/USD 1M forward",
        "asset_class": "fx",
        "cost_bps": 1.0,
        "effectiveness": 0.93,
    },
    {
        "instrument_id": "FUT_BRENT",
        "name": "ICE Brent crude future",
        "asset_class": "commodity",
        "cost_bps": 2.7,
        "effectiveness": 0.82,
    },
]


def _load_positions(runtime: TwinRuntime) -> list[dict[str, Any]]:
    """Load position entities and enrich them with instrument reference data."""
    rows: list[dict[str, Any]] = []
    for ref, attrs in entities_of_type(runtime.graph, "position"):
        instrument_id = str(attrs.get("instrument_id") or "")
        instrument: dict[str, Any] = {}
        if instrument_id:
            found = runtime.graph.get_entity(
                EntityRef(entity_type="instrument", entity_id=instrument_id)
            )
            instrument = dict(found or {})
        as_of = attrs.get("as_of")
        rows.append(
            {
                "position_id": ref.entity_id,
                "account_id": attrs.get("account_id"),
                "instrument_id": instrument_id or None,
                "symbol": attrs.get("symbol") or instrument.get("symbol"),
                "asset_class": attrs.get("asset_class")
                or instrument.get("asset_class")
                or "unknown",
                "currency": attrs.get("currency") or instrument.get("currency") or "unknown",
                "quantity": float(attrs.get("quantity", 0.0)),
                "market_value": float(attrs.get("market_value", 0.0)),
                "as_of": str(as_of) if as_of is not None else None,
            }
        )
    rows.sort(key=lambda r: (-abs(r["market_value"]), r["position_id"]))
    return rows


def _aggregate(positions: list[dict[str, Any]], key_field: str) -> dict[str, dict[str, float]]:
    """Aggregate market values into gross/net/long/short buckets by one key."""
    groups: dict[str, dict[str, float]] = {}
    for position in positions:
        key = str(position.get(key_field) or "unknown")
        bucket = groups.setdefault(
            key, {"gross": 0.0, "net": 0.0, "long": 0.0, "short": 0.0, "positions": 0}
        )
        market_value = float(position.get("market_value", 0.0))
        bucket["gross"] += abs(market_value)
        bucket["net"] += market_value
        if market_value >= 0:
            bucket["long"] += market_value
        else:
            bucket["short"] += market_value
        bucket["positions"] += 1
    for bucket in groups.values():
        for field in ("gross", "net", "long", "short"):
            bucket[field] = round(bucket[field], 2)
    return dict(sorted(groups.items()))


def register(registry: ToolRegistry, runtime: TwinRuntime) -> None:
    """Register the risk tool pack on the given registry against the given runtime."""

    # -- observe_positions ------------------------------------------------------

    def observe_positions(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        positions = _load_positions(runtime)
        account_id = arguments.get("account_id")
        asset_class = arguments.get("asset_class")
        if account_id is not None:
            positions = [p for p in positions if p["account_id"] == account_id]
        if asset_class is not None:
            positions = [p for p in positions if p["asset_class"] == asset_class]
        limit = int(arguments.get("limit", 100))
        return {
            "count": len(positions),
            "total_market_value": round(sum(p["market_value"] for p in positions), 2),
            "total_gross_market_value": round(
                sum(abs(p["market_value"]) for p in positions), 2
            ),
            "positions": positions[:limit],
            "provenance": [provenance_entry("twin_core.graph", "entity_type:position")],
        }

    registry.register(
        ToolSpec(
            name="observe_positions",
            description=(
                "List positions held in the twin, enriched with instrument symbol, asset "
                "class and currency, sorted by absolute market value. Optionally filter by "
                "account or asset class."
            ),
            input_schema=object_schema(
                {
                    "account_id": {
                        "type": "string",
                        "description": "Restrict to one account.",
                    },
                    "asset_class": {
                        "type": "string",
                        "description": "Restrict to one asset class, e.g. 'equity'.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Maximum number of positions to return (default 100).",
                    },
                }
            ),
            band=ToolBand.observe,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.read,
            owner=OWNER,
        ),
        observe_positions,
    )

    # -- observe_exposures ------------------------------------------------------

    def observe_exposures(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        positions = _load_positions(runtime)
        group_by = arguments.get("by", "both")
        data: dict[str, Any] = {
            "totals": {
                "gross": round(sum(abs(p["market_value"]) for p in positions), 2),
                "net": round(sum(p["market_value"] for p in positions), 2),
                "positions": len(positions),
            },
            "provenance": [provenance_entry("twin_core.graph", "entity_type:position")],
        }
        if group_by in {"asset_class", "both"}:
            data["by_asset_class"] = _aggregate(positions, "asset_class")
        if group_by in {"currency", "both"}:
            data["by_currency"] = _aggregate(positions, "currency")
        return data

    registry.register(
        ToolSpec(
            name="observe_exposures",
            description=(
                "Aggregate current positions into gross/net/long/short exposures by asset "
                "class and by currency, plus portfolio totals."
            ),
            input_schema=object_schema(
                {
                    "by": {
                        "type": "string",
                        "enum": ["asset_class", "currency", "both"],
                        "description": "Aggregation dimension (default 'both').",
                    }
                }
            ),
            band=ToolBand.observe,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.read,
            owner=OWNER,
        ),
        observe_exposures,
    )

    # -- observe_limit_utilisation -----------------------------------------------

    def observe_limit_utilisation(
        arguments: dict[str, Any], context: CallContext
    ) -> dict[str, Any]:
        warn_threshold = float(arguments.get("warn_threshold", 0.75))
        positions = _load_positions(runtime)
        by_asset_class = _aggregate(positions, "asset_class")
        by_currency = _aggregate(positions, "currency")
        total_gross = round(sum(abs(p["market_value"]) for p in positions), 2)

        rows: list[dict[str, Any]] = []
        for ref, attrs in entities_of_type(runtime.graph, "limit"):
            scope_type = str(attrs.get("scope_type", "global"))
            scope_value = str(attrs.get("scope_value", ""))
            if arguments.get("scope_type") and scope_type != arguments["scope_type"]:
                continue
            limit_value = float(attrs.get("limit_value", 0.0))
            if scope_type == "asset_class":
                exposure = by_asset_class.get(scope_value, {}).get("gross", 0.0)
            elif scope_type == "currency":
                exposure = by_currency.get(scope_value, {}).get("gross", 0.0)
            else:
                exposure = total_gross
            if limit_value > 0:
                utilisation = round(exposure / limit_value, 4)
                if utilisation >= 1.0:
                    status = "breach"
                elif utilisation >= warn_threshold:
                    status = "warning"
                else:
                    status = "ok"
            else:
                utilisation = None
                status = "invalid_limit"
            rows.append(
                {
                    "limit_id": ref.entity_id,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "limit_value": round(limit_value, 2),
                    "exposure": round(float(exposure), 2),
                    "utilisation": utilisation,
                    "status": status,
                }
            )
        rows.sort(key=lambda r: (-(r["utilisation"] or 0.0), r["limit_id"]))
        return {
            "count": len(rows),
            "breaches": sum(1 for r in rows if r["status"] == "breach"),
            "warnings_count": sum(1 for r in rows if r["status"] == "warning"),
            "limits": rows,
            "provenance": [
                provenance_entry("twin_core.graph", "entity_type:limit + entity_type:position")
            ],
        }

    registry.register(
        ToolSpec(
            name="observe_limit_utilisation",
            description=(
                "Measure every risk limit in the twin against live gross exposures and "
                "report utilisation with ok/warning/breach status."
            ),
            input_schema=object_schema(
                {
                    "scope_type": {
                        "type": "string",
                        "enum": ["asset_class", "currency", "global"],
                        "description": "Restrict to limits of one scope type.",
                    },
                    "warn_threshold": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "description": "Utilisation ratio that flags a warning (default 0.75).",
                    },
                }
            ),
            band=ToolBand.observe,
            risk_tier=RiskTier.medium,
            side_effect=SideEffectClass.read,
            owner=OWNER,
        ),
        observe_limit_utilisation,
    )

    # -- simulate_market_shock ----------------------------------------------------

    def simulate_market_shock(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        scenario_name = arguments["scenario_name"]
        shock_pct = float(arguments["shock_pct"])
        scenario = Scenario(
            name=scenario_name,
            kind="stress",
            params={
                "shock_pct": shock_pct,
                "asset_classes": arguments.get("asset_classes"),
                "horizon_days": int(arguments.get("horizon_days", 10)),
                "n_paths": int(arguments.get("n_paths", 2000)),
            },
        )
        seed = arguments.get("seed")
        if seed is None:
            seed = derive_seed(registry.settings.seed, "market", scenario_name, shock_pct)
        result = run_simulator(runtime, "market", scenario, int(seed))
        return simulation_payload(result, scenario, "twin_sim.market")

    registry.register(
        ToolSpec(
            name="simulate_market_shock",
            description=(
                "Run the calibrated market simulator for an instantaneous shock scenario "
                "and return P&L / drawdown metrics with confidence intervals and a "
                "calibration block. Deterministic for identical inputs."
            ),
            input_schema=object_schema(
                {
                    "scenario_name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Human-readable scenario name, e.g. 'equity_down_10'.",
                    },
                    "shock_pct": {
                        "type": "number",
                        "minimum": -90,
                        "maximum": 90,
                        "description": "Instantaneous market move in percent (negative = down).",
                    },
                    "asset_classes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Asset classes the shock applies to (default: all).",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 90,
                        "description": "Re-pricing horizon in business days (default 10).",
                    },
                    "n_paths": {
                        "type": "integer",
                        "minimum": 100,
                        "maximum": 20000,
                        "description": "Monte-Carlo path count (default 2000).",
                    },
                    "seed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Override the deterministic derived seed.",
                    },
                },
                ["scenario_name", "shock_pct"],
            ),
            band=ToolBand.simulate,
            risk_tier=RiskTier.medium,
            side_effect=SideEffectClass.none,
            provenance_required=True,
            owner=OWNER,
        ),
        simulate_market_shock,
    )

    # -- propose_hedge_candidates ---------------------------------------------------

    def propose_hedge_candidates(
        arguments: dict[str, Any], context: CallContext
    ) -> dict[str, Any]:
        positions = _load_positions(runtime)
        exposures = _aggregate(positions, "asset_class")
        target = arguments.get("asset_class")
        hedge_ratio = float(arguments.get("hedge_ratio", 0.8))
        max_cost_bps = arguments.get("max_cost_bps")
        max_candidates = int(arguments.get("max_candidates", 5))

        candidates: list[dict[str, Any]] = []
        for entry in HEDGE_UNIVERSE:
            asset_class = entry["asset_class"]
            if target is not None and asset_class != target:
                continue
            net = float(exposures.get(asset_class, {}).get("net", 0.0))
            if abs(net) < 1e-9:
                continue
            if max_cost_bps is not None and entry["cost_bps"] > float(max_cost_bps):
                continue
            notional = round(abs(net) * hedge_ratio, 2)
            reduction = round(notional * entry["effectiveness"], 2)
            cost = round(notional * entry["cost_bps"] / 1e4, 2)
            score = round(reduction / cost, 4) if cost > 0 else 0.0
            direction = "sell" if net > 0 else "buy"
            candidates.append(
                {
                    "instrument_id": entry["instrument_id"],
                    "instrument_name": entry["name"],
                    "asset_class": asset_class,
                    "direction": direction,
                    "notional": notional,
                    "expected_exposure_reduction": reduction,
                    "estimated_cost": cost,
                    "cost_bps": entry["cost_bps"],
                    "effectiveness": entry["effectiveness"],
                    "score": score,
                    "rationale": (
                        f"{direction} {entry['name']} notional {notional:,.0f} to offset "
                        f"{hedge_ratio:.0%} of the {asset_class} net exposure of {net:,.0f}; "
                        f"expected reduction {reduction:,.0f} at ~{entry['cost_bps']}bps cost."
                    ),
                }
            )
        candidates.sort(key=lambda c: (-c["score"], c["instrument_id"]))
        candidates = candidates[:max_candidates]
        return {
            "count": len(candidates),
            "ranking_metric": "expected exposure reduction per unit of transaction cost",
            "hedge_ratio": hedge_ratio,
            "candidates": candidates,
            "exposures_considered": exposures,
            "provenance": [
                provenance_entry(
                    "fintwinos.tools.risk", "deterministic hedge ranking over twin exposures"
                )
            ],
        }

    registry.register(
        ToolSpec(
            name="propose_hedge_candidates",
            description=(
                "Rank candidate hedge instruments by expected exposure reduction per unit of "
                "transaction cost against the twin's current net exposures. Deterministic, "
                "rule-based ranking; proposals only — nothing is traded."
            ),
            input_schema=object_schema(
                {
                    "asset_class": {
                        "type": "string",
                        "description": "Only hedge this asset class (default: all exposed).",
                    },
                    "hedge_ratio": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                        "description": "Fraction of net exposure to offset (default 0.8).",
                    },
                    "max_cost_bps": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Exclude instruments costing more than this many bps.",
                    },
                    "max_candidates": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum candidates to return (default 5).",
                    },
                }
            ),
            band=ToolBand.propose,
            risk_tier=RiskTier.medium,
            side_effect=SideEffectClass.none,
            owner=OWNER,
        ),
        propose_hedge_candidates,
    )
