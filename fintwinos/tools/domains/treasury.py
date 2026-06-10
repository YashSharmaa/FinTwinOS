"""Treasury & liquidity tool pack.

Tools registered here:

- ``observe_cash_ladder`` — projected daily cash ladder per currency with troughs.
- ``simulate_liquidity_stress`` — runs the ``treasury`` simulator with the founding-brief
  input schema (high risk tier, provenance required, no side effects).
- ``propose_funding_plan`` — deterministic cheapest-first funding mix for a shortfall.
- ``execute_post_transfer`` — writes a transfer instruction JSON into the local outbox;
  reversible, human approval always required.

Only ``execute_post_transfer`` has a side effect, and that side effect is strictly a
file under ``settings.data_dir / "outbox"`` — never a real payment system.
"""

from __future__ import annotations

from typing import Any

from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import RiskTier, Scenario, SideEffectClass, ToolBand, new_id, utcnow
from fintwinos.tools.domains.common import (
    derive_seed,
    entities_of_type,
    get_entity_attributes,
    now_iso,
    object_schema,
    provenance_entry,
    run_simulator,
    simulation_payload,
    write_outbox_json,
)
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec

OWNER = "treasury"

#: The founding-brief input schema for ``simulate_liquidity_stress`` — kept verbatim.
SIMULATE_LIQUIDITY_STRESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "horizon_days": {"type": "integer", "minimum": 1, "maximum": 30},
        "stress_name": {"type": "string"},
        "currencies": {"type": "array", "items": {"type": "string"}},
        "shock_bps": {"type": "number"},
        "assumptions_version": {"type": "string"},
        "ticket_id": {"type": "string"},
    },
    "required": ["horizon_days", "stress_name", "currencies", "assumptions_version", "ticket_id"],
    "additionalProperties": False,
}

#: Static funding source table used by ``propose_funding_plan``. Costs are indicative
#: all-in spreads in basis points per annum; capacities are per-source headroom.
FUNDING_SOURCES: list[dict[str, Any]] = [
    {
        "source": "overnight_repo_gc",
        "description": "Overnight general-collateral repo, rolled daily",
        "cost_bps": 18.0,
        "capacity": 150_000_000.0,
        "max_tenor_days": 7,
    },
    {
        "source": "interbank_deposit",
        "description": "Unsecured interbank term deposit",
        "cost_bps": 24.0,
        "capacity": 100_000_000.0,
        "max_tenor_days": 14,
    },
    {
        "source": "fx_swap",
        "description": "FX swap raising the shortfall currency against surplus currency",
        "cost_bps": 29.0,
        "capacity": 200_000_000.0,
        "max_tenor_days": 30,
    },
    {
        "source": "committed_credit_facility",
        "description": "Drawdown on the committed revolving credit facility",
        "cost_bps": 55.0,
        "capacity": 400_000_000.0,
        "max_tenor_days": 30,
    },
]


def _build_ladders(
    runtime: TwinRuntime, currencies: list[str] | None, horizon_days: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build the projected cash ladder per currency from twin state.

    Prefers ``cash_ladder`` graph entities (attributes: ``currency``,
    ``opening_balance`` and ``buckets`` — a list of ``{day, inflow, outflow}``).
    Falls back to the latest ``*.cash.<CCY>`` timeseries balances (flat ladder, with
    a warning) when no ladder entities exist.

    Returns:
        ``(ladders, warnings)`` where each ladder carries the daily rows plus the
        trough (minimum projected closing balance) and the day it occurs.
    """
    wanted = {c.upper() for c in currencies} if currencies else None
    ladders: list[dict[str, Any]] = []
    warnings: list[str] = []

    entries = entities_of_type(runtime.graph, "cash_ladder")
    if entries:
        for ref, attrs in entries:
            currency = str(attrs.get("currency", ref.entity_id)).upper()
            if wanted is not None and currency not in wanted:
                continue
            opening = float(attrs.get("opening_balance", 0.0))
            raw_buckets = attrs.get("buckets") or []
            by_day: dict[int, dict[str, Any]] = {}
            for index, bucket in enumerate(raw_buckets):
                by_day[int(bucket.get("day", index + 1))] = bucket
            running = opening
            trough, trough_day = opening, 0
            rows: list[dict[str, Any]] = []
            for day in range(1, horizon_days + 1):
                bucket = by_day.get(day, {})
                inflow = float(bucket.get("inflow", 0.0))
                outflow = float(bucket.get("outflow", 0.0))
                net = inflow - outflow
                running += net
                rows.append(
                    {
                        "day": day,
                        "inflow": round(inflow, 2),
                        "outflow": round(outflow, 2),
                        "net": round(net, 2),
                        "closing_balance": round(running, 2),
                    }
                )
                if running < trough:
                    trough, trough_day = running, day
            ladders.append(
                {
                    "currency": currency,
                    "opening_balance": round(opening, 2),
                    "ladder": rows,
                    "trough": round(trough, 2),
                    "trough_day": trough_day,
                }
            )
        return ladders, warnings

    for key in sorted(runtime.timeseries.keys()):
        parts = key.split(".")
        if len(parts) < 2 or "cash" not in {p.lower() for p in parts[:-1]}:
            continue
        currency = parts[-1].upper()
        if len(currency) != 3:
            continue
        if wanted is not None and currency not in wanted:
            continue
        latest = runtime.timeseries.latest(key)
        if latest is None:
            continue
        opening = float(latest[1])
        rows = [
            {"day": day, "inflow": 0.0, "outflow": 0.0, "net": 0.0,
             "closing_balance": round(opening, 2)}
            for day in range(1, horizon_days + 1)
        ]
        ladders.append(
            {
                "currency": currency,
                "opening_balance": round(opening, 2),
                "ladder": rows,
                "trough": round(opening, 2),
                "trough_day": 0,
            }
        )
    if ladders:
        warnings.append(
            "no cash_ladder entities found in the twin graph; ladder built from latest "
            "cash timeseries balances with no projected flows"
        )
    else:
        warnings.append(
            "no cash ladder data found in the twin (neither cash_ladder graph entities "
            "nor cash.* timeseries)"
        )
    return ladders, warnings


def register(registry: ToolRegistry, runtime: TwinRuntime) -> None:
    """Register the treasury tool pack on the given registry against the runtime."""

    # -- observe_cash_ladder -------------------------------------------------------

    def observe_cash_ladder(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        horizon_days = int(arguments.get("horizon_days", 7))
        currencies = arguments.get("currencies")
        ladders, warnings = _build_ladders(runtime, currencies, horizon_days)
        return {
            "as_of": now_iso(),
            "horizon_days": horizon_days,
            "currencies": [ladder["currency"] for ladder in ladders],
            "ladders": ladders,
            "warnings": warnings,
            "provenance": [provenance_entry("twin_core.graph", "entity_type:cash_ladder")],
        }

    registry.register(
        ToolSpec(
            name="observe_cash_ladder",
            description=(
                "Projected daily cash ladder per currency: opening balance, inflows, "
                "outflows, closing balances and the trough (worst projected balance) "
                "with the day it occurs."
            ),
            input_schema=object_schema(
                {
                    "currencies": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[A-Za-z]{3}$"},
                        "minItems": 1,
                        "description": "Restrict to these ISO currency codes.",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "description": "Projection horizon in days (default 7).",
                    },
                }
            ),
            band=ToolBand.observe,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.read,
            owner=OWNER,
        ),
        observe_cash_ladder,
    )

    # -- simulate_liquidity_stress ---------------------------------------------------

    def simulate_liquidity_stress(
        arguments: dict[str, Any], context: CallContext
    ) -> dict[str, Any]:
        stress_name = arguments["stress_name"]
        ticket_id = arguments["ticket_id"]
        scenario = Scenario(
            name=stress_name,
            kind="stress",
            params={
                "horizon_days": int(arguments["horizon_days"]),
                "currencies": [str(c).upper() for c in arguments["currencies"]],
                "shock_bps": float(arguments.get("shock_bps", 0.0)),
                "ticket_id": ticket_id,
            },
            assumptions_version=arguments["assumptions_version"],
        )
        seed = derive_seed(
            registry.settings.seed,
            "treasury",
            stress_name,
            ticket_id,
            arguments["horizon_days"],
            arguments.get("shock_bps", 0.0),
        )
        result = run_simulator(runtime, "treasury", scenario, seed)
        payload = simulation_payload(result, scenario, "twin_sim.treasury")
        payload["ticket_id"] = ticket_id
        return payload

    registry.register(
        ToolSpec(
            name="simulate_liquidity_stress",
            description=(
                "Run the calibrated treasury simulator for a named liquidity stress over a "
                "1-30 day horizon and return survival metrics with confidence intervals and "
                "a calibration block. Read-only; deterministic for identical inputs."
            ),
            input_schema=SIMULATE_LIQUIDITY_STRESS_SCHEMA,
            band=ToolBand.simulate,
            risk_tier=RiskTier.high,
            side_effect=SideEffectClass.none,
            provenance_required=True,
            owner=OWNER,
        ),
        simulate_liquidity_stress,
    )

    # -- propose_funding_plan ----------------------------------------------------------

    def propose_funding_plan(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        currency = str(arguments["currency"]).upper()
        horizon_days = int(arguments.get("horizon_days", 7))
        buffer_ratio = float(arguments.get("buffer_ratio", 0.0))

        warnings: list[str] = []
        amount = arguments.get("amount")
        shortfall_source = "caller-supplied amount"
        if amount is None:
            ladders, ladder_warnings = _build_ladders(runtime, [currency], horizon_days)
            warnings.extend(ladder_warnings)
            trough = float(ladders[0]["trough"]) if ladders else 0.0
            amount = max(0.0, -trough)
            shortfall_source = (
                f"projected {currency} trough of {trough:,.0f} over {horizon_days} days"
            )
        amount = float(amount)
        target = round(amount * (1.0 + buffer_ratio), 2)

        if target <= 0:
            return {
                "currency": currency,
                "horizon_days": horizon_days,
                "funding_required": False,
                "shortfall": 0.0,
                "basis": shortfall_source,
                "tranches": [],
                "warnings": warnings,
                "provenance": [
                    provenance_entry(
                        "fintwinos.tools.treasury", "deterministic funding plan (no shortfall)"
                    )
                ],
            }

        eligible = [s for s in FUNDING_SOURCES if s["max_tenor_days"] >= horizon_days]
        eligible.sort(key=lambda s: (s["cost_bps"], s["source"]))
        remaining = target
        tranches: list[dict[str, Any]] = []
        for source in eligible:
            if remaining <= 0:
                break
            take = round(min(remaining, float(source["capacity"])), 2)
            if take <= 0:
                continue
            tranches.append(
                {
                    "source": source["source"],
                    "description": source["description"],
                    "amount": take,
                    "cost_bps": source["cost_bps"],
                    "tenor_days": horizon_days,
                }
            )
            remaining = round(remaining - take, 2)
        if remaining > 0:
            warnings.append(
                f"uncovered shortfall of {remaining:,.0f} {currency}: eligible source "
                "capacity is exhausted — escalate to the treasury desk"
            )
        raised = round(sum(t["amount"] for t in tranches), 2)
        blended_bps = (
            round(sum(t["amount"] * t["cost_bps"] for t in tranches) / raised, 4)
            if raised > 0
            else 0.0
        )
        estimated_cost = round(raised * blended_bps / 1e4 * horizon_days / 360.0, 2)
        return {
            "currency": currency,
            "horizon_days": horizon_days,
            "funding_required": True,
            "shortfall": round(amount, 2),
            "buffer_ratio": buffer_ratio,
            "target_amount": target,
            "basis": shortfall_source,
            "tranches": tranches,
            "amount_raised": raised,
            "uncovered": remaining if remaining > 0 else 0.0,
            "blended_cost_bps": blended_bps,
            "estimated_carry_cost": estimated_cost,
            "warnings": warnings,
            "provenance": [
                provenance_entry(
                    "fintwinos.tools.treasury",
                    "deterministic cheapest-first funding mix over the static source table",
                )
            ],
        }

    registry.register(
        ToolSpec(
            name="propose_funding_plan",
            description=(
                "Propose a cheapest-first funding mix to cover a cash shortfall in one "
                "currency. If no amount is supplied, the shortfall is derived from the "
                "projected cash-ladder trough. Deterministic; proposal only — no funds move."
            ),
            input_schema=object_schema(
                {
                    "currency": {
                        "type": "string",
                        "pattern": "^[A-Za-z]{3}$",
                        "description": "ISO currency code of the shortfall.",
                    },
                    "amount": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Shortfall to fund; default derives from the ladder trough.",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "description": "Funding tenor in days (default 7).",
                    },
                    "buffer_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 0.5,
                        "description": "Extra buffer on top of the shortfall (default 0).",
                    },
                },
                ["currency"],
            ),
            band=ToolBand.propose,
            risk_tier=RiskTier.medium,
            side_effect=SideEffectClass.none,
            owner=OWNER,
        ),
        propose_funding_plan,
    )

    # -- execute_post_transfer ----------------------------------------------------------

    def execute_post_transfer(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        warnings: list[str] = []
        for field in ("from_account", "to_account"):
            account_id = str(arguments[field])
            if get_entity_attributes(runtime.graph, "account", account_id) is None:
                warnings.append(f"{field} '{account_id}' is not a known account in the twin")

        instruction_id = new_id("txi")
        approval = context.approval
        record = {
            "instruction_id": instruction_id,
            "instruction_type": "payment_transfer",
            "from_account": arguments["from_account"],
            "to_account": arguments["to_account"],
            "amount": float(arguments["amount"]),
            "currency": str(arguments["currency"]).upper(),
            "value_date": arguments.get("value_date") or utcnow().date().isoformat(),
            "reference": arguments.get("reference", ""),
            "ticket_id": arguments["ticket_id"],
            "requested_by": context.caller,
            "approved_by": approval.granted_by if approval else None,
            "approval_token_id": approval.token_id if approval else None,
            "status": "queued",
            "reversible": True,
            "created_at": now_iso(),
        }
        path = write_outbox_json(
            registry.settings, f"transfer_{instruction_id}.json", record
        )
        runtime.audit.append(
            context.caller,
            "outbox.transfer_queued",
            {
                "instruction_id": instruction_id,
                "path": str(path),
                "ticket_id": arguments["ticket_id"],
                "amount": record["amount"],
                "currency": record["currency"],
            },
        )
        return {
            "instruction_id": instruction_id,
            "status": "queued",
            "outbox_path": str(path),
            "warnings": warnings,
            "provenance": [
                provenance_entry("fintwinos.outbox", f"transfer:{instruction_id}")
            ],
        }

    registry.register(
        ToolSpec(
            name="execute_post_transfer",
            description=(
                "Queue an internal cash transfer instruction by writing a JSON document to "
                "the local outbox (never a real payment rail). Reversible while queued; "
                "always requires a human approval token, a policy allow rule and the "
                "execute-band enable flag."
            ),
            input_schema=object_schema(
                {
                    "from_account": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Source account identifier.",
                    },
                    "to_account": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Destination account identifier.",
                    },
                    "amount": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Transfer amount in the instruction currency.",
                    },
                    "currency": {
                        "type": "string",
                        "pattern": "^[A-Za-z]{3}$",
                        "description": "ISO currency code.",
                    },
                    "value_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Settlement date YYYY-MM-DD (default: today).",
                    },
                    "reference": {
                        "type": "string",
                        "maxLength": 140,
                        "description": "Payment reference text.",
                    },
                    "ticket_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Change/incident ticket authorising this transfer.",
                    },
                },
                ["from_account", "to_account", "amount", "currency", "ticket_id"],
            ),
            band=ToolBand.execute,
            risk_tier=RiskTier.high,
            side_effect=SideEffectClass.reversible,
            requires_human_approval=True,
            provenance_required=True,
            idempotent=False,
            owner=OWNER,
        ),
        execute_post_transfer,
    )
