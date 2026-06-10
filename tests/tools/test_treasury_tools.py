"""Treasury domain tests: cash ladder, liquidity stress contract, funding plan,
and the full execute_post_transfer governance ladder (disabled flag, approval,
policy allow rule, dry-run, outbox write, idempotency)."""

from __future__ import annotations

import json

from conftest import allow_execute_gate, approval_for, build_fake_runtime

from fintwinos.core.types import RiskTier, SideEffectClass
from fintwinos.tools.catalog import build_default_registry
from fintwinos.tools.registry import CallContext

CTX = CallContext(caller="test-treasury")

VALID_STRESS = {
    "horizon_days": 14,
    "stress_name": "ccy_run_usd",
    "currencies": ["USD", "EUR"],
    "shock_bps": 75.0,
    "assumptions_version": "v1",
    "ticket_id": "TCK-1001",
}

TRANSFER = {
    "from_account": "ACC_TRD_2",
    "to_account": "ACC_TRD_1",
    "amount": 250_000.0,
    "currency": "USD",
    "ticket_id": "TCK-2002",
    "reference": "intraday liquidity rebalance",
}


# -- observe_cash_ladder ----------------------------------------------------------


async def test_observe_cash_ladder_finds_trough(registry):
    result = await registry.call("observe_cash_ladder", {"horizon_days": 7}, CTX)
    assert result.ok, result.error
    ladders = {ladder["currency"]: ladder for ladder in result.data["ladders"]}
    assert set(ladders) == {"USD", "EUR"}
    usd = ladders["USD"]
    assert usd["opening_balance"] == 5_000_000.0
    assert usd["trough"] == -500_000.0
    assert usd["trough_day"] == 3
    assert len(usd["ladder"]) == 7
    day3 = usd["ladder"][2]
    assert day3["closing_balance"] == -500_000.0


async def test_observe_cash_ladder_currency_filter(registry):
    result = await registry.call("observe_cash_ladder", {"currencies": ["EUR"]}, CTX)
    assert result.ok
    assert result.data["currencies"] == ["EUR"]
    assert result.data["ladders"][0]["trough_day"] == 0  # EUR never dips below opening


# -- simulate_liquidity_stress -------------------------------------------------------


def test_simulate_liquidity_stress_spec_governance(registry):
    spec = registry.spec("simulate_liquidity_stress")
    assert spec.risk_tier == RiskTier.high
    assert spec.side_effect == SideEffectClass.none
    assert spec.provenance_required is True
    schema = spec.input_schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "horizon_days", "stress_name", "currencies", "assumptions_version", "ticket_id"
    }
    assert schema["properties"]["horizon_days"] == {
        "type": "integer", "minimum": 1, "maximum": 30
    }
    assert schema["properties"]["shock_bps"] == {"type": "number"}
    assert schema["properties"]["currencies"]["type"] == "array"
    public = spec.to_public_dict()
    assert public["x_risk_tier"] == "high"
    assert public["x_side_effect"] == "none"
    assert public["x_provenance_required"] is True


async def test_simulate_liquidity_stress_end_to_end_returns_cis(registry):
    result = await registry.call("simulate_liquidity_stress", VALID_STRESS, CTX)
    assert result.ok, result.error
    data = result.data
    assert data["metrics"]
    assert data["confidence"]
    for low, high in data["confidence"].values():
        assert low <= high
    assert data["calibration"]["method"] == "bootstrap"
    assert data["ticket_id"] == "TCK-1001"
    assert data["result"]["scenario_name"] == "ccy_run_usd"
    assert data["assumptions_version"] == "v1"
    assert result.provenance and result.provenance[0].source_system == "twin_sim.treasury"


async def test_simulate_liquidity_stress_is_deterministic(registry):
    first = await registry.call("simulate_liquidity_stress", VALID_STRESS, CTX)
    second = await registry.call("simulate_liquidity_stress", VALID_STRESS, CTX)
    assert first.data["metrics"] == second.data["metrics"]
    assert first.data["confidence"] == second.data["confidence"]


async def test_simulate_liquidity_stress_schema_is_strict(registry):
    missing = {k: v for k, v in VALID_STRESS.items() if k != "ticket_id"}
    result = await registry.call("simulate_liquidity_stress", missing, CTX)
    assert not result.ok and "failed validation" in result.error

    extra = dict(VALID_STRESS, seed=1)  # seed is NOT part of the contracted schema
    result = await registry.call("simulate_liquidity_stress", extra, CTX)
    assert not result.ok and "failed validation" in result.error

    for bad_horizon in (0, 31):
        result = await registry.call(
            "simulate_liquidity_stress", dict(VALID_STRESS, horizon_days=bad_horizon), CTX
        )
        assert not result.ok and "failed validation" in result.error

    no_shock = {k: v for k, v in VALID_STRESS.items() if k != "shock_bps"}
    result = await registry.call("simulate_liquidity_stress", no_shock, CTX)
    assert result.ok  # shock_bps is optional


# -- propose_funding_plan ----------------------------------------------------------------


async def test_propose_funding_plan_with_explicit_amount(registry):
    result = await registry.call(
        "propose_funding_plan",
        {"currency": "USD", "amount": 200_000_000.0, "horizon_days": 7},
        CTX,
    )
    assert result.ok, result.error
    data = result.data
    assert data["funding_required"] is True
    assert data["amount_raised"] == 200_000_000.0
    assert [t["source"] for t in data["tranches"]] == [
        "overnight_repo_gc", "interbank_deposit",
    ]
    assert data["tranches"][0]["amount"] == 150_000_000.0
    assert data["tranches"][1]["amount"] == 50_000_000.0
    assert data["blended_cost_bps"] == 19.5
    assert data["uncovered"] == 0.0


async def test_propose_funding_plan_derives_shortfall_from_ladder(registry):
    result = await registry.call("propose_funding_plan", {"currency": "USD"}, CTX)
    assert result.ok
    data = result.data
    assert data["funding_required"] is True
    assert data["shortfall"] == 500_000.0  # the USD ladder trough
    assert data["tranches"][0]["source"] == "overnight_repo_gc"


async def test_propose_funding_plan_no_shortfall(registry):
    result = await registry.call("propose_funding_plan", {"currency": "EUR"}, CTX)
    assert result.ok
    assert result.data["funding_required"] is False
    assert result.data["tranches"] == []


# -- execute_post_transfer governance ladder ------------------------------------------------


async def test_execute_blocked_when_band_disabled(registry):
    """Even with an approval token, the hard enable flag wins."""
    context = CallContext(caller="ops", approval=approval_for("execute_post_transfer"))
    result = await registry.call("execute_post_transfer", TRANSFER, context)
    assert not result.ok
    assert result.requires_approval is True
    assert "disabled" in result.error


async def test_execute_blocked_without_approval_token(exec_registry):
    result = await exec_registry.call(
        "execute_post_transfer", TRANSFER, CallContext(caller="ops")
    )
    assert not result.ok
    assert result.requires_approval is True
    assert "approval" in result.error


async def test_execute_blocked_without_policy_allow_rule(fake_runtime, exec_settings):
    """Execute band is default-deny: flag on + token are not enough without an allow rule."""
    registry = build_default_registry(fake_runtime, settings=exec_settings)
    context = CallContext(caller="ops", approval=approval_for("execute_post_transfer"))
    result = await registry.call("execute_post_transfer", TRANSFER, context)
    assert not result.ok
    assert "denied by policy" in result.error


async def test_execute_dry_run_always_permitted_and_writes_nothing(registry, settings):
    context = CallContext(caller="ops", dry_run=True)
    result = await registry.call("execute_post_transfer", TRANSFER, context)
    assert result.ok, result.error
    assert result.data["dry_run"] is True
    assert result.data["would_execute"] == "execute_post_transfer"
    outbox = settings.data_dir / "outbox"
    assert not outbox.exists() or not list(outbox.iterdir())


async def test_execute_approved_path_writes_outbox_instruction(exec_registry, exec_settings):
    context = CallContext(
        caller="treasury-ops",
        ticket_id="TCK-2002",
        approval=approval_for("execute_post_transfer"),
    )
    result = await exec_registry.call("execute_post_transfer", TRANSFER, context)
    assert result.ok, result.error
    assert result.data["status"] == "queued"

    outbox = exec_settings.data_dir / "outbox"
    files = sorted(outbox.glob("transfer_*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["amount"] == 250_000.0
    assert record["currency"] == "USD"
    assert record["from_account"] == "ACC_TRD_2"
    assert record["approved_by"] == "approver@bank.example"
    assert record["reversible"] is True

    # The shared audit chain recorded the side effect and remains intact.
    actions = [r.action for r in exec_registry.audit.records()]
    assert "outbox.transfer_queued" in actions
    assert exec_registry.audit.verify()


async def test_execute_idempotency_key_replays_without_duplicate_writes(
    exec_registry, exec_settings
):
    context = CallContext(
        caller="treasury-ops",
        approval=approval_for("execute_post_transfer"),
        idempotency_key="transfer-TCK-2002",
    )
    first = await exec_registry.call("execute_post_transfer", TRANSFER, context)
    second = await exec_registry.call("execute_post_transfer", TRANSFER, context)
    assert first.ok and second.ok
    assert first.data["instruction_id"] == second.data["instruction_id"]
    files = list((exec_settings.data_dir / "outbox").glob("transfer_*.json"))
    assert len(files) == 1


async def test_execute_post_transfer_schema_rejects_bad_currency(exec_registry):
    context = CallContext(caller="ops", approval=approval_for("execute_post_transfer"))
    result = await exec_registry.call(
        "execute_post_transfer", dict(TRANSFER, currency="DOLLARS"), context
    )
    assert not result.ok and "failed validation" in result.error


async def test_execute_warns_on_unknown_account_but_still_queues(exec_settings):
    runtime = build_fake_runtime()
    registry = build_default_registry(
        runtime, policy_gate=allow_execute_gate("execute_*"), settings=exec_settings
    )
    context = CallContext(caller="ops", approval=approval_for("execute_post_transfer"))
    result = await registry.call(
        "execute_post_transfer", dict(TRANSFER, to_account="ACC_GHOST"), context
    )
    assert result.ok
    assert any("ACC_GHOST" in warning for warning in result.data["warnings"])
