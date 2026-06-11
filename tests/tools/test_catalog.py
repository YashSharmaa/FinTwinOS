"""Catalog-level tests: size, bands, governance metadata and shared wiring."""

from __future__ import annotations

from fintwinos.core.types import SideEffectClass, ToolBand
from fintwinos.policy.gates import PolicyGate
from fintwinos.policy.rbac import RbacGate, Role
from fintwinos.tools.catalog import build_default_registry, catalog_summary

EXPECTED_TOOLS = {
    # risk
    "observe_positions",
    "observe_exposures",
    "observe_limit_utilisation",
    "simulate_market_shock",
    "propose_hedge_candidates",
    # treasury
    "observe_cash_ladder",
    "simulate_liquidity_stress",
    "propose_funding_plan",
    "execute_post_transfer",
    # compliance
    "observe_alerts",
    "observe_case",
    "observe_entity_graph",
    "simulate_alert_threshold",
    "propose_case_narrative",
    "execute_close_case",
    # customer_ops
    "observe_queue_state",
    "simulate_staffing_change",
    "propose_response_draft",
    "execute_send_response",
    # filings
    "observe_filing_search",
    "observe_policy_lookup",
}


def test_catalog_has_at_least_18_tools_across_all_bands(registry):
    assert len(registry) >= 18
    bands = {spec.band for spec in registry.list_specs()}
    assert bands == {ToolBand.observe, ToolBand.simulate, ToolBand.propose, ToolBand.execute}


def test_catalog_contains_every_contracted_tool(registry):
    names = {spec.name for spec in registry.list_specs()}
    assert EXPECTED_TOOLS <= names


def test_band_distribution(registry):
    by_band = {band: registry.list_specs(band=band) for band in ToolBand}
    assert len(by_band[ToolBand.observe]) >= 10
    assert len(by_band[ToolBand.simulate]) >= 4
    assert len(by_band[ToolBand.propose]) >= 4
    assert len(by_band[ToolBand.execute]) >= 3


def test_registry_shares_runtime_audit(fake_runtime, settings):
    registry = build_default_registry(fake_runtime, settings=settings)
    assert registry.audit is fake_runtime.audit
    actions = [record.action for record in registry.audit.records()]
    assert "catalog.built" in actions


def test_every_schema_is_closed_with_explicit_required(registry):
    for spec in registry.list_specs():
        schema = spec.input_schema
        assert schema["type"] == "object", spec.name
        assert schema.get("additionalProperties") is False, spec.name
        assert isinstance(schema.get("required"), list), spec.name
        assert isinstance(schema.get("properties"), dict), spec.name


def test_tool_names_match_band_prefixes(registry):
    for spec in registry.list_specs():
        assert spec.name.startswith(f"{spec.band.value}_"), spec.name


def test_public_catalog_carries_governance_extensions(registry):
    catalog = registry.public_catalog()
    assert len(catalog) == len(registry)
    for entry in catalog:
        for field in (
            "x_band",
            "x_risk_tier",
            "x_side_effect",
            "x_requires_human_approval",
            "x_provenance_required",
            "x_idempotent",
        ):
            assert field in entry, (entry["name"], field)


def test_execute_tools_declare_approval_and_real_side_effects(registry):
    for spec in registry.list_specs(band=ToolBand.execute):
        assert spec.requires_human_approval is True, spec.name
        assert spec.side_effect in {
            SideEffectClass.reversible,
            SideEffectClass.irreversible,
        }, spec.name


def test_non_execute_tools_are_side_effect_free(registry):
    for band in (ToolBand.observe, ToolBand.simulate, ToolBand.propose):
        for spec in registry.list_specs(band=band):
            assert spec.side_effect in {SideEffectClass.none, SideEffectClass.read}, spec.name


def test_default_policy_gate_attached_and_custom_gate_respected(fake_runtime, settings):
    default = build_default_registry(fake_runtime, settings=settings)
    # The default gate chain is RBAC wrapping the conservative policy gate.
    assert isinstance(default.policy_gate, RbacGate)
    assert isinstance(default.policy_gate.inner, PolicyGate)
    assert default.policy_gate.default_role == Role.analyst

    custom = PolicyGate(rules=[])
    registry = build_default_registry(fake_runtime, policy_gate=custom, settings=settings)
    assert registry.policy_gate is custom


def test_owners_cover_all_five_domains(registry):
    owners = {spec.owner for spec in registry.list_specs()}
    assert owners == {"risk", "treasury", "compliance", "customer_ops", "filings"}


def test_catalog_summary_shape(registry):
    summary = catalog_summary(registry)
    assert summary["tool_count"] == len(registry)
    assert set(summary["by_band"]) == {"observe", "simulate", "propose", "execute"}
    assert set(summary["by_owner"]) == {"risk", "treasury", "compliance", "customer_ops", "filings"}
