"""Tests for YAML policy packs: loading, validation, operators and gating."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from fintwinos.core.types import RiskTier, SideEffectClass, ToolBand
from fintwinos.policy.gates import RuleEffect, default_rules
from fintwinos.policy.rules import (
    PACKS_DIR,
    PolicyPackError,
    YamlRule,
    gate_from_packs,
    load_default_packs,
    load_pack,
    load_pack_model,
)
from fintwinos.tools.registry import ToolSpec

EXPECTED_PACKS = {"baseline", "treasury", "compliance", "trading_limits"}


def exec_spec(
    name: str = "execute_post_transfer",
    tier: RiskTier = RiskTier.high,
    side_effect: SideEffectClass = SideEffectClass.reversible,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test execute tool",
        input_schema={"type": "object"},
        band=ToolBand.execute,
        risk_tier=tier,
        side_effect=side_effect,
        requires_human_approval=True,
    )


def test_deny_irreversible_critical_targets_only_irreversible():
    """The shipped deny rule fires on irreversible critical tools, not reversible ones."""
    deny = next(r for r in default_rules() if r.name == "deny-irreversible-critical")
    irreversible = exec_spec("execute_wire", RiskTier.critical, SideEffectClass.irreversible)
    reversible = exec_spec("execute_close_case", RiskTier.critical, SideEffectClass.reversible)
    assert deny.matches(irreversible, {}) is True
    assert deny.matches(reversible, {}) is False


def test_yaml_baseline_deny_rule_honours_side_effects():
    """The baseline YAML pack's deny rule also distinguishes side-effect classes."""
    pack = load_pack_model(PACKS_DIR / "baseline.yaml")
    deny = next(r for r in pack.rules if r.name == "deny-irreversible-critical")
    irreversible = exec_spec("execute_wire", RiskTier.critical, SideEffectClass.irreversible)
    reversible = exec_spec("execute_close_case", RiskTier.critical, SideEffectClass.reversible)
    assert deny.matches(irreversible, {}) is True
    assert deny.matches(reversible, {}) is False


def write_pack(tmp_path, text: str):
    path = tmp_path / "pack.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- shipped packs -----------------------------------------------------------------


def test_all_shipped_packs_load():
    packs = load_default_packs()
    assert set(packs) == EXPECTED_PACKS
    for name, rules in packs.items():
        assert rules, f"pack '{name}' is empty"
        assert all(isinstance(rule, YamlRule) for rule in rules)


def test_baseline_pack_mirrors_default_rules():
    baseline = load_pack(PACKS_DIR / "baseline.yaml")
    defaults = default_rules()
    assert [r.name for r in baseline] == [r.name for r in defaults]
    for yaml_rule, default in zip(baseline, defaults, strict=True):
        assert yaml_rule.effect == default.effect
        assert yaml_rule.bands == default.bands
        assert yaml_rule.min_risk_tier == default.min_risk_tier


def test_packs_round_trip_through_yaml(tmp_path):
    for source in sorted(PACKS_DIR.glob("*.yaml")):
        pack = load_pack_model(source)
        rewritten = tmp_path / source.name
        rewritten.write_text(pack.to_yaml(), encoding="utf-8")
        reloaded = load_pack_model(rewritten)
        assert reloaded.name == pack.name
        assert reloaded.version == pack.version
        assert reloaded.description == pack.description
        assert reloaded.rules == pack.rules


# --- operator matching ----------------------------------------------------------------


def rule_with(conditions) -> YamlRule:
    return YamlRule(name="r", effect=RuleEffect.allow, conditions=conditions)


def test_lte_gte_thresholds_are_inclusive():
    spec = SimpleNamespace()  # bands/tools/tier unset, so spec is never inspected
    lte = rule_with({"amount": {"lte": 100}})
    assert lte.matches(spec, {"amount": 100})
    assert lte.matches(spec, {"amount": 99.5})
    assert not lte.matches(spec, {"amount": 100.01})
    gte = rule_with({"amount": {"gte": 100}})
    assert gte.matches(spec, {"amount": 100})
    assert not gte.matches(spec, {"amount": 99.999})


def test_lt_gt_eq_ne_in_operators():
    spec = SimpleNamespace()
    assert rule_with({"x": {"lt": 5}}).matches(spec, {"x": 4})
    assert not rule_with({"x": {"lt": 5}}).matches(spec, {"x": 5})
    assert rule_with({"x": {"gt": 5}}).matches(spec, {"x": 6})
    assert rule_with({"x": {"eq": "USD"}}).matches(spec, {"x": "USD"})
    assert rule_with({"x": {"ne": "USD"}}).matches(spec, {"x": "EUR"})
    assert rule_with({"x": {"in": ["USD", "EUR"]}}).matches(spec, {"x": "EUR"})
    assert not rule_with({"x": {"in": ["USD", "EUR"]}}).matches(spec, {"x": "JPY"})


def test_scalar_condition_normalises_to_eq():
    rule = rule_with({"currency": "USD"})
    assert rule.conditions == {"currency": {"eq": "USD"}}
    assert rule.matches(SimpleNamespace(), {"currency": "USD"})
    assert not rule.matches(SimpleNamespace(), {"currency": "EUR"})


def test_missing_field_and_non_numeric_values_never_match():
    spec = SimpleNamespace()
    rule = rule_with({"amount": {"lte": 100}})
    assert not rule.matches(spec, {})  # missing field
    assert not rule.matches(spec, {"amount": "not-a-number"})


def test_dotted_path_reaches_nested_arguments():
    rule = rule_with({"details.amount": {"gte": 5}})
    assert rule.matches(SimpleNamespace(), {"details": {"amount": 7}})
    assert not rule.matches(SimpleNamespace(), {"details": {"amount": 3}})
    assert not rule.matches(SimpleNamespace(), {"details": "flat"})


def test_multiple_operators_on_one_field_all_must_hold():
    rule = rule_with({"amount": {"gte": 10, "lte": 20}})
    assert rule.matches(SimpleNamespace(), {"amount": 15})
    assert not rule.matches(SimpleNamespace(), {"amount": 25})


def test_in_operator_requires_list_operand():
    with pytest.raises(ValidationError):
        rule_with({"x": {"in": 5}})


def test_empty_operator_mapping_rejected():
    with pytest.raises(ValidationError):
        rule_with({"x": {}})


# --- gating through shipped packs --------------------------------------------------------


def test_treasury_thresholds_gate_amounts():
    gate = gate_from_packs("treasury")
    spec = exec_spec("execute_post_transfer")

    below = gate.check_tool_call(spec, {"amount": 50_000, "currency": "USD"})
    assert below.allowed and below.requires_human_review

    at_limit = gate.check_tool_call(spec, {"amount": 250_000, "currency": "GBP"})
    assert at_limit.allowed and at_limit.requires_human_review

    above = gate.check_tool_call(spec, {"amount": 250_000.01, "currency": "USD"})
    assert not above.allowed
    assert "treasury-deny-large-transfers" in above.matched_rules

    bad_currency = gate.check_tool_call(spec, {"amount": 50_000, "currency": "JPY"})
    assert not bad_currency.allowed  # execute band stays default-deny


def test_compliance_pack_respects_recall_floor():
    gate = gate_from_packs("compliance")
    spec = exec_spec("execute_close_case", tier=RiskTier.medium)

    healthy = gate.check_tool_call(
        spec, {"model_recall": 0.95, "disposition": "close", "priority": "low"}
    )
    assert healthy.allowed and healthy.requires_human_review

    weak_detector = gate.check_tool_call(
        spec, {"model_recall": 0.80, "disposition": "close", "priority": "low"}
    )
    assert not weak_detector.allowed
    assert "compliance-deny-closure-below-recall-floor" in weak_detector.matched_rules

    dismiss_high = gate.check_tool_call(
        spec, {"model_recall": 0.99, "disposition": "dismiss", "priority": "critical"}
    )
    assert not dismiss_high.allowed
    assert "compliance-deny-dismiss-high-priority" in dismiss_high.matched_rules


def test_trading_limits_pack():
    gate = gate_from_packs("trading_limits")
    order_spec = exec_spec("execute_place_order")
    cancel_spec = exec_spec("execute_cancel_order")

    bounded = gate.check_tool_call(
        order_spec, {"notional": 500_000, "instrument_restricted": False}
    )
    assert bounded.allowed and bounded.requires_human_review

    oversize = gate.check_tool_call(
        order_spec, {"notional": 1_500_000, "instrument_restricted": False}
    )
    assert not oversize.allowed

    restricted = gate.check_tool_call(
        order_spec, {"notional": 1_000, "instrument_restricted": True}
    )
    assert not restricted.allowed
    assert "trading-deny-restricted-instruments" in restricted.matched_rules

    cancel = gate.check_tool_call(cancel_spec, {"order_id": "ord-1"})
    assert cancel.allowed and cancel.requires_human_review


def test_gate_from_packs_unknown_pack():
    with pytest.raises(PolicyPackError, match="unknown policy pack"):
        gate_from_packs("nonexistent")


# --- validation errors --------------------------------------------------------------------


def test_missing_pack_file_raises(tmp_path):
    with pytest.raises(PolicyPackError, match="not found"):
        load_pack(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path):
    path = write_pack(tmp_path, "rules: [unclosed\nname: broken")
    with pytest.raises(PolicyPackError, match="not valid YAML"):
        load_pack(path)


def test_non_mapping_document_raises(tmp_path):
    path = write_pack(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(PolicyPackError, match="YAML mapping"):
        load_pack(path)


def test_missing_name_raises_with_clear_error(tmp_path):
    path = write_pack(
        tmp_path,
        "rules:\n  - name: r1\n    effect: allow\n",
    )
    with pytest.raises(PolicyPackError, match="'name' is a required property"):
        load_pack(path)


def test_bad_effect_raises_with_clear_error(tmp_path):
    path = write_pack(
        tmp_path,
        "name: bad\nrules:\n  - name: r1\n    effect: maybe\n",
    )
    with pytest.raises(PolicyPackError, match="maybe"):
        load_pack(path)


def test_unknown_operator_raises_with_clear_error(tmp_path):
    path = write_pack(
        tmp_path,
        "name: bad\nrules:\n  - name: r1\n    effect: allow\n    conditions:\n"
        "      amount: {approx: 5}\n",
    )
    with pytest.raises(PolicyPackError, match="approx"):
        load_pack(path)


def test_unknown_rule_key_raises(tmp_path):
    path = write_pack(
        tmp_path,
        "name: bad\nrules:\n  - name: r1\n    effect: allow\n    condition: {a: 1}\n",
    )
    with pytest.raises(PolicyPackError, match="condition"):
        load_pack(path)


def test_duplicate_rule_names_raise(tmp_path):
    path = write_pack(
        tmp_path,
        "name: bad\nrules:\n"
        "  - name: r1\n    effect: allow\n"
        "  - name: r1\n    effect: deny\n",
    )
    with pytest.raises(PolicyPackError, match="duplicate rule name"):
        load_pack(path)
