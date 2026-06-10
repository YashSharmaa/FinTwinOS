"""Unit tests for the shared demo plumbing in ``fintwinos.demos.stack``."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from fintwinos.core.types import SideEffectClass, SimulationResult, ToolBand
from fintwinos.demos.stack import (
    as_plain,
    build_stack,
    demo_settings,
    ensure_allow_rule,
    extract_simulation,
    find_tool,
    metric_named,
    normalize_case,
    schema_args,
)
from fintwinos.policy.gates import PolicyGate, RuleEffect
from fintwinos.tools.registry import ToolSpec


def _quiet() -> Console:
    return Console(file=io.StringIO(), width=120)


def test_demo_settings_forces_offline_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINTWIN_OFFLINE", raising=False)
    settings = demo_settings(offline_ok=True)
    assert settings.offline is True
    assert settings.llm_available is False


def test_demo_settings_raises_when_offline_not_ok() -> None:
    with pytest.raises(RuntimeError, match="offline"):
        demo_settings(offline_ok=False)


def test_build_stack_assembles_runtime_registry_and_llm() -> None:
    stack = build_stack(seed=7, console=_quiet())
    assert len(stack.registry) > 0
    assert stack.registry.policy_gate is not None
    assert stack.registry.settings is stack.settings
    assert stack.llm.offline is True
    assert stack.runtime.audit.verify()


def test_find_tool_exact_and_glob() -> None:
    stack = build_stack(seed=7, console=_quiet())
    name = find_tool(stack.registry, ["simulate_liquidity_stress"])
    assert name == "simulate_liquidity_stress"
    glob_name = find_tool(stack.registry, ["observe_*ladder*"])
    assert glob_name is not None and glob_name.startswith("observe_")
    assert find_tool(stack.registry, ["observe_nonexistent", "simulate_nope_*"]) is None


def test_schema_args_filters_and_fills_required() -> None:
    spec = ToolSpec(
        name="observe_thing",
        description="t",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
                "flag": {"type": "boolean"},
            },
            "required": ["query", "flag"],
            "additionalProperties": False,
        },
        band=ToolBand.observe,
        side_effect=SideEffectClass.read,
    )
    args = schema_args(spec, {"k": 3, "unknown_key": "dropped", "noneval": None})
    assert args["k"] == 3
    assert "unknown_key" not in args and "noneval" not in args
    assert args["query"] == "risk factors"  # named default
    assert args["flag"] is False  # type default


def test_schema_args_loose_schema_passes_preferred() -> None:
    spec = ToolSpec(
        name="observe_loose",
        description="t",
        input_schema={"type": "object"},
        band=ToolBand.observe,
    )
    assert schema_args(spec, {"a": 1, "b": None}) == {"a": 1}


def test_extract_simulation_unwraps_nested_payloads() -> None:
    sim = SimulationResult(
        simulator="s", scenario_name="x", metrics={"survival_days": 12.0},
        confidence={"survival_days": [8.0, 20.0]},
    )
    direct = extract_simulation(sim.model_dump())
    nested = extract_simulation({"result": sim.model_dump(), "extra": 1})
    model = extract_simulation(sim)
    for shaped in (direct, nested, model):
        assert shaped["metrics"]["survival_days"] == 12.0
        assert metric_named(shaped, "survival") == 12.0
    assert extract_simulation({"no": "metrics"})["metrics"] == {}
    assert metric_named({"metrics": {}}, "survival") is None


def test_normalize_case_fills_contracted_keys() -> None:
    shaped = normalize_case({"status": "awaiting_human", "decision": {"objective": "x"}})
    assert shaped["status"] == "awaiting_human"
    assert shaped["policy"] is None
    weird = normalize_case("just a string")
    assert weird["status"] == "complete"
    assert weird["decision"] == "just a string"


def test_ensure_allow_rule_is_idempotent() -> None:
    gate = PolicyGate()
    before = len(gate.rules)
    ensure_allow_rule(gate, "execute_close_case", "demo-allow")
    ensure_allow_rule(gate, "execute_close_case", "demo-allow")
    added = [r for r in gate.rules if r.name == "demo-allow"]
    assert len(added) == 1
    assert added[0].effect == RuleEffect.allow
    assert len(gate.rules) == before + 1
    ensure_allow_rule(None, "execute_close_case", "demo-allow")  # no-op, no crash


def test_as_plain_handles_models_and_containers() -> None:
    sim = SimulationResult(simulator="s", scenario_name="x")
    plain = as_plain({"a": [sim], "b": (1, 2)})
    assert plain["a"][0]["simulator"] == "s"
    assert plain["b"] == [1, 2]


def test_run_demo_helper_validates_names_and_runs() -> None:
    import importlib

    from fintwinos.demos import DEMO_NAMES, run_demo

    with pytest.raises(ValueError, match="unknown demo"):
        run_demo("not_a_demo")
    for name in DEMO_NAMES:  # every advertised demo module exposes the contract
        module = importlib.import_module(f"fintwinos.demos.{name}")
        assert callable(module.main) and callable(module.run)
    result = run_demo("customer_ops", console=_quiet(), seed=3)
    assert result["demo"] == "customer_ops"
