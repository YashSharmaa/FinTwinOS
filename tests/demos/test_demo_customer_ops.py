"""End-to-end offline tests for the customer ops (complaints spike) demo."""

from __future__ import annotations

import io
from typing import Any

from rich.console import Console

from fintwinos.demos import customer_ops


def _run(seed: int = 7) -> dict[str, Any]:
    return customer_ops.run(seed=seed, console=Console(file=io.StringIO(), width=120))


def test_run_offline_end_to_end_returns_contracted_keys() -> None:
    result = _run()
    assert result["demo"] == "customer_ops"
    assert result["offline"] is True
    for key in ("queue", "sla_table", "decision", "policy", "audit", "warnings"):
        assert key in result


def test_baseline_and_counterfactual_metrics() -> None:
    queue = _run()["queue"]
    for scenario_key in ("baseline", "plus_two_staff"):
        scenario = queue[scenario_key]
        assert isinstance(scenario["avg_wait"], float)
        assert isinstance(scenario["sla_breach_rate"], float)
        assert 0.0 <= scenario["sla_breach_rate"] <= 1.0
        ci = scenario["ci"]["sla_breach_rate"]
        assert ci is None or (len(ci) == 2 and ci[0] <= ci[1])
        assert scenario["metrics"]
    assert queue["plus_two_staff"]["staff"] == queue["baseline"]["staff"] + 2


def test_counterfactual_improves_under_common_random_numbers() -> None:
    queue = _run()["queue"]
    baseline, plus_two = queue["baseline"], queue["plus_two_staff"]
    assert plus_two["sla_breach_rate"] <= baseline["sla_breach_rate"]
    assert plus_two["avg_wait"] <= baseline["avg_wait"]


def test_sla_breach_table_rows() -> None:
    rows = _run()["sla_table"]
    metrics = {row["metric"] for row in rows}
    assert any("sla" in m or "breach" in m for m in metrics)
    assert len(rows) >= 2
    for row in rows:
        assert isinstance(row["baseline"], float)
        assert isinstance(row["plus_two"], float)
        assert row["delta"] is not None


def test_routing_change_is_propose_only_and_policy_allowed() -> None:
    result = _run()
    decision = result["decision"]
    assert decision["action_type"] == "propose_only"
    assert decision["planned_tool_calls"]
    policy = result["policy"]
    assert policy["allowed"] is True
    audit = result["audit"]
    assert audit["verified"] is True
    actions = {row["action"] for row in audit["excerpt"]}
    assert "decision.proposed" in actions


def test_determinism_same_seed_same_queue_numbers() -> None:
    first = _run(seed=5)
    second = _run(seed=5)
    assert first["queue"]["baseline"]["metrics"] == second["queue"]["baseline"]["metrics"]
    assert first["sla_table"] == second["sla_table"]
