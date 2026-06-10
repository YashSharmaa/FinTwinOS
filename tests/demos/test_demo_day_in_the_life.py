"""End-to-end offline tests for the composed day-in-the-life demo."""

from __future__ import annotations

import io
from typing import Any

from rich.console import Console

from fintwinos.demos import day_in_the_life


def _run(seed: int = 7) -> dict[str, Any]:
    return day_in_the_life.run(seed=seed, console=Console(file=io.StringIO(), width=120))


def test_all_four_sections_run_on_one_stack() -> None:
    result = _run()
    assert result["demo"] == "day_in_the_life"
    assert set(result["sections"]) == {
        "liquidity",
        "aml_triage",
        "analyst_research",
        "customer_ops",
    }
    assert result["sections"]["liquidity"]["stress"]
    assert result["sections"]["aml_triage"]["close_case"]["approved"]["ok"] is True
    assert result["sections"]["analyst_research"]["critic"]["passed"] is True
    assert result["sections"]["customer_ops"]["decision"]["action_type"] == "propose_only"


def test_governance_audit_chain_verifies() -> None:
    governance = _run()["governance"]
    assert governance["audit_verified"] is True
    assert governance["audit_records"] > 20


def test_governance_tool_calls_by_band() -> None:
    bands = _run()["governance"]["tool_calls_by_band"]
    assert bands  # at least one band saw traffic
    assert all(isinstance(count, int) and count > 0 for count in bands.values())
    # the day includes simulation rehearsals and the approved case closure
    assert bands.get("simulate", 0) >= 1
    assert bands.get("execute", 0) >= 1


def test_governance_llm_usage_summary_offline() -> None:
    usage = _run()["governance"]["llm_usage"]
    assert usage["offline"] is True
    assert usage["estimated_cost_usd"] == 0.0
    assert "calls" in usage and "input_tokens" in usage and "output_tokens" in usage


def test_sections_share_one_audit_trail() -> None:
    result = _run()
    # each section's audit length is a snapshot of the same growing chain
    lengths = [result["sections"][k]["audit"]["records"] for k in
               ("liquidity", "aml_triage", "analyst_research", "customer_ops")]
    assert lengths == sorted(lengths)
    assert result["governance"]["audit_records"] >= lengths[-1]
