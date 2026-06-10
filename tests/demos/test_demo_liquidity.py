"""End-to-end offline tests for the liquidity squeeze demo."""

from __future__ import annotations

import io
from typing import Any

from rich.console import Console

from fintwinos.demos import liquidity


def _run(seed: int = 7) -> dict[str, Any]:
    return liquidity.run(seed=seed, console=Console(file=io.StringIO(), width=120))


def test_run_offline_end_to_end_returns_contracted_keys() -> None:
    result = _run()
    assert result["demo"] == "liquidity"
    assert result["offline"] is True
    for key in ("cash_ladder", "stress", "case", "audit", "warnings"):
        assert key in result


def test_cash_ladder_has_rows() -> None:
    result = _run()
    rows = result["cash_ladder"]["rows"]
    assert isinstance(rows, list) and len(rows) >= 7
    assert all("closing_cash" in row or "net" in row for row in rows[:3])


def test_both_stress_presets_with_survival_and_ci() -> None:
    result = _run()
    stress = result["stress"]
    assert set(stress) == {"moderate_outflow", "severe_squeeze"}
    for row in stress.values():
        assert isinstance(row["survival_days"], float)
        assert row["survival_days"] > 0
        ci = row["ci"]
        assert isinstance(ci, list) and len(ci) == 2 and ci[0] <= ci[1]
        assert "breach_probability" in row  # may be None when the simulator omits it
        if row["breach_probability"] is not None:
            assert 0.0 <= row["breach_probability"] <= 1.0
        assert row["metrics"], "stress rows must carry the simulator metrics"


def test_severe_preset_shows_more_stress_than_moderate() -> None:
    stress = _run()["stress"]
    moderate, severe = stress["moderate_outflow"], stress["severe_squeeze"]
    # Severity must be visible on at least one comparable axis: funding cost
    # (real treasury simulator) or breach probability (bundled fallback model).
    cost_pair = (moderate["peak_funding_cost_bps"], severe["peak_funding_cost_bps"])
    breach_pair = (moderate["breach_probability"], severe["breach_probability"])
    if None not in cost_pair:
        assert cost_pair[1] > cost_pair[0]
    else:
        assert None not in breach_pair
        assert breach_pair[1] >= breach_pair[0]
        assert severe["survival_days"] <= moderate["survival_days"]


def test_case_has_contracted_shape_and_audit_verifies() -> None:
    result = _run()
    case = result["case"]
    assert case["status"] in {"complete", "awaiting_human", "blocked"}
    assert case["decision"] is not None
    assert case["decision"]["action_type"] in {"propose_only", "execute"}
    assert case["policy"] is not None
    audit = result["audit"]
    assert audit["verified"] is True
    assert audit["records"] > 0
    assert len(audit["excerpt"]) > 0


def test_determinism_same_seed_same_stress_numbers() -> None:
    first = _run(seed=11)
    second = _run(seed=11)
    assert first["stress"] == second["stress"]
    assert first["cash_ladder"]["rows"] == second["cash_ladder"]["rows"]


def test_main_entry_point_runs(capsys) -> None:
    liquidity.main()  # contracted CLI hook; must not raise offline
    # rich writes to stdout; just confirm something rendered
    assert capsys.readouterr().out
