"""End-to-end tests for the contracted ``run_suites`` entry point (fully offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fintwinos.evals.runner import SUITE_NAMES, run_suites


def test_run_suites_all_offline_end_to_end(tmp_path: Path):
    prefix = tmp_path / "reports" / "eval-report"
    outcome = run_suites("all", report_prefix=prefix, offline=True)

    # every declared suite is present in the summary, ok or explicitly skipped
    assert set(outcome["suites"]) == set(SUITE_NAMES)
    for name, summary in outcome["suites"].items():
        assert summary["status"] in {"ok", "skipped"}, (name, summary)

    # the self-contained suites must actually have run offline
    fc = outcome["suites"]["function_calls"]
    assert fc["status"] == "ok" and fc["n_cases"] >= 24
    assert fc["mean_score"] >= 0.90
    assert fc["metrics"]["hallucinated_tool_rate"] == 0.0
    assert outcome["suites"]["secque_style"]["status"] == "ok"
    assert outcome["suites"]["finmcp_style"]["status"] == "ok"
    assert outcome["suites"]["bfcl_style"]["status"] == "ok"

    # gates were evaluated and the run is honest about offline mode
    assert outcome["gates_total"] >= 8
    assert outcome["offline"] is True
    assert outcome["llm_usage"]["offline"] is True
    assert outcome["config_fingerprint"]

    # both report files exist and carry the same fingerprint
    md_path = Path(outcome["reports"]["markdown"])
    json_path = Path(outcome["reports"]["json"])
    assert md_path.exists() and json_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert outcome["config_fingerprint"] in md
    assert "Yash Sharma" in md
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["config_fingerprint"] == outcome["config_fingerprint"]
    assert set(payload["suites"]) == set(SUITE_NAMES)

    # the whole return value is JSON-serialisable (orchestrators depend on this)
    json.dumps(outcome)


def test_run_suites_is_deterministic_offline(tmp_path: Path):
    first = run_suites("function_calls", report_prefix=tmp_path / "a", offline=True)
    second = run_suites("function_calls", report_prefix=tmp_path / "b", offline=True)

    def stable(outcome: dict) -> dict:
        summary = dict(outcome["suites"]["function_calls"])
        summary.pop("mean_latency_ms", None)
        summary.pop("total_latency_ms", None)
        return summary

    assert stable(first) == stable(second)
    assert first["config_fingerprint"] == second["config_fingerprint"]


def test_run_suites_single_and_comma_separated_selection(tmp_path: Path):
    single = run_suites("secque_style", report_prefix=tmp_path / "single", offline=True)
    assert set(single["suites"]) == {"secque_style"}

    pair = run_suites(
        "function_calls, finmcp_style", report_prefix=tmp_path / "pair", offline=True
    )
    assert set(pair["suites"]) == {"function_calls", "finmcp_style"}
    # gates over suites that were not part of this run fail conservatively
    risk_gates = [g for g in pair["gates"] if g["suite"] == "domain_risk"]
    assert risk_gates and all(not g["passed"] for g in risk_gates)


def test_run_suites_rejects_unknown_suite(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown suite"):
        run_suites("not_a_suite", report_prefix=tmp_path / "x", offline=True)


async def test_run_suites_callable_from_inside_an_event_loop(tmp_path: Path):
    # e.g. a notebook or an async server calling the sync entry point
    outcome = run_suites("secque_style", report_prefix=tmp_path / "loop", offline=True)
    assert outcome["suites"]["secque_style"]["status"] == "ok"
