"""Tests for report rendering: Markdown, JSON, fingerprint, credit footer."""

from __future__ import annotations

import json

from fintwinos.core.config import Settings
from fintwinos.evals.harness import ReleaseGate, SuiteSummary, release_gates
from fintwinos.evals.reports import (
    config_fingerprint,
    render_json,
    render_markdown,
    write_reports,
)


def _fixture_summaries() -> dict[str, SuiteSummary]:
    return {
        "function_calls": SuiteSummary(
            suite="function_calls", status="ok", n_cases=26, n_passed=25,
            pass_rate=0.961538, mean_score=0.97, mean_latency_ms=1.2,
            total_latency_ms=31.2,
            metrics={"hallucinated_tool_rate": 0.0, "tool_match_rate": 1.0},
        ),
        "domain_risk": SuiteSummary(
            suite="domain_risk", status="skipped", error="twin_sim not built"
        ),
    }


def _fixture_gates():
    gates = [
        ReleaseGate(name="function-call-exactness", description="d",
                    suite="function_calls", metric="pass_rate", op="ge",
                    threshold=0.95, blocks="execute tools"),
        ReleaseGate(name="risk-scenario-coverage", description="d",
                    suite="domain_risk", metric="scenario_coverage", op="ge",
                    threshold=0.8, blocks="risk promotion"),
    ]
    return release_gates(_fixture_summaries(), gates)


def test_render_markdown_contains_tables_gates_and_credit():
    md = render_markdown(_fixture_summaries(), _fixture_gates(), "abcd1234deadbeef",
                         notes=["twin runtime unavailable"])
    assert "# FinTwinOS evaluation report" in md
    assert "`function_calls`" in md and "`domain_risk`" in md
    assert "**PASS**" in md and "**FAIL**" in md
    assert "abcd1234deadbeef" in md
    assert "1/2 release gates passing." in md
    assert "twin runtime unavailable" in md
    # attribution footer is non-negotiable
    assert "Yash Sharma" in md and "linkedin.com/in/yashsharmaa" in md


def test_render_json_is_serialisable_and_complete():
    payload = render_json(_fixture_summaries(), _fixture_gates(), "abcd1234deadbeef")
    encoded = json.dumps(payload)  # must not raise
    decoded = json.loads(encoded)
    assert decoded["config_fingerprint"] == "abcd1234deadbeef"
    assert decoded["gates_total"] == 2 and decoded["gates_passed"] == 1
    assert decoded["suites"]["function_calls"]["pass_rate"] == 0.961538
    assert decoded["suites"]["domain_risk"]["status"] == "skipped"
    assert "Yash Sharma" in decoded["credit"]


def test_write_reports_creates_both_files(tmp_path):
    prefix = tmp_path / "nested" / "eval-report.v2"
    paths = write_reports(prefix, _fixture_summaries(), _fixture_gates(), "ff00ff00ff00ff00")
    assert paths["markdown"].endswith("eval-report.v2.md")
    assert paths["json"].endswith("eval-report.v2.json")
    md = (tmp_path / "nested" / "eval-report.v2.md").read_text(encoding="utf-8")
    assert "FinTwinOS evaluation report" in md
    payload = json.loads((tmp_path / "nested" / "eval-report.v2.json").read_text("utf-8"))
    assert payload["report"] == "fintwinos-eval"


def test_config_fingerprint_is_stable_and_sensitive():
    base = Settings(offline=True, openai_api_key=None, seed=7)
    same = Settings(offline=True, openai_api_key=None, seed=7)
    different_seed = Settings(offline=True, openai_api_key=None, seed=8)
    online = Settings(offline=False, openai_api_key=None, seed=7)
    assert config_fingerprint(base) == config_fingerprint(same)
    assert config_fingerprint(base) != config_fingerprint(different_seed)
    assert config_fingerprint(base) != config_fingerprint(online)
    assert len(config_fingerprint(base)) == 16
