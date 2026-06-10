"""Tests for the multi-run consistency suite (with injected fake subjects)."""

from __future__ import annotations

import pytest

from fintwinos.evals.consistency import ConsistencySuite
from fintwinos.evals.harness import Runner, SuiteUnavailable, summarise


async def _steady_subject(case_id: str, objective: str) -> dict:
    return {
        "status": "complete",
        "decision": {"action_type": "propose_only", "objective": objective},
        "policy": {"allowed": True},
    }


def _flaky_subject():
    """Cycles status on every call: a maximally inconsistent runtime."""
    counter = {"n": 0}

    async def subject(case_id: str, objective: str) -> dict:
        counter["n"] += 1
        status = ["complete", "awaiting_human", "blocked"][counter["n"] % 3]
        return {"status": status, "decision": {"action_type": "propose_only"}}

    return subject


async def test_perfect_agreement_passes_with_score_one():
    suite = ConsistencySuite(subject=_steady_subject, n_runs=3)
    results = await suite.run()
    assert len(results) == len(suite.cases) == 3
    assert all(r.passed and r.score == 1.0 for r in results)
    summary = summarise(suite, results)
    assert summary.metrics["status_agreement_mean"] == 1.0
    assert summary.metrics["action_agreement_mean"] == 1.0


async def test_flaky_status_fails_but_keeps_action_credit():
    suite = ConsistencySuite(subject=_flaky_subject(), n_runs=3)
    results = await suite.run()
    for result in results:
        assert not result.passed
        # action_type always agrees (0.5), status never does fully (< 0.5)
        assert result.details["action_agreement"] == 1.0
        assert result.details["status_agreement"] < 1.0
        assert 0.5 < result.score < 1.0


async def test_decision_object_with_attribute_access_is_supported():
    class _Decision:
        action_type = "execute"

    async def subject(case_id: str, objective: str) -> dict:
        return {"status": "awaiting_human", "decision": _Decision()}

    suite = ConsistencySuite(subject=subject, n_runs=2)
    results = await suite.run()
    assert all(r.passed for r in results)
    assert results[0].details["modal_action_type"] == "execute"


async def test_missing_runtime_marks_suite_unavailable():
    suite = ConsistencySuite(runtime=None, registry=None, llm=None)
    with pytest.raises(SuiteUnavailable):
        await suite.run()
    # and the harness runner reports it as skipped rather than crashing
    summaries = await Runner([ConsistencySuite()]).run_all()
    assert summaries["consistency"].status == "skipped"


def test_n_runs_must_allow_agreement():
    with pytest.raises(ValueError):
        ConsistencySuite(subject=_steady_subject, n_runs=1)
