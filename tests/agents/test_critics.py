"""Critic tests: fabrication, missing CIs, execute routing, confidence consistency."""

from __future__ import annotations

from fintwinos.agents.base import AgentOutput
from fintwinos.agents.critics import CriticAgent


def _clean_output() -> AgentOutput:
    """A well-evidenced output that should survive every check."""
    return AgentOutput(
        agent="treasury",
        step_id="step_1",
        summary="liquidity healthy",
        data={
            "risk_tier": "low",
            "requests_execution": False,
            "metrics": {"sim_lcr": 1.4},
            "candidate_actions": [
                {
                    "action": "monitor_liquidity_buffers",
                    "band": "propose",
                    "requires_approval": False,
                    "score": 0.8,
                }
            ],
            "simulation": {"metrics": {"lcr": 1.4}, "confidence": {"lcr": [1.25, 1.55]}},
        },
        tool_results=[{"tool": "observe_liquidity_ladder", "ok": True, "audit_ref": "abc"}],
        confidence=0.85,
        warnings=[],
    )


def _challenges_by_check(critique: AgentOutput) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for challenge in critique.data["challenges"]:
        grouped.setdefault(challenge["check"], []).append(challenge)
    return grouped


async def test_clean_output_passes_without_escalation(ctx):
    critique = await CriticAgent().review("benign objective", [_clean_output()], ctx)
    assert critique.data["challenges"] == []
    assert critique.data["requires_human"] is False
    assert critique.data["checked_outputs"] == 1


async def test_fabricated_output_with_no_tool_results_is_caught(ctx):
    fabricated = AgentOutput(
        agent="risk",
        step_id="step_1",
        summary="huge confident claims with zero evidence",
        data={
            "risk_tier": "low",
            "metrics": {"var_99": 12.5},
            "candidate_actions": [{"action": "hedge_everything", "band": "propose"}],
        },
        tool_results=[],
        confidence=0.9,
    )
    critique = await CriticAgent().review("objective", [fabricated], ctx)
    grouped = _challenges_by_check(critique)
    assert any(c["severity"] == "high" for c in grouped["evidence"])
    assert "fabrication" in grouped["evidence"][0]["detail"]
    assert critique.data["requires_human"] is True


async def test_missing_confidence_intervals_flagged(ctx):
    output = _clean_output()
    output.data["simulation"] = {"metrics": {"lcr": 1.4}, "confidence": {}}
    critique = await CriticAgent().review("objective", [output], ctx)
    grouped = _challenges_by_check(critique)
    assert any(c["severity"] == "high" for c in grouped["simulation"])
    assert critique.data["requires_human"] is True


async def test_candidates_without_any_rehearsal_flagged(ctx):
    output = _clean_output()
    output.data["simulation"] = None
    critique = await CriticAgent().review("objective", [output], ctx)
    grouped = _challenges_by_check(critique)
    assert any("without any" in c["detail"] for c in grouped["simulation"])
    assert critique.data["requires_human"] is True


async def test_execute_band_without_approval_routing_flagged(ctx):
    output = _clean_output()
    output.data["candidate_actions"] = [
        {
            "action": "send_wire",
            "band": "execute",
            "tool": "execute_payment",
            "requires_approval": False,
        }
    ]
    critique = await CriticAgent().review("objective", [output], ctx)
    grouped = _challenges_by_check(critique)
    assert any(c["severity"] == "high" for c in grouped["execution_routing"])
    assert critique.data["requires_human"] is True


async def test_execution_request_without_approval_route_flagged(ctx):
    output = _clean_output()
    output.data["requests_execution"] = True
    output.data["candidate_actions"][0]["requires_approval"] = False
    critique = await CriticAgent().review("objective", [output], ctx)
    grouped = _challenges_by_check(critique)
    assert any(
        "requests execution" in c["detail"] for c in grouped["execution_routing"]
    )
    assert critique.data["requires_human"] is True


async def test_execution_request_with_approval_route_passes(ctx):
    output = _clean_output()
    output.data["requests_execution"] = True
    output.data["candidate_actions"][0]["requires_approval"] = True
    critique = await CriticAgent().review("objective", [output], ctx)
    assert "execution_routing" not in _challenges_by_check(critique)


async def test_overconfidence_with_failed_tools_is_medium(ctx):
    output = AgentOutput(
        agent="compliance",
        step_id="step_1",
        summary="confident despite failures",
        data={"risk_tier": "low", "metrics": {"open_alerts": 1.0}},
        tool_results=[
            {"tool": "observe_alerts", "ok": True, "audit_ref": "h1"},
            {"tool": "observe_case_graph", "ok": False, "error": "boom"},
        ],
        confidence=0.9,
    )
    critique = await CriticAgent().review("objective", [output], ctx)
    grouped = _challenges_by_check(critique)
    assert grouped["confidence"][0]["severity"] == "medium"
    # a medium challenge alone does not escalate
    assert critique.data["requires_human"] is False


async def test_confidence_out_of_range_is_high(ctx):
    output = _clean_output()
    output.confidence = 1.7
    critique = await CriticAgent().review("objective", [output], ctx)
    grouped = _challenges_by_check(critique)
    assert grouped["confidence"][0]["severity"] == "high"
    assert critique.data["requires_human"] is True


async def test_honest_empty_output_is_not_escalated(ctx):
    """An agent that found nothing and says so is challenged gently, not escalated."""
    output = AgentOutput(
        agent="customer_ops",
        step_id="step_1",
        summary="no tools available; nothing to report",
        data={"risk_tier": "medium", "metrics": {}, "candidate_actions": []},
        tool_results=[],
        confidence=0.2,
        warnings=["no observe tools matched"],
    )
    critique = await CriticAgent().review("objective", [output], ctx)
    assert all(c["severity"] != "high" for c in critique.data["challenges"])
    assert critique.data["requires_human"] is False


async def test_critique_posted_to_blackboard_and_audited(ctx):
    before = len(ctx.audit)
    await CriticAgent().review("objective", [_clean_output()], ctx)
    assert ctx.blackboard.latest("critique") is not None
    assert len(ctx.audit.records(action="critic.review_completed")) == 1
    assert len(ctx.audit) > before


async def test_run_reads_outputs_from_blackboard(ctx):
    from fintwinos.agents.base import TaskSpec

    ctx.blackboard.post("outputs", _clean_output().model_dump(), actor="treasury")
    critique = await CriticAgent().run(
        TaskSpec(step_id="critic", owner="critic", objective="review"), ctx
    )
    assert critique.data["checked_outputs"] == 1
