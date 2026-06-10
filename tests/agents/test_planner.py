"""Planner tests: keyword routing, dependencies, LLM refinement, aggregation."""

from __future__ import annotations

import json

import pytest

from fintwinos.agents.base import AgentContext, AgentOutput, Blackboard
from fintwinos.agents.planner import PlannerAgent
from fintwinos.core.types import RiskTier
from fintwinos.policy.gates import PolicyGate

from .conftest import ScriptedLLM


@pytest.fixture
def planner() -> PlannerAgent:
    return PlannerAgent()


# ---------------------------------------------------------------- routing --


@pytest.mark.parametrize(
    ("objective", "owner"),
    [
        ("Review overnight liquidity and the funding runway", "treasury"),
        ("Triage the new aml alerts and the suspected mule ring", "compliance"),
        ("Stress the equity var under a three sigma shock and propose a hedge", "risk"),
        ("Clear the complaint queue before the sla deadline", "customer_ops"),
    ],
)
async def test_routes_single_domain(planner, ctx, objective, owner):
    plan = await planner.decompose(objective, ctx)
    assert len(plan) == 1
    assert plan[0].owner == owner
    assert plan[0].depends_on == []
    assert plan[0].inputs["routed_by"] == "keyword"
    assert plan[0].inputs["keywords_matched"]


async def test_unmatched_objective_defaults_to_risk(planner, ctx):
    plan = await planner.decompose("Summarise the board minutes from yesterday", ctx)
    assert len(plan) == 1
    assert plan[0].owner == "risk"
    assert plan[0].inputs["routed_by"] == "default"


async def test_word_boundary_matching_avoids_false_positives(planner, ctx):
    # "various" must not trigger the risk keyword "var".
    plan = await planner.decompose("Summarise various board updates", ctx)
    assert plan[0].inputs["routed_by"] == "default"


async def test_multi_domain_plan_has_dependencies(planner, ctx):
    objective = (
        "Liquidity buffers are strained while aml alerts spike across the payments ring"
    )
    plan = await planner.decompose(objective, ctx)
    owners = [task.owner for task in plan]
    assert owners == ["treasury", "compliance"]  # lead = first domain mentioned
    assert plan[0].depends_on == []
    assert plan[1].depends_on == ["step_1"]
    assert plan[0].inputs["lead"] is True
    assert plan[1].inputs["lead"] is False


# ----------------------------------------------------------- LLM refinement --


def _online_ctx(runtime, registry, settings, llm) -> AgentContext:
    return AgentContext(
        runtime=runtime,
        registry=registry,
        llm=llm,  # type: ignore[arg-type] — structural stand-in for LLMClient
        blackboard=Blackboard(),
        settings=settings,
        audit=runtime.audit,
    )


async def test_llm_refinement_invalid_json_falls_back(planner, runtime, registry, settings):
    llm = ScriptedLLM("this is not json at all")
    ctx = _online_ctx(runtime, registry, settings, llm)
    plan = await planner.decompose(
        "Liquidity buffers are strained while aml alerts spike", ctx
    )
    assert llm.calls >= 1
    assert [t.owner for t in plan] == ["treasury", "compliance"]
    assert not any(t.inputs.get("refined_by_llm") for t in plan)


async def test_llm_refinement_wrong_owner_set_falls_back(planner, runtime, registry, settings):
    refined = {
        "steps": [
            {"step_id": "a", "owner": "risk", "objective": "wrong desk", "depends_on": []}
        ]
    }
    llm = ScriptedLLM(json.dumps(refined))
    ctx = _online_ctx(runtime, registry, settings, llm)
    plan = await planner.decompose(
        "Liquidity buffers are strained while aml alerts spike", ctx
    )
    assert {t.owner for t in plan} == {"treasury", "compliance"}
    assert not any(t.inputs.get("refined_by_llm") for t in plan)


async def test_llm_refinement_forward_dependency_falls_back(
    planner, runtime, registry, settings
):
    refined = {
        "steps": [
            {"step_id": "a", "owner": "treasury", "objective": "x", "depends_on": ["b"]},
            {"step_id": "b", "owner": "compliance", "objective": "y", "depends_on": []},
        ]
    }
    llm = ScriptedLLM(json.dumps(refined))
    ctx = _online_ctx(runtime, registry, settings, llm)
    plan = await planner.decompose(
        "Liquidity buffers are strained while aml alerts spike", ctx
    )
    assert not any(t.inputs.get("refined_by_llm") for t in plan)


async def test_llm_refinement_valid_plan_accepted(planner, runtime, registry, settings):
    refined = {
        "steps": [
            {
                "step_id": "c1",
                "owner": "compliance",
                "objective": "Sweep the alert ring first",
                "depends_on": [],
            },
            {
                "step_id": "t1",
                "owner": "treasury",
                "objective": "Then quantify the liquidity impact",
                "depends_on": ["c1"],
            },
        ]
    }
    llm = ScriptedLLM(json.dumps(refined))
    ctx = _online_ctx(runtime, registry, settings, llm)
    plan = await planner.decompose(
        "Liquidity buffers are strained while aml alerts spike", ctx
    )
    assert [t.owner for t in plan] == ["compliance", "treasury"]
    assert plan[1].depends_on == ["c1"]
    assert all(t.inputs.get("refined_by_llm") for t in plan)


# --------------------------------------------------------------- aggregation --


def _output(agent: str, tier: str, *, requests_execution: bool = False) -> AgentOutput:
    return AgentOutput(
        agent=agent,
        step_id="step_1",
        summary=f"{agent} summary",
        data={
            "risk_tier": tier,
            "requests_execution": requests_execution,
            "candidate_actions": [
                {
                    "action": f"{agent}_top_action",
                    "band": "propose",
                    "arguments": {"k": 1},
                    "score": 0.8,
                    "requires_approval": requests_execution,
                }
            ],
        },
        tool_results=[{"tool": "observe_x", "ok": True, "audit_ref": "h"}],
        confidence=0.8,
    )


async def test_aggregate_defaults_to_propose_only_and_max_tier(planner, ctx):
    outputs = [_output("treasury", "low"), _output("compliance", "medium")]
    decision = await planner.aggregate("benign objective", outputs, None, ctx)
    assert decision.action_type == "propose_only"
    assert decision.risk_tier == RiskTier.medium
    assert len(decision.planned_tool_calls) == 2
    assert decision.owner == "planner"


async def test_aggregate_missing_tier_counts_as_medium(planner, ctx):
    output = AgentOutput(agent="risk", summary="no tier given", confidence=0.5)
    decision = await planner.aggregate("objective", [output], None, ctx)
    assert decision.risk_tier == RiskTier.medium


async def test_aggregate_execute_request_flips_action_type(planner, ctx):
    outputs = [_output("risk", "low", requests_execution=True)]
    decision = await planner.aggregate("execute the hedge", outputs, None, ctx)
    assert decision.action_type == "execute"


async def test_execute_decisions_always_flagged_for_review(planner, ctx):
    """An execute decision can never pass the gate without human review."""
    outputs = [_output("risk", "low", requests_execution=True)]
    decision = await planner.aggregate("execute the hedge", outputs, None, ctx)
    verdict = PolicyGate().check_decision(decision)
    assert verdict.allowed is True
    assert verdict.requires_human_review is True
    assert "decision-execute-review" in verdict.matched_rules


async def test_aggregate_critic_escalation_ratchets_tier_high(planner, ctx):
    critique = AgentOutput(
        agent="critic",
        summary="escalating",
        data={"requires_human": True, "challenges": [{"severity": "high", "check": "evidence"}]},
        confidence=0.8,
    )
    outputs = [_output("treasury", "low")]
    decision = await planner.aggregate("objective", outputs, critique, ctx)
    assert decision.risk_tier == RiskTier.high
    assert PolicyGate().check_decision(decision).requires_human_review is True
