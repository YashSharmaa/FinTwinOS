"""Domain agent tests: observe -> simulate -> propose pipeline and tier logic."""

from __future__ import annotations

from fintwinos.agents.base import AgentContext, Blackboard, TaskSpec
from fintwinos.agents.compliance import ComplianceAgent
from fintwinos.agents.customer_ops import CustomerOpsAgent
from fintwinos.agents.risk import RiskAgent
from fintwinos.agents.treasury import TreasuryAgent

from .conftest import build_fake_registry, build_fake_runtime


def _task(owner: str, objective: str) -> TaskSpec:
    return TaskSpec(step_id="step_1", owner=owner, objective=objective)


def _ctx_with(registry, runtime, offline_llm, settings) -> AgentContext:
    return AgentContext(
        runtime=runtime, registry=registry, llm=offline_llm,
        blackboard=Blackboard(), settings=settings,
        audit=runtime.audit if runtime is not None else None,
    )


# ------------------------------------------------------------------ treasury --


async def test_treasury_runs_full_pipeline(ctx):
    output = await TreasuryAgent().run(
        _task("treasury", "Review the current liquidity and cash buffer position"), ctx
    )
    tools_called = [t["tool"] for t in output.tool_results]
    assert "observe_liquidity_ladder" in tools_called
    assert "observe_funding_profile" in tools_called
    assert "simulate_liquidity_stress" in tools_called
    assert "propose_funding_plan" in tools_called
    assert all(t["ok"] for t in output.tool_results)
    assert all(t["audit_ref"] for t in output.tool_results)

    assert output.data["risk_tier"] == "low"
    assert output.data["metrics"]["sim_lcr"] == 1.40
    assert output.data["simulation"]["confidence"]["lcr"] == [1.25, 1.55]
    assert output.data["candidate_actions"][0]["action"] == "monitor_liquidity_buffers"
    assert output.data["requests_execution"] is False
    assert output.data["proposal"] == {"plan_id": "fp_001", "accepted": True}
    assert output.confidence > 0.8
    assert output.warnings == []


async def test_treasury_high_tier_under_severe_stress(ctx):
    output = await TreasuryAgent().run(
        _task("treasury", "Assess liquidity under a severe deposit outflow"), ctx
    )
    assert output.data["risk_tier"] == "high"
    assert output.data["metrics"]["sim_lcr"] == 0.82
    assert output.data["candidate_actions"][0]["action"] == "raise_short_term_funding"
    assert output.data["candidate_actions"][0]["requires_approval"] is True


# ---------------------------------------------------------------------- risk --


async def test_risk_low_tier_on_baseline_shock(ctx):
    output = await RiskAgent().run(
        _task("risk", "Check the equity exposure under a mild market shock"), ctx
    )
    assert output.data["risk_tier"] == "low"
    assert output.data["metrics"]["gross_exposure"] == 1000.0
    assert output.data["metrics"]["sim_shock_loss"] == 20.0
    assert output.data["candidate_actions"][0]["action"] == "hold_position"


async def test_risk_high_tier_on_severe_shock(ctx):
    output = await RiskAgent().run(
        _task("risk", "Hedge the equity exposure against a severe market shock"), ctx
    )
    assert output.data["risk_tier"] == "high"
    assert output.data["metrics"]["sim_shock_loss"] == 150.0
    assert output.data["metrics"]["loss_to_gross_ratio"] == 0.15
    assert output.data["candidate_actions"][0]["action"] == "hedge_with_index_futures"
    assert output.data["candidate_actions"][0]["requires_approval"] is True


async def test_risk_execution_request_carries_approval_route(ctx):
    output = await RiskAgent().run(
        _task("risk", "Execute the approved hedge for the shock book now"), ctx
    )
    assert output.data["requests_execution"] is True
    top = output.data["candidate_actions"][0]
    assert top["requires_approval"] is True
    assert "approval" in top["approval_route"].lower()


# ---------------------------------------------------------------- compliance --


async def test_compliance_medium_tier_on_routine_alerts(ctx):
    output = await ComplianceAgent().run(
        _task("compliance", "Triage the open aml alerts"), ctx
    )
    assert output.data["risk_tier"] == "medium"
    assert output.data["metrics"]["open_alerts"] == 1.0
    assert output.data["candidate_actions"][0]["action"] == "triage_alerts"


async def test_compliance_high_tier_on_severe_ring(runtime, offline_llm, settings):
    registry = build_fake_registry(
        runtime, settings,
        alert_severity="high", alert_count=3,
        ring_members=("cus_001", "cus_002", "cus_003", "cus_004"),
    )
    ctx = _ctx_with(registry, runtime, offline_llm, settings)
    output = await ComplianceAgent().run(
        _task("compliance", "Investigate the structuring alerts and the mule ring"), ctx
    )
    assert output.data["risk_tier"] == "high"
    assert output.data["metrics"]["ring_size"] == 4.0
    assert output.data["metrics"]["high_severity_alerts"] == 3.0
    actions = {a["action"]: a for a in output.data["candidate_actions"]}
    assert "open_investigation_case" in actions
    # SAR drafting is never autonomous
    assert actions["prepare_sar_draft"]["requires_approval"] is True


# -------------------------------------------------------------- customer ops --


async def test_customer_ops_low_tier_when_healthy(ctx):
    output = await CustomerOpsAgent().run(
        _task("customer_ops", "Review the complaint queue and sla health"), ctx
    )
    assert output.data["risk_tier"] == "low"
    assert output.data["metrics"]["sla_breach_rate"] == 0.03
    assert output.data["candidate_actions"][0]["action"] == "maintain_current_staffing"


async def test_customer_ops_high_tier_on_sla_failure(runtime, offline_llm, settings):
    registry = build_fake_registry(
        runtime, settings, sla_breach_rate=0.30, queue_depth=240.0
    )
    ctx = _ctx_with(registry, runtime, offline_llm, settings)
    output = await CustomerOpsAgent().run(
        _task("customer_ops", "Recover the queue: complaints and sla are slipping"), ctx
    )
    assert output.data["risk_tier"] == "high"
    actions = [a["action"] for a in output.data["candidate_actions"]]
    assert actions[0] == "surge_staffing"
    assert "prioritise_vulnerable_customers" in actions


# ------------------------------------------------------------- degraded mode --


async def test_domain_agent_degrades_without_tools(offline_llm, settings):
    """No registered tools: warnings, low confidence, never a crash."""
    runtime = build_fake_runtime()
    registry = build_fake_registry(runtime, settings, with_tools=False)
    ctx = _ctx_with(registry, runtime, offline_llm, settings)
    output = await RiskAgent().run(_task("risk", "Assess the exposure"), ctx)
    assert output.tool_results == []
    assert output.data["simulation"] is None
    assert output.confidence < 0.5
    assert any("no observe tools" in w for w in output.warnings)
    assert any("no simulate tool" in w for w in output.warnings)
    # flying blind is conservatively scored as medium tier
    assert output.data["risk_tier"] == "medium"


async def test_domain_outputs_are_posted_to_blackboard(ctx):
    await TreasuryAgent().run(_task("treasury", "Check the cash position"), ctx)
    posted = ctx.blackboard.latest("domain.treasury")
    assert posted is not None
    assert posted["domain"] == "treasury"


async def test_domain_runs_are_deterministic(ctx, runtime, offline_llm, settings):
    """Two identical runs produce identical metrics, ranking and confidence."""
    first = await TreasuryAgent().run(
        _task("treasury", "Review the liquidity buffers"), ctx
    )
    registry = build_fake_registry(runtime, settings)
    ctx2 = _ctx_with(registry, runtime, offline_llm, settings)
    second = await TreasuryAgent().run(
        _task("treasury", "Review the liquidity buffers"), ctx2
    )
    assert first.data["metrics"] == second.data["metrics"]
    assert [a["action"] for a in first.data["candidate_actions"]] == [
        a["action"] for a in second.data["candidate_actions"]
    ]
    assert first.confidence == second.confidence
    assert first.data["risk_tier"] == second.data["risk_tier"]
