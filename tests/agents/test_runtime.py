"""Orchestrator tests: handle_case end-to-end semantics, statuses and audit."""

from __future__ import annotations

from fintwinos.agents.base import TaskSpec
from fintwinos.agents.runtime import handle_case, plan_waves
from fintwinos.policy.gates import PolicyGate, Verdict

from .conftest import build_fake_registry, build_fake_runtime, make_settings

# ---------------------------------------------------------------- wave logic --


def test_plan_waves_honours_dependencies():
    plan = [
        TaskSpec(step_id="a", owner="treasury", objective="x"),
        TaskSpec(step_id="b", owner="compliance", objective="y", depends_on=["a"]),
        TaskSpec(step_id="c", owner="risk", objective="z", depends_on=["a"]),
        TaskSpec(step_id="d", owner="customer_ops", objective="w", depends_on=["b", "c"]),
    ]
    waves = plan_waves(plan)
    assert [[t.step_id for t in wave] for wave in waves] == [["a"], ["b", "c"], ["d"]]


def test_plan_waves_breaks_cycles_instead_of_deadlocking():
    plan = [
        TaskSpec(step_id="a", owner="treasury", objective="x", depends_on=["b"]),
        TaskSpec(step_id="b", owner="compliance", objective="y", depends_on=["a"]),
    ]
    waves = plan_waves(plan)
    assert sum(len(w) for w in waves) == 2  # every step still runs exactly once


# ----------------------------------------------------------------- statuses --


async def test_benign_research_objective_completes(runtime, registry, offline_llm):
    result = await handle_case(
        "case-001",
        "Research the current liquidity and cash buffer position",
        runtime,
        registry,
        llm=offline_llm,
    )
    assert result["status"] == "complete"
    assert result["decision"]["action_type"] == "propose_only"
    assert result["decision"]["risk_tier"] == "low"
    assert result["decision"]["ticket_id"] == "case-001"
    assert result["policy"]["allowed"] is True
    assert result["policy"]["requires_human_review"] is False
    assert result["critique"]["data"]["requires_human"] is False
    assert len(result["outputs"]) == 1
    assert result["outputs"][0]["agent"] == "treasury"


async def test_high_risk_objective_awaits_human(runtime, registry, offline_llm):
    result = await handle_case(
        "case-002",
        "Hedge the equity exposure against a severe market shock",
        runtime,
        registry,
        llm=offline_llm,
    )
    assert result["status"] == "awaiting_human"
    assert result["decision"]["action_type"] == "propose_only"
    assert result["decision"]["risk_tier"] == "high"
    assert result["policy"]["requires_human_review"] is True


async def test_execute_objective_awaits_human(runtime, registry, offline_llm):
    result = await handle_case(
        "case-003",
        "Execute the approved hedge for the shock book now",
        runtime,
        registry,
        llm=offline_llm,
    )
    assert result["status"] == "awaiting_human"
    assert result["decision"]["action_type"] == "execute"
    assert result["policy"]["requires_human_review"] is True
    assert "decision-execute-review" in result["policy"]["matched_rules"]


async def test_denying_gate_blocks_the_case(runtime, offline_llm):
    class FreezeGate(PolicyGate):
        def check_decision(self, decision):
            return Verdict(
                allowed=False,
                requires_human_review=True,
                reasons=["change freeze in effect"],
                matched_rules=["change-freeze"],
            )

    registry = build_fake_registry(runtime, make_settings(), policy_gate=FreezeGate())
    result = await handle_case(
        "case-004",
        "Research the current liquidity and cash buffer position",
        runtime,
        registry,
        llm=offline_llm,
    )
    assert result["status"] == "blocked"
    assert result["policy"]["allowed"] is False


# --------------------------------------------------------------- multi-domain --


async def test_multi_domain_case_runs_all_owners(runtime, registry, offline_llm):
    result = await handle_case(
        "case-005",
        "Liquidity buffers are strained while aml alerts spike across the payments ring",
        runtime,
        registry,
        llm=offline_llm,
    )
    assert [step["owner"] for step in result["plan"]] == ["treasury", "compliance"]
    assert result["plan"][1]["depends_on"] == ["step_1"]
    assert {output["agent"] for output in result["outputs"]} == {"treasury", "compliance"}
    assert result["critique"]["data"]["checked_outputs"] == 2


# --------------------------------------------------------- audit and contract --


async def test_audit_grows_monotonically_through_the_stages(runtime, registry, offline_llm):
    before = len(runtime.audit)
    result = await handle_case(
        "case-006",
        "Research the current liquidity and cash buffer position",
        runtime,
        registry,
        llm=offline_llm,
    )
    assert result["audit_len"] == len(runtime.audit)
    assert result["audit_len"] > before
    assert runtime.audit.verify()

    records = runtime.audit.records()
    seqs = [r.seq for r in records]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    stage_order = [
        "case.opened",
        "case.planned",
        "case.sensing_completed",
        "case.step_completed",
        "case.critique_completed",
        "case.decision_aggregated",
        "case.policy_checked",
        "case.closed",
    ]
    positions = []
    for action in stage_order:
        matching = [r.seq for r in records if r.action == action]
        assert matching, f"missing audit stage '{action}'"
        positions.append(matching[0])
    assert positions == sorted(positions)


async def test_result_contract_shape(runtime, registry, offline_llm):
    result = await handle_case(
        "case-007",
        "Triage the open aml alerts",
        runtime,
        registry,
        llm=offline_llm,
    )
    for key in (
        "status", "decision", "policy", "outputs", "critique",
        "sensing", "plan", "audit_len", "case_id",
    ):
        assert key in result, f"missing contract key '{key}'"
    assert result["status"] in {"complete", "awaiting_human", "blocked"}
    assert isinstance(result["outputs"], list)
    assert len(result["outputs"]) == len(result["plan"])
    assert isinstance(result["critique"]["data"]["challenges"], list)
    # sensing surfaced the deliberately stale series in the case record
    assert result["sensing"]["data"]["stale_series"] == ["ops.queue_depth"]


async def test_handle_case_is_offline_deterministic(offline_llm):
    """Same twin, same objective, same seed: same decision and status."""
    results = []
    for _ in range(2):
        runtime = build_fake_runtime()
        registry = build_fake_registry(runtime, make_settings())
        results.append(
            await handle_case(
                "case-008",
                "Hedge the equity exposure against a severe market shock",
                runtime,
                registry,
                llm=offline_llm,
            )
        )
    first, second = results
    assert first["status"] == second["status"] == "awaiting_human"
    assert first["decision"]["risk_tier"] == second["decision"]["risk_tier"]
    assert first["decision"]["planned_tool_calls"] == second["decision"]["planned_tool_calls"]
    assert (
        first["outputs"][0]["data"]["metrics"] == second["outputs"][0]["data"]["metrics"]
    )
