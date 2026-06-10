"""The case orchestrator: the contracted ``handle_case`` entry point.

Implements the founding-brief semantics end to end:

``plan -> sensing -> parallel domain agents (dependency waves) -> critic review ->
planner.aggregate -> policy gate -> status``

Every stage appends to the shared audit trail, all shared state flows through the
``Blackboard``, and the final status is decided purely by the policy verdict:

- ``complete`` — only for an *allowed*, ``propose_only`` decision that needs no human
  review;
- ``awaiting_human`` — whenever the verdict requires human review (which includes
  every ``execute`` decision and every high/critical-tier proposal);
- ``blocked`` — anything else (a denying gate, or an execute decision that somehow
  escaped the review flag — belt and braces, it never auto-runs).

The whole pipeline is offline-safe: with ``FINTWIN_OFFLINE=1`` every agent uses its
deterministic rule-based path and the orchestration is fully reproducible.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fintwinos.agents.base import AgentContext, AgentOutput, BaseAgent, Blackboard, TaskSpec
from fintwinos.agents.compliance import ComplianceAgent
from fintwinos.agents.critics import CriticAgent
from fintwinos.agents.customer_ops import CustomerOpsAgent
from fintwinos.agents.planner import PlannerAgent
from fintwinos.agents.risk import RiskAgent
from fintwinos.agents.sensing import SensingAgent
from fintwinos.agents.treasury import TreasuryAgent
from fintwinos.core.interfaces import TwinRuntime
from fintwinos.models.llm_routing.client import LLMClient
from fintwinos.policy.gates import PolicyGate
from fintwinos.tools.registry import ToolRegistry

__all__ = ["handle_case", "build_domain_agents", "plan_waves"]


def build_domain_agents() -> dict[str, BaseAgent]:
    """The default owner -> agent mapping for the four domain desks."""
    return {
        "risk": RiskAgent(),
        "treasury": TreasuryAgent(),
        "compliance": ComplianceAgent(),
        "customer_ops": CustomerOpsAgent(),
    }


def plan_waves(plan: list[TaskSpec]) -> list[list[TaskSpec]]:
    """Group plan steps into dependency waves for ``asyncio.gather`` scheduling.

    A step joins a wave once everything it depends on has been scheduled in an
    earlier wave. Dangling or cyclic dependencies cannot deadlock the case: if no
    step is ready, the remaining steps are scheduled together in a final wave (the
    breakage is the planner's bug, not a reason to strand a live case).
    """
    order = {task.step_id: index for index, task in enumerate(plan)}
    remaining = {task.step_id: task for task in plan}
    completed: set[str] = set()
    waves: list[list[TaskSpec]] = []
    while remaining:
        ready = [
            task
            for task in remaining.values()
            if all(dep in completed for dep in task.depends_on if dep in order)
        ]
        if not ready:  # cyclic plan: break it deterministically
            ready = list(remaining.values())
        ready.sort(key=lambda task: order[task.step_id])
        waves.append(ready)
        for task in ready:
            completed.add(task.step_id)
            del remaining[task.step_id]
    return waves


async def _run_step(
    agents: dict[str, BaseAgent], task: TaskSpec, ctx: AgentContext
) -> AgentOutput:
    """Run one plan step with its owning agent; unknown owners degrade safely."""
    agent = agents.get(task.owner)
    if agent is None:
        return AgentOutput(
            agent=task.owner,
            step_id=task.step_id,
            summary=f"no agent registered for owner '{task.owner}'; step skipped",
            confidence=0.0,
            warnings=[f"unknown owner '{task.owner}'"],
        )
    return await agent.run(task, ctx)


async def handle_case(
    case_id: str,
    objective: str,
    runtime: TwinRuntime | None,
    registry: ToolRegistry,
    llm: LLMClient | None = None,
    blackboard: Blackboard | None = None,
) -> dict[str, Any]:
    """Orchestrate one case from objective to policy-gated decision.

    Parameters
    ----------
    case_id:
        Caller-supplied case identifier; threaded into every audit record and set as
        the decision's ``ticket_id``.
    objective:
        The natural-language case objective the planner decomposes.
    runtime:
        The assembled federated twin. Its audit trail is the case's audit trail.
    registry:
        The typed tool registry every agent must go through for twin access.
    llm:
        Optional shared LLM client; a fresh client (which honours offline mode) is
        created when omitted.
    blackboard:
        Optional shared blackboard, e.g. to seed prior context; created when omitted.

    Returns
    -------
    dict with keys ``status`` (``complete`` | ``awaiting_human`` | ``blocked``),
    ``decision``, ``policy``, ``outputs``, ``critique``, ``sensing``, ``plan``,
    ``case_id`` and ``audit_len``.
    """
    settings = registry.settings
    llm = llm or LLMClient(settings=settings)
    blackboard = blackboard or Blackboard()
    audit = runtime.audit if runtime is not None else registry.audit
    ctx = AgentContext(
        runtime=runtime,
        registry=registry,
        llm=llm,
        blackboard=blackboard,
        settings=settings,
        audit=audit,
    )

    audit.append(
        "orchestrator", "case.opened", {"case_id": case_id, "objective": objective[:300]}
    )

    # 1. Plan -----------------------------------------------------------------
    planner = PlannerAgent()
    plan = await planner.decompose(objective, ctx)
    audit.append(
        "orchestrator",
        "case.planned",
        {
            "case_id": case_id,
            "steps": [
                {"step_id": t.step_id, "owner": t.owner, "depends_on": t.depends_on}
                for t in plan
            ],
        },
    )

    # 2. Sensing -----------------------------------------------------------------
    sensing_agent = SensingAgent()
    sensing_output = await sensing_agent.run(
        TaskSpec(
            step_id="sensing",
            owner="sensing",
            objective=f"Assess twin freshness for case {case_id}",
        ),
        ctx,
    )
    audit.append(
        "orchestrator",
        "case.sensing_completed",
        {
            "case_id": case_id,
            "series": len(sensing_output.data.get("series", {})),
            "stale_series": len(sensing_output.data.get("stale_series", [])),
            "warnings": len(sensing_output.warnings),
        },
    )

    # 3. Domain swarm in dependency waves ----------------------------------------
    agents = build_domain_agents()
    outputs: list[AgentOutput] = []
    for wave in plan_waves(plan):
        results = await asyncio.gather(
            *(_run_step(agents, task, ctx) for task in wave), return_exceptions=True
        )
        for task, result in zip(wave, results, strict=True):
            if isinstance(result, BaseException):
                result = AgentOutput(
                    agent=task.owner,
                    step_id=task.step_id,
                    summary=f"agent crashed: {type(result).__name__}: {result}",
                    confidence=0.0,
                    warnings=[f"agent exception: {type(result).__name__}"],
                )
            outputs.append(result)
            blackboard.post("outputs", result.model_dump(), actor=result.agent)
            audit.append(
                "orchestrator",
                "case.step_completed",
                {
                    "case_id": case_id,
                    "step_id": task.step_id,
                    "agent": result.agent,
                    "confidence": result.confidence,
                    "warnings": len(result.warnings),
                },
            )

    # 4. Critic review ---------------------------------------------------------------
    critic = CriticAgent()
    critique = await critic.review(objective, outputs, ctx)
    audit.append(
        "orchestrator",
        "case.critique_completed",
        {
            "case_id": case_id,
            "challenges": len(critique.data.get("challenges", [])),
            "high_count": critique.data.get("high_count", 0),
            "requires_human": critique.data.get("requires_human", False),
        },
    )

    # 5. Aggregate ---------------------------------------------------------------------
    decision = await planner.aggregate(objective, outputs, critique, ctx)
    decision.ticket_id = case_id
    audit.append(
        "orchestrator",
        "case.decision_aggregated",
        {
            "case_id": case_id,
            "decision_id": decision.decision_id,
            "action_type": decision.action_type,
            "risk_tier": decision.risk_tier.value,
        },
    )

    # 6. Policy gate ----------------------------------------------------------------------
    gate = registry.policy_gate
    if gate is None or not hasattr(gate, "check_decision"):
        gate = PolicyGate()
    verdict = gate.check_decision(decision)
    audit.append(
        "orchestrator",
        "case.policy_checked",
        {
            "case_id": case_id,
            "decision_id": decision.decision_id,
            "allowed": verdict.allowed,
            "requires_human_review": verdict.requires_human_review,
            "rules": verdict.matched_rules,
        },
    )

    # 7. Status -------------------------------------------------------------------------------
    if not verdict.allowed:
        status = "blocked"
    elif verdict.requires_human_review:
        status = "awaiting_human"
    elif decision.action_type == "propose_only":
        status = "complete"
    else:  # an unreviewed execute decision must never fall through to "complete"
        status = "blocked"

    audit.append(
        "orchestrator", "case.closed", {"case_id": case_id, "status": status}
    )
    return {
        "status": status,
        "case_id": case_id,
        "decision": decision.model_dump(mode="json"),
        "policy": verdict.model_dump(mode="json"),
        "outputs": [output.model_dump(mode="json") for output in outputs],
        "critique": critique.model_dump(mode="json"),
        "sensing": sensing_output.model_dump(mode="json"),
        "plan": [task.model_dump(mode="json") for task in plan],
        "audit_len": len(audit),
    }
