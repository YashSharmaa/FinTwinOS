"""The planner: decomposes case objectives into a domain plan and aggregates results.

The planner is the root of the hierarchical-and-debating orchestration pattern:

- ``decompose`` maps an objective onto the four domain desks with deterministic
  keyword/intent routing. Multi-domain objectives become multi-step plans where the
  lead desk (the first domain mentioned) runs in wave one and the remaining desks
  depend on it. When an LLM is available it may *refine* the rule-based plan via a
  strict JSON schema; any structural deviation rejects the refinement and the
  rule-based plan stands, so planning never degrades below the deterministic floor.
- ``aggregate`` folds the domain outputs and the critic's challenge list into a single
  ``Decision``. The decision is ``propose_only`` unless an output explicitly requested
  execution, its risk tier is the maximum of the outputs' implied tiers (ratcheted to
  ``high`` when the critic escalates), and it then flows to the policy gate.
"""

from __future__ import annotations

import re
from typing import Any

from fintwinos.agents.base import AgentContext, AgentOutput, BaseAgent, TaskSpec
from fintwinos.agents.domain import bump_tier
from fintwinos.core.types import RISK_ORDER, Decision, RiskTier
from fintwinos.models.llm_routing.router import TaskClass

#: Keyword -> domain routing table. Matching is word-boundary based so short keywords
#: such as "var" do not fire inside unrelated words ("various").
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "treasury": (
        "liquidity", "funding", "cash", "lcr", "nsfr", "buffer",
        "collateral", "deposit", "deposits", "treasury", "intraday",
    ),
    "compliance": (
        "alert", "alerts", "aml", "ring", "kyc", "sanction", "sanctions",
        "laundering", "fraud", "sar", "screening", "mule", "compliance",
    ),
    "risk": (
        "hedge", "exposure", "var", "shock", "stress", "volatility",
        "drawdown", "concentration", "cvar", "risk",
    ),
    "customer_ops": (
        "queue", "queues", "complaint", "complaints", "sla", "customer",
        "customers", "backlog", "churn", "onboarding",
    ),
}

#: Deterministic tie-break when two domains first match at the same position.
DOMAIN_PRIORITY: tuple[str, ...] = ("risk", "treasury", "compliance", "customer_ops")

#: Owner used when no domain keyword matches: the risk desk is the generalist
#: analytical owner for unclassified objectives.
DEFAULT_OWNER = "risk"

#: Strict schema the LLM must satisfy when refining the rule-based plan.
PLAN_REFINEMENT_SCHEMA: dict[str, Any] = {
    "title": "plan_refinement",
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string", "minLength": 1},
                    "owner": {"type": "string", "enum": sorted(DOMAIN_KEYWORDS)},
                    "objective": {"type": "string", "minLength": 1},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["step_id", "owner", "objective", "depends_on"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}

_MAX_REFINED_STEPS = 8


def _keyword_positions(objective: str) -> dict[str, dict[str, Any]]:
    """First match position and matched keywords per domain, word-boundary matched."""
    lowered = objective.lower()
    matches: dict[str, dict[str, Any]] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits: list[tuple[int, str]] = []
        for keyword in keywords:
            found = re.search(rf"\b{re.escape(keyword)}\b", lowered)
            if found:
                hits.append((found.start(), keyword))
        if hits:
            hits.sort()
            matches[domain] = {
                "position": hits[0][0],
                "keywords": [kw for _, kw in hits],
            }
    return matches


class PlannerAgent(BaseAgent):
    """Decomposes objectives into domain task plans and aggregates the results."""

    name = "planner"
    role = "planner"
    task_class = TaskClass.planning

    # ------------------------------------------------------------------ decompose --

    async def decompose(self, objective: str, ctx: AgentContext) -> list[TaskSpec]:
        """Build the rule-based plan, then optionally let the LLM refine it.

        The rule-based plan is the deterministic floor: domain owners are chosen by
        keyword routing, ordered by where each domain is first mentioned in the
        objective, with later steps depending on the lead step. An available LLM may
        re-order steps or rewrite step objectives, but only within a strictly
        validated structure over the *same* owner set; any violation falls back.
        """
        matches = _keyword_positions(objective)
        if matches:
            ordered = sorted(
                matches,
                key=lambda d: (matches[d]["position"], DOMAIN_PRIORITY.index(d)),
            )
        else:
            ordered = [DEFAULT_OWNER]

        plan: list[TaskSpec] = []
        for index, domain in enumerate(ordered):
            keywords = matches.get(domain, {}).get("keywords", [])
            plan.append(
                TaskSpec(
                    step_id=f"step_{index + 1}",
                    owner=domain,
                    objective=f"As the {domain} desk, assess: {objective}",
                    inputs={
                        "domain": domain,
                        "keywords_matched": keywords,
                        "lead": index == 0,
                        "routed_by": "keyword" if matches else "default",
                    },
                    depends_on=[] if index == 0 else ["step_1"],
                )
            )

        if not ctx.llm.offline:
            refined = await self._refine_with_llm(objective, plan, ctx)
            if refined is not None:
                plan = refined

        if ctx.audit is not None:
            ctx.audit.append(
                self.name,
                "planner.plan_built",
                {
                    "objective": objective[:200],
                    "steps": [
                        {"step_id": t.step_id, "owner": t.owner, "depends_on": t.depends_on}
                        for t in plan
                    ],
                    "refined_by_llm": any(t.inputs.get("refined_by_llm") for t in plan),
                },
            )
        return plan

    async def _refine_with_llm(
        self, objective: str, base_plan: list[TaskSpec], ctx: AgentContext
    ) -> list[TaskSpec] | None:
        """Ask the LLM to refine the plan; reject anything structurally off-contract."""
        base_steps = [
            {"step_id": t.step_id, "owner": t.owner, "objective": t.objective,
             "depends_on": t.depends_on}
            for t in base_plan
        ]
        prompt = (
            "Refine this rule-based case plan for the objective below. You may reorder "
            "steps, rewrite step objectives, or adjust depends_on, but you must keep "
            "exactly the same set of owners and every depends_on entry must reference "
            "an earlier step_id. Return JSON matching the schema.\n"
            f"Objective: {objective}\n"
            f"Rule-based plan: {base_steps}"
        )
        try:
            response = await self.ask_llm(ctx, prompt, json_schema=PLAN_REFINEMENT_SCHEMA)
        except Exception:
            self._audit_refinement(ctx, accepted=False, reason="llm call failed")
            return None
        if response.offline:
            return None
        refined = self._validate_refined(response.json_data(), base_plan, objective)
        self._audit_refinement(
            ctx,
            accepted=refined is not None,
            reason="accepted" if refined is not None else "structural validation failed",
        )
        return refined

    def _validate_refined(
        self, raw: Any, base_plan: list[TaskSpec], objective: str
    ) -> list[TaskSpec] | None:
        """Strict structural validation of an LLM-refined plan.

        Rules: a non-empty bounded ``steps`` list, unique non-empty step ids, owners
        drawn from the known domain set with *exactly* the same owner set as the
        rule-based plan, non-empty objectives, and ``depends_on`` references that only
        point at earlier steps (which guarantees a DAG schedulable in waves).
        """
        if not isinstance(raw, dict):
            return None
        steps = raw.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= _MAX_REFINED_STEPS:
            return None

        base_owners = {t.owner for t in base_plan}
        seen_ids: set[str] = set()
        refined: list[TaskSpec] = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                return None
            step_id = step.get("step_id")
            owner = step.get("owner")
            step_objective = step.get("objective")
            depends_on = step.get("depends_on", [])
            if not isinstance(step_id, str) or not step_id.strip() or step_id in seen_ids:
                return None
            if owner not in DOMAIN_KEYWORDS:
                return None
            if not isinstance(step_objective, str) or not step_objective.strip():
                return None
            if not isinstance(depends_on, list) or not all(
                isinstance(d, str) and d in seen_ids for d in depends_on
            ):
                return None
            seen_ids.add(step_id)
            refined.append(
                TaskSpec(
                    step_id=step_id,
                    owner=owner,
                    objective=f"{step_objective.strip()} (case objective: {objective})",
                    inputs={"domain": owner, "lead": index == 0, "refined_by_llm": True},
                    depends_on=list(depends_on),
                )
            )
        if {t.owner for t in refined} != base_owners:
            return None
        return refined

    def _audit_refinement(self, ctx: AgentContext, *, accepted: bool, reason: str) -> None:
        if ctx.audit is not None:
            ctx.audit.append(
                self.name, "planner.llm_refinement", {"accepted": accepted, "reason": reason}
            )

    # ------------------------------------------------------------------ aggregate --

    async def aggregate(
        self,
        objective: str,
        outputs: list[AgentOutput],
        critique: AgentOutput | None,
        ctx: AgentContext,
    ) -> Decision:
        """Fold domain outputs and the critique into one policy-gated ``Decision``.

        Deterministic semantics:

        - ``action_type`` is ``propose_only`` unless an output explicitly set
          ``requests_execution`` in its data;
        - ``risk_tier`` is the maximum of the outputs' implied tiers (outputs without
          an explicit tier count as ``medium``, the cautious default), ratcheted to at
          least ``high`` when the critic escalated;
        - ``planned_tool_calls`` carries each output's top-ranked candidate action so
          the human reviewer sees exactly what would be done and via which band.
        """
        tier = RiskTier.low if outputs else RiskTier.medium
        requests_execution = False
        planned_tool_calls: list[dict[str, Any]] = []
        for output in outputs:
            tier = bump_tier(tier, self._implied_tier(output))
            if output.data.get("requests_execution"):
                requests_execution = True
            for action in (output.data.get("candidate_actions") or [])[:1]:
                planned_tool_calls.append(
                    {
                        "owner": output.agent,
                        "step_id": output.step_id,
                        "action": action.get("action"),
                        "tool": action.get("tool"),
                        "band": action.get("band", "propose"),
                        "arguments": dict(action.get("arguments") or {}),
                        "requires_approval": bool(action.get("requires_approval")),
                    }
                )

        critic_escalated = bool(critique is not None and critique.data.get("requires_human"))
        if critic_escalated:
            tier = bump_tier(tier, RiskTier.high)

        rationale = self._deterministic_rationale(
            objective, outputs, critique, tier, requests_execution
        )
        if not ctx.llm.offline:
            rationale = await self._narrate_rationale(ctx, rationale)

        decision = Decision(
            objective=objective,
            action_type="execute" if requests_execution else "propose_only",
            planned_tool_calls=planned_tool_calls,
            rationale=rationale,
            risk_tier=tier,
            owner=self.name,
        )
        if ctx.audit is not None:
            ctx.audit.append(
                self.name,
                "planner.decision_aggregated",
                {
                    "decision_id": decision.decision_id,
                    "action_type": decision.action_type,
                    "risk_tier": decision.risk_tier.value,
                    "planned_tool_calls": len(planned_tool_calls),
                    "critic_escalated": critic_escalated,
                },
            )
        return decision

    @staticmethod
    def _implied_tier(output: AgentOutput) -> RiskTier:
        """Parse the tier an output implies; unknown or missing tiers count as medium."""
        raw = output.data.get("risk_tier")
        if raw is None:
            return RiskTier.medium
        try:
            return RiskTier(str(raw))
        except ValueError:
            return RiskTier.medium

    @staticmethod
    def _deterministic_rationale(
        objective: str,
        outputs: list[AgentOutput],
        critique: AgentOutput | None,
        tier: RiskTier,
        requests_execution: bool,
    ) -> str:
        """Offline-safe rationale assembled from the actual run evidence."""
        domain_bits = "; ".join(
            f"{o.agent}: tier={o.data.get('risk_tier', 'medium')}, "
            f"confidence={o.confidence:.2f}"
            for o in outputs
        ) or "no domain outputs"
        high_challenges = 0
        if critique is not None:
            high_challenges = sum(
                1
                for c in critique.data.get("challenges", [])
                if c.get("severity") == "high"
            )
        mode = "execution explicitly requested" if requests_execution else "proposal only"
        return (
            f"Aggregated {len(outputs)} domain output(s) for objective "
            f"'{objective[:120]}'. {domain_bits}. Critic raised {high_challenges} "
            f"high-severity challenge(s). Overall risk tier '{tier.value}'; {mode}."
        )

    async def _narrate_rationale(self, ctx: AgentContext, deterministic: str) -> str:
        """LLM-polished rationale text; never changes action_type or tier."""
        try:
            response = await self.ask_llm(
                ctx,
                "Rewrite this decision rationale for a case record in at most three "
                "sentences without adding facts:\n" + deterministic,
            )
        except Exception:
            return deterministic
        if response.offline or not response.text.strip():
            return deterministic
        return response.text.strip()[:800]

    # ------------------------------------------------------------------------ run --

    async def run(self, task: TaskSpec, ctx: AgentContext) -> AgentOutput:
        """BaseAgent entry point: plan for the task objective and return it as data."""
        plan = await self.decompose(task.objective, ctx)
        return self.output(
            step_id=task.step_id,
            summary=(
                f"planned {len(plan)} step(s) across owners "
                f"{sorted({t.owner for t in plan})}"
            ),
            data={"plan": [t.model_dump() for t in plan]},
            confidence=0.9 if plan else 0.1,
        )

    @staticmethod
    def implied_tier_order(tier: RiskTier) -> int:
        """Expose the canonical tier ordering for callers that rank decisions."""
        return RISK_ORDER[tier]
