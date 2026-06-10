"""The critic agent: adversarial review of every domain output before aggregation.

This is the "debating" half of the hierarchical-and-debating pattern. The critic
red-teams each candidate output with four deterministic checks:

1. **Evidence** — every output that makes claims (metrics, candidate actions, or a
   confident summary) must reference at least one successful, audited tool result.
   An output with claims but no tool evidence is a fabrication and is challenged at
   high severity.
2. **Rehearsal** — candidate actions must be backed by a simulation block carrying
   confidence intervals; bare point estimates (or no rehearsal at all) are challenged
   at high severity.
3. **Execution routing** — no execute-band call may be planned without explicit human
   approval routing (``requires_approval``); a bare execute plan is challenged at
   high severity.
4. **Confidence consistency** — confidence must lie in [0, 1] and high confidence is
   inconsistent with failed tool calls or a pile of warnings.

Any standing high-severity challenge sets ``requires_human=True`` in the critique
data, which the planner ratchets into a high-tier decision — so a successfully
challenged case always reaches a human. The LLM may narrate the critique but never
alters the challenge list.
"""

from __future__ import annotations

from typing import Any

from fintwinos.agents.base import AgentContext, AgentOutput, BaseAgent, TaskSpec
from fintwinos.models.llm_routing.router import TaskClass

#: Confidence above this is "high" for the consistency check.
HIGH_CONFIDENCE = 0.75
#: Warning count at which high confidence becomes inconsistent.
WARNING_TOLERANCE = 2

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


class CriticAgent(BaseAgent):
    """Red-teams domain outputs; escalates to a human when challenges stand."""

    name = "critic"
    role = "critic"
    task_class = TaskClass.critique

    async def review(
        self, objective: str, outputs: list[AgentOutput], ctx: AgentContext
    ) -> AgentOutput:
        """Run all checks over the outputs and return the structured critique."""
        challenges: list[dict[str, Any]] = []
        for output in outputs:
            challenges.extend(self._check_evidence(output))
            challenges.extend(self._check_simulation(output))
            challenges.extend(self._check_execution_routing(output))
            challenges.extend(self._check_confidence(output))

        challenges.sort(
            key=lambda c: (-SEVERITY_RANK.get(c["severity"], 0), c["check"], str(c["agent"]))
        )
        high_count = sum(1 for c in challenges if c["severity"] == "high")
        medium_count = sum(1 for c in challenges if c["severity"] == "medium")
        requires_human = high_count > 0

        data: dict[str, Any] = {
            "challenges": challenges,
            "requires_human": requires_human,
            "high_count": high_count,
            "medium_count": medium_count,
            "checked_outputs": len(outputs),
        }
        ctx.blackboard.post("critique", data, actor=self.name)
        if ctx.audit is not None:
            ctx.audit.append(
                self.name,
                "critic.review_completed",
                {
                    "objective": objective[:200],
                    "checked_outputs": len(outputs),
                    "challenges": len(challenges),
                    "high_count": high_count,
                    "requires_human": requires_human,
                },
            )

        summary = (
            f"critic reviewed {len(outputs)} output(s): {len(challenges)} challenge(s) "
            f"({high_count} high, {medium_count} medium); "
            f"{'ESCALATING to human review' if requires_human else 'no escalation required'}"
        )
        if not ctx.llm.offline:
            summary = await self._narrate(ctx, summary, challenges)

        confidence = round(
            min(0.9, max(0.4, 0.9 - 0.08 * high_count - 0.04 * medium_count)), 3
        )
        return self.output(
            summary=summary,
            data=data,
            confidence=confidence if outputs else 0.3,
        )

    async def run(self, task: TaskSpec, ctx: AgentContext) -> AgentOutput:
        """BaseAgent entry point: review the outputs posted on the blackboard."""
        outputs: list[AgentOutput] = []
        for item in ctx.blackboard.read("outputs"):
            if isinstance(item, AgentOutput):
                outputs.append(item)
            elif isinstance(item, dict):
                try:
                    outputs.append(AgentOutput.model_validate(item))
                except Exception:  # noqa: BLE001 — malformed posts are skipped, not fatal
                    continue
        return await self.review(task.objective, outputs, ctx)

    # -------------------------------------------------------------------- checks --

    @staticmethod
    def _challenge(
        check: str, severity: str, output: AgentOutput, detail: str
    ) -> dict[str, Any]:
        return {
            "check": check,
            "severity": severity,
            "agent": output.agent,
            "step_id": output.step_id,
            "detail": detail,
        }

    def _check_evidence(self, output: AgentOutput) -> list[dict[str, Any]]:
        """Claims must trace to at least one successful, audited tool result."""
        challenges: list[dict[str, Any]] = []
        ok_results = [t for t in output.tool_results if t.get("ok")]
        metrics = output.data.get("metrics") or {}
        candidates = output.data.get("candidate_actions") or []
        claims_present = bool(metrics) or bool(candidates) or output.confidence >= 0.5

        if claims_present and not ok_results:
            challenges.append(
                self._challenge(
                    "evidence",
                    "high",
                    output,
                    f"output claims {len(metrics)} metric(s) and {len(candidates)} "
                    "candidate action(s) but references no successful tool result — "
                    "possible fabrication",
                )
            )
        elif not output.tool_results:
            challenges.append(
                self._challenge(
                    "evidence",
                    "medium",
                    output,
                    "output carries no tool results at all; nothing is verifiable",
                )
            )
        for record in ok_results:
            if not record.get("audit_ref"):
                challenges.append(
                    self._challenge(
                        "evidence",
                        "medium",
                        output,
                        f"tool result '{record.get('tool')}' lacks an audit_ref; "
                        "the call cannot be traced in the audit chain",
                    )
                )
        return challenges

    def _check_simulation(self, output: AgentOutput) -> list[dict[str, Any]]:
        """Candidate actions need a rehearsal with confidence intervals behind them."""
        candidates = output.data.get("candidate_actions") or []
        if not candidates:
            return []
        simulation = output.data.get("simulation")
        if not isinstance(simulation, dict):
            return [
                self._challenge(
                    "simulation",
                    "high",
                    output,
                    f"{len(candidates)} candidate action(s) proposed without any "
                    "simulation rehearsal",
                )
            ]
        if not simulation.get("confidence"):
            return [
                self._challenge(
                    "simulation",
                    "high",
                    output,
                    "simulation rehearsal carries no confidence intervals; "
                    "bare point estimates are not decision-grade",
                )
            ]
        return []

    def _check_execution_routing(self, output: AgentOutput) -> list[dict[str, Any]]:
        """No execute-band plan may exist without explicit human approval routing."""
        challenges: list[dict[str, Any]] = []
        candidates = output.data.get("candidate_actions") or []
        for action in candidates:
            tool_name = str(action.get("tool") or "")
            is_execute = action.get("band") == "execute" or tool_name.startswith("execute_")
            if is_execute and not action.get("requires_approval"):
                challenges.append(
                    self._challenge(
                        "execution_routing",
                        "high",
                        output,
                        f"candidate '{action.get('action')}' plans an execute-band "
                        "call without approval routing",
                    )
                )
        if output.data.get("requests_execution"):
            routed = any(c.get("requires_approval") for c in candidates)
            if not routed:
                challenges.append(
                    self._challenge(
                        "execution_routing",
                        "high",
                        output,
                        "output requests execution but no candidate action carries "
                        "human approval routing",
                    )
                )
        return challenges

    def _check_confidence(self, output: AgentOutput) -> list[dict[str, Any]]:
        """Confidence must be well-formed and consistent with the evidence trail."""
        challenges: list[dict[str, Any]] = []
        confidence = output.confidence
        if confidence < 0.0 or confidence > 1.0:
            challenges.append(
                self._challenge(
                    "confidence",
                    "high",
                    output,
                    f"confidence {confidence} is outside [0, 1]",
                )
            )
            return challenges
        any_failed = any(not t.get("ok") for t in output.tool_results)
        if confidence >= HIGH_CONFIDENCE and (
            any_failed or len(output.warnings) > WARNING_TOLERANCE
        ):
            challenges.append(
                self._challenge(
                    "confidence",
                    "medium",
                    output,
                    f"confidence {confidence:.2f} is inconsistent with "
                    f"{len(output.warnings)} warning(s) and "
                    f"{'failed tool calls' if any_failed else 'the warning trail'}",
                )
            )
        return challenges

    # ------------------------------------------------------------------ narration --

    async def _narrate(
        self, ctx: AgentContext, deterministic_summary: str, challenges: list[dict[str, Any]]
    ) -> str:
        """LLM-polished critique narration; the challenge list itself never changes."""
        try:
            response = await self.ask_llm(
                ctx,
                "Rewrite this critique summary in at most three sentences without "
                "adding or removing findings:\n"
                f"{deterministic_summary}\nChallenges: "
                f"{[(c['check'], c['severity'], c['agent']) for c in challenges[:8]]}",
            )
        except Exception:
            return deterministic_summary
        if response.offline or not response.text.strip():
            return deterministic_summary
        return response.text.strip()[:600]
