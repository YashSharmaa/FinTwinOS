"""Shared machinery for FinTwinOS domain agents.

Every domain agent (risk, compliance, treasury, customer ops) follows the same
auditable pipeline:

1. **Observe** — discover ``observe_*`` tools matching the domain's patterns in the
   registry and call them through ``ctx.tool`` so every read is audited.
2. **Simulate** — run one ``simulate_*`` rehearsal for the objective with a
   deterministic seed, capturing point estimates *and* confidence intervals.
3. **Propose** — rank candidate actions with deterministic domain logic and, when a
   ``propose_*`` tool exists, lodge the top-ranked candidate as a formal proposal.

The LLM only ever *narrates* the summary; selection, ranking, tier inference and
confidence are pure deterministic functions of the tool results, so the agent behaves
identically offline (``FINTWIN_OFFLINE=1``) and online. Tool discovery is
pattern-based so the agents keep working as the tool catalog evolves; missing tools
degrade into recorded warnings, never crashes.
"""

from __future__ import annotations

import re
from typing import Any

from fintwinos.agents.base import AgentContext, AgentOutput, BaseAgent, TaskSpec
from fintwinos.core.types import RISK_ORDER, RiskTier, SimulationResult, ToolBand, ToolResult

#: Objectives containing these words are treated as elevated-severity situations and
#: bump the implied risk tier to at least ``high`` (conservative by construction).
ESCALATION_PATTERN = re.compile(
    r"\b(severe|crisis|breach|breaching|urgent|emergency|critical|immediately)\b",
    re.IGNORECASE,
)

#: Objectives containing these verbs are read as an *explicit* request to act on the
#: world rather than to analyse it. The planner turns this into an ``execute``
#: decision, which the policy gate always routes to a human.
EXECUTION_PATTERN = re.compile(
    r"\b(execute|disburse|wire|settle|enact|transact)\b",
    re.IGNORECASE,
)

_MAX_FLAT_ENTRIES = 200
_MAX_FLAT_DEPTH = 6

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def flatten_numeric(
    data: Any,
    prefix: str = "",
    out: dict[str, float] | None = None,
    depth: int = 0,
) -> dict[str, float]:
    """Flatten arbitrarily nested tool payloads into ``dotted.key -> float`` pairs.

    Booleans are excluded (they are ints in Python but not metrics), recursion is
    bounded by depth and entry count so a pathological payload cannot stall an agent.
    """
    out = out if out is not None else {}
    if depth > _MAX_FLAT_DEPTH or len(out) >= _MAX_FLAT_ENTRIES:
        return out
    if isinstance(data, bool):
        return out
    if isinstance(data, int | float):
        out[prefix or "value"] = float(data)
        return out
    if isinstance(data, dict):
        for key in sorted(data, key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_numeric(data[key], child_prefix, out, depth + 1)
    elif isinstance(data, list | tuple):
        for idx, item in enumerate(data):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            flatten_numeric(item, child_prefix, out, depth + 1)
    return out


def match_numeric(flat: dict[str, float], substrings: tuple[str, ...]) -> dict[str, float]:
    """Select flattened numeric leaves whose key contains any of the substrings."""
    lowered = tuple(s.lower() for s in substrings)
    return {k: v for k, v in flat.items() if any(s in k.lower() for s in lowered)}


def sum_matched(flat: dict[str, float], substrings: tuple[str, ...]) -> float:
    """Sum of all numeric leaves matching the substrings (0.0 when none match)."""
    return float(sum(match_numeric(flat, substrings).values()))


def max_matched(flat: dict[str, float], substrings: tuple[str, ...]) -> float | None:
    """Maximum numeric leaf matching the substrings, or ``None`` when none match."""
    matched = match_numeric(flat, substrings)
    return max(matched.values()) if matched else None


def extract_simulation(payload: Any) -> dict[str, Any] | None:
    """Normalise a simulate-tool payload into a compact simulation block.

    Accepts a ``SimulationResult`` instance, its ``model_dump`` dict, or a dict that
    nests one under ``simulation``/``result``. Returns ``None`` when nothing
    simulation-shaped is present, so the critic can flag the missing rehearsal.
    """
    if isinstance(payload, SimulationResult):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        return None
    candidate = payload
    for nested_key in ("simulation", "result"):
        nested = candidate.get(nested_key)
        if isinstance(nested, dict) and ("metrics" in nested or "confidence" in nested):
            candidate = nested
            break
    if "metrics" not in candidate and "confidence" not in candidate:
        return None
    return {
        "simulator": candidate.get("simulator"),
        "scenario_name": candidate.get("scenario_name"),
        "metrics": dict(candidate.get("metrics") or {}),
        "confidence": dict(candidate.get("confidence") or {}),
        "calibration": dict(candidate.get("calibration") or {}),
        "warnings": list(candidate.get("warnings") or []),
        "seed": candidate.get("seed"),
    }


def bump_tier(current: RiskTier, floor: RiskTier) -> RiskTier:
    """Return the higher of two risk tiers (tiers only ever ratchet upwards)."""
    return current if RISK_ORDER[current] >= RISK_ORDER[floor] else floor


def severity_rank(value: Any) -> int:
    """Rank an alert/severity label; unknown labels rank as medium (cautious)."""
    return _SEVERITY_RANK.get(str(value).lower(), 1)


class DomainAgent(BaseAgent):
    """Base class for the four domain swarm agents.

    Subclasses configure tool-discovery patterns and override the three deterministic
    hooks — ``_metrics``, ``_candidate_actions`` and ``_implied_tier`` — with domain
    logic. Everything else (discovery, tolerant invocation, evidence capture,
    confidence scoring, optional LLM narration) is shared and identical across
    domains, which keeps the critic's cross-checks meaningful.
    """

    domain: str = "generic"
    role: str = "domain"
    observe_patterns: tuple[str, ...] = ()
    simulate_patterns: tuple[str, ...] = ()
    propose_patterns: tuple[str, ...] = ()
    max_observe_tools: int = 3

    # ------------------------------------------------------------------ pipeline --

    async def run(self, task: TaskSpec, ctx: AgentContext) -> AgentOutput:
        """Execute observe -> simulate -> propose and return a structured output."""
        objective = task.objective or ""
        warnings: list[str] = []
        tool_results: list[dict[str, Any]] = []
        observations: dict[str, Any] = {}

        # 1. Observe ------------------------------------------------------------
        observe_specs = self._discover(ctx, ToolBand.observe, self.observe_patterns)
        observe_specs = observe_specs[: self.max_observe_tools]
        if not observe_specs:
            warnings.append(
                f"no observe tools matched patterns for domain '{self.domain}'; "
                "running on blackboard context only"
            )
        for spec in observe_specs:
            result = await self._call_tolerant(
                ctx, spec.name, self._observe_argument_variants(objective), task
            )
            tool_results.append(self._compact(result))
            if result.ok:
                observations[spec.name] = result.data
            else:
                warnings.append(f"observe tool '{spec.name}' failed: {result.error}")

        # 2. Simulate -------------------------------------------------------------
        simulation: dict[str, Any] | None = None
        simulate_specs = self._discover(ctx, ToolBand.simulate, self.simulate_patterns)
        if simulate_specs:
            spec = simulate_specs[0]
            result = await self._call_tolerant(
                ctx, spec.name, self._simulate_argument_variants(objective, ctx), task
            )
            tool_results.append(self._compact(result))
            if result.ok:
                simulation = extract_simulation(result.data)
                if simulation is None:
                    warnings.append(
                        f"simulate tool '{spec.name}' returned no recognisable "
                        "simulation block"
                    )
                elif not simulation.get("confidence"):
                    warnings.append(
                        f"simulate tool '{spec.name}' returned no confidence intervals"
                    )
            else:
                warnings.append(f"simulate tool '{spec.name}' failed: {result.error}")
        else:
            warnings.append(
                f"no simulate tool matched patterns for domain '{self.domain}'; "
                "rehearsal skipped"
            )

        # 3. Deterministic domain analysis ------------------------------------------
        metrics = self._metrics(task, observations, simulation)
        candidates = sorted(
            self._candidate_actions(task, metrics, simulation),
            key=lambda c: (-float(c.get("score", 0.0)), str(c.get("action", ""))),
        )
        tier = self._implied_tier(task, metrics, simulation)
        if ESCALATION_PATTERN.search(objective):
            tier = bump_tier(tier, RiskTier.high)

        requests_execution = bool(EXECUTION_PATTERN.search(objective))
        if requests_execution and candidates:
            # An explicit execution request must carry its human approval routing so
            # the critic and the policy gate can both see it.
            candidates[0]["requires_approval"] = True
            candidates[0]["approval_route"] = (
                "human dual-control approval via ApprovalToken before any execute call"
            )

        # 4. Propose -----------------------------------------------------------------
        proposal: Any = None
        propose_specs = self._discover(ctx, ToolBand.propose, self.propose_patterns)
        propose_tool = propose_specs[0].name if propose_specs else None
        if propose_tool and candidates:
            best = candidates[0]
            best.setdefault("tool", propose_tool)
            result = await self._call_tolerant(
                ctx, propose_tool, self._propose_argument_variants(best, objective), task
            )
            tool_results.append(self._compact(result))
            if result.ok:
                proposal = result.data
            else:
                warnings.append(f"propose tool '{propose_tool}' failed: {result.error}")
        elif candidates and not propose_tool:
            warnings.append(
                f"no propose tool matched patterns for domain '{self.domain}'; "
                "candidates recorded without a formal proposal"
            )

        # 5. Narrate + score -----------------------------------------------------------
        summary = self._deterministic_summary(objective, metrics, candidates, tier, simulation)
        if not ctx.llm.offline:
            summary = await self._narrate(ctx, summary, metrics, candidates, warnings)

        confidence = self._confidence(
            observed_ok=len(observations),
            observed_total=len(observe_specs),
            simulation=simulation,
            proposal_ok=proposal is not None,
            warning_count=len(warnings),
        )

        data: dict[str, Any] = {
            "domain": self.domain,
            "metrics": metrics,
            "candidate_actions": candidates,
            "simulation": simulation,
            "risk_tier": tier.value,
            "requests_execution": requests_execution,
            "proposal": proposal,
            "observations_used": sorted(observations),
        }
        ctx.blackboard.post(f"domain.{self.domain}", data, actor=self.name)
        if ctx.audit is not None:
            ctx.audit.append(
                self.name,
                "agent.completed",
                {
                    "step_id": task.step_id,
                    "domain": self.domain,
                    "risk_tier": tier.value,
                    "candidates": len(candidates),
                    "confidence": confidence,
                },
            )
        return self.output(
            step_id=task.step_id,
            summary=summary,
            data=data,
            tool_results=tool_results,
            confidence=confidence,
            warnings=warnings,
        )

    # ----------------------------------------------------------------- domain hooks --

    def _metrics(
        self,
        task: TaskSpec,
        observations: dict[str, Any],
        simulation: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Derive named metrics from observations and the simulation rehearsal."""
        metrics: dict[str, float] = {}
        flat = flatten_numeric(observations)
        if flat:
            metrics["observed_numeric_fields"] = float(len(flat))
        if simulation:
            for key, value in (simulation.get("metrics") or {}).items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metrics[f"sim_{key}"] = float(value)
        return metrics

    def _candidate_actions(
        self,
        task: TaskSpec,
        metrics: dict[str, float],
        simulation: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build the ranked candidate-action list (deterministic, domain-specific)."""
        return [
            {
                "action": f"monitor_{self.domain}",
                "band": "propose",
                "arguments": {},
                "score": 0.5,
                "requires_approval": False,
                "rationale": "default monitoring stance: no domain-specific signal",
            }
        ]

    def _implied_tier(
        self,
        task: TaskSpec,
        metrics: dict[str, float],
        simulation: dict[str, Any] | None,
    ) -> RiskTier:
        """Infer the risk tier this output implies for the aggregated decision."""
        return RiskTier.low

    # -------------------------------------------------------------------- helpers --

    def _discover(
        self, ctx: AgentContext, band: ToolBand, patterns: tuple[str, ...]
    ) -> list[Any]:
        """Find registered tools in a band matching any of the glob patterns."""
        seen: set[str] = set()
        found: list[Any] = []
        for pattern in patterns:
            for spec in ctx.registry.list_specs(band=band, pattern=pattern):
                if spec.name not in seen:
                    seen.add(spec.name)
                    found.append(spec)
        return found

    async def _call_tolerant(
        self,
        ctx: AgentContext,
        name: str,
        argument_variants: list[dict[str, Any]],
        task: TaskSpec,
    ) -> ToolResult:
        """Call a tool, retrying with simpler argument shapes on schema rejection.

        Tool input schemas are owned by the tools module and may evolve; the agent
        tries its richest argument shape first and falls back to leaner ones only
        when the registry's JSON Schema validation rejects the call. Genuine handler
        failures are never retried.
        """
        last: ToolResult | None = None
        for arguments in argument_variants:
            last = await ctx.tool(name, arguments, caller=self.name, ticket_id=task.step_id)
            if last.ok:
                return last
            if not (last.error and "failed validation" in last.error):
                break
        assert last is not None  # argument_variants is never empty
        return last

    def _observe_argument_variants(self, objective: str) -> list[dict[str, Any]]:
        """Argument shapes attempted for observe tools, richest-compatible first."""
        return [{}, {"query": objective[:120]}, {"limit": 10}]

    def _simulate_argument_variants(
        self, objective: str, ctx: AgentContext
    ) -> list[dict[str, Any]]:
        """Argument shapes attempted for simulate tools.

        The scenario severity is derived deterministically from the objective text,
        and the seed always comes from settings so reruns are reproducible.
        """
        seed = int(ctx.settings.seed)
        severity = "severe" if ESCALATION_PATTERN.search(objective) else "baseline"
        params = {"severity": severity, "objective": objective[:200], "horizon_days": 10}
        scenario = {"name": f"{self.domain}_rehearsal", "kind": "what_if", "params": params}
        return [
            {"scenario": scenario, "seed": seed},
            {"scenario_name": scenario["name"], "params": params, "seed": seed},
            {"severity": severity, "seed": seed},
            {"seed": seed},
            {},
        ]

    def _propose_argument_variants(
        self, candidate: dict[str, Any], objective: str
    ) -> list[dict[str, Any]]:
        """Argument shapes attempted for propose tools."""
        action = str(candidate.get("action", "monitor"))
        return [
            {
                "action": action,
                "rationale": str(candidate.get("rationale", ""))[:300],
                "params": dict(candidate.get("arguments") or {}),
            },
            {"action": action},
            {},
        ]

    @staticmethod
    def _compact(result: ToolResult) -> dict[str, Any]:
        """Compact evidence record kept on the output for the critic's cross-checks."""
        record: dict[str, Any] = {
            "tool": result.tool,
            "ok": result.ok,
            "audit_ref": result.audit_ref,
            "error": result.error,
        }
        if isinstance(result.data, dict):
            record["data_keys"] = sorted(result.data)[:8]
        return record

    @staticmethod
    def _confidence(
        *,
        observed_ok: int,
        observed_total: int,
        simulation: dict[str, Any] | None,
        proposal_ok: bool,
        warning_count: int,
    ) -> float:
        """Deterministic confidence score in [0.05, 0.95].

        Confidence is earned from evidence: successful observations, a rehearsal with
        confidence intervals, and an accepted proposal each add to it; every warning
        subtracts. No randomness, no LLM input.
        """
        obs_fraction = (observed_ok / observed_total) if observed_total else 0.0
        has_ci = bool(simulation and simulation.get("confidence"))
        score = (
            0.35
            + 0.30 * obs_fraction
            + 0.20 * (1.0 if has_ci else 0.0)
            + 0.10 * (1.0 if proposal_ok else 0.0)
            - 0.05 * warning_count
        )
        return round(min(0.95, max(0.05, score)), 3)

    def _deterministic_summary(
        self,
        objective: str,
        metrics: dict[str, float],
        candidates: list[dict[str, Any]],
        tier: RiskTier,
        simulation: dict[str, Any] | None,
    ) -> str:
        """Offline-safe summary template citing the evidence actually gathered."""
        top = candidates[0]["action"] if candidates else "none"
        rehearsal = "rehearsed with confidence intervals" if (
            simulation and simulation.get("confidence")
        ) else "not rehearsed"
        return (
            f"[{self.domain}] objective '{objective[:90]}': {len(metrics)} metrics derived "
            f"from tool results, {len(candidates)} candidate action(s) ranked "
            f"(top: {top}), implied risk tier '{tier.value}', scenario {rehearsal}."
        )

    async def _narrate(
        self,
        ctx: AgentContext,
        deterministic_summary: str,
        metrics: dict[str, float],
        candidates: list[dict[str, Any]],
        warnings: list[str],
    ) -> str:
        """Ask the LLM to polish the narration; the deterministic text always wins on failure."""
        prompt = (
            "Rewrite this domain-agent summary for a case file in at most three "
            "sentences. Do not add facts, numbers or actions that are not present.\n"
            f"Summary: {deterministic_summary}\n"
            f"Metrics: {metrics}\nTop candidates: {[c.get('action') for c in candidates[:3]]}"
        )
        try:
            response = await self.ask_llm(ctx, prompt)
        except Exception:
            warnings.append("llm narration failed; using deterministic summary")
            return deterministic_summary
        if response.offline or not response.text.strip():
            return deterministic_summary
        return response.text.strip()[:600]
