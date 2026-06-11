"""The policy gate consulted on every tool call and every planner decision.

Semantics:

- Rules are evaluated in order; the first matching ``deny`` wins immediately.
- ``require_approval`` matches accumulate — any one of them flags human review.
- ``allow`` matches mark the call as explicitly permitted.
- The execute band is **default-deny**: with no explicit ``allow`` rule, it is blocked.
- observe/simulate/propose are default-allow, with high/critical-risk propose calls
  flagged for human review by the shipped default rules.
"""

from __future__ import annotations

import fnmatch
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from fintwinos.core.types import RISK_ORDER, Decision, RiskTier, SideEffectClass, ToolBand


class RuleEffect(StrEnum):
    allow = "allow"
    deny = "deny"
    require_approval = "require_approval"


class PolicyRule(BaseModel):
    name: str
    description: str = ""
    effect: RuleEffect
    bands: list[ToolBand] | None = None        # None = any band
    tools: list[str] | None = None             # glob patterns; None = any tool
    min_risk_tier: RiskTier | None = None      # rule applies at-or-above this tier
    side_effects: list[SideEffectClass] | None = None  # None = any side-effect class
    conditions: dict[str, Any] = Field(default_factory=dict)  # equality on arguments

    def matches(self, spec: Any, arguments: dict[str, Any]) -> bool:
        if self.bands is not None and spec.band not in self.bands:
            return False
        if self.tools is not None and not any(
            fnmatch.fnmatch(spec.name, pattern) for pattern in self.tools
        ):
            return False
        if self.min_risk_tier is not None and (
            RISK_ORDER[spec.risk_tier] < RISK_ORDER[self.min_risk_tier]
        ):
            return False
        if self.side_effects is not None and spec.side_effect not in self.side_effects:
            return False
        for key, expected in self.conditions.items():
            if arguments.get(key) != expected:
                return False
        return True


class Verdict(BaseModel):
    allowed: bool
    requires_human_review: bool = False
    reasons: list[str] = Field(default_factory=list)
    matched_rules: list[str] = Field(default_factory=list)


def default_rules() -> list[PolicyRule]:
    """The shipped default policy pack: conservative on autonomy."""
    return [
        PolicyRule(
            name="deny-irreversible-critical",
            description="No critical-tier irreversible action may run via policy alone.",
            effect=RuleEffect.deny,
            bands=[ToolBand.execute],
            min_risk_tier=RiskTier.critical,
            side_effects=[SideEffectClass.irreversible],
        ),
        PolicyRule(
            name="review-high-risk-proposals",
            description="High and critical risk proposals are flagged for human review.",
            effect=RuleEffect.require_approval,
            bands=[ToolBand.propose],
            min_risk_tier=RiskTier.high,
        ),
        PolicyRule(
            name="review-all-execute",
            description="Every execute call requires human review on top of approval tokens.",
            effect=RuleEffect.require_approval,
            bands=[ToolBand.execute],
        ),
    ]


class PolicyGate:
    def __init__(self, rules: list[PolicyRule] | None = None):
        self.rules = rules if rules is not None else default_rules()

    def add_rule(self, rule: PolicyRule, prepend: bool = False) -> None:
        if prepend:
            self.rules.insert(0, rule)
        else:
            self.rules.append(rule)

    # -- tool calls ---------------------------------------------------------------

    def check_tool_call(self, spec: Any, arguments: dict[str, Any], context: Any = None) -> Verdict:
        matched: list[str] = []
        reasons: list[str] = []
        requires_review = False
        explicitly_allowed = False

        for rule in self.rules:
            if not rule.matches(spec, arguments):
                continue
            matched.append(rule.name)
            if rule.effect == RuleEffect.deny:
                return Verdict(
                    allowed=False,
                    requires_human_review=True,
                    reasons=[rule.description or rule.name],
                    matched_rules=matched,
                )
            if rule.effect == RuleEffect.require_approval:
                requires_review = True
                reasons.append(rule.description or rule.name)
            if rule.effect == RuleEffect.allow:
                explicitly_allowed = True

        if spec.band == ToolBand.execute and not explicitly_allowed:
            return Verdict(
                allowed=False,
                requires_human_review=True,
                reasons=["execute band is default-deny: no allow rule matched"],
                matched_rules=matched,
            )

        return Verdict(
            allowed=True,
            requires_human_review=requires_review,
            reasons=reasons,
            matched_rules=matched,
        )

    # -- planner decisions ----------------------------------------------------------

    def check_decision(self, decision: Decision) -> Verdict:
        if decision.action_type == "execute":
            return Verdict(
                allowed=True,
                requires_human_review=True,
                reasons=["decisions that execute always require human review"],
                matched_rules=["decision-execute-review"],
            )
        requires_review = RISK_ORDER[decision.risk_tier] >= RISK_ORDER[RiskTier.high]
        return Verdict(
            allowed=True,
            requires_human_review=requires_review,
            reasons=(["high-risk proposal flagged for review"] if requires_review else []),
            matched_rules=(["decision-high-risk-review"] if requires_review else []),
        )
