"""The customer-operations desk agent: queues, complaints and SLA health.

Observes operational queues and complaint stocks through the registry, rehearses a
staffing scenario, then ranks operational actions with deterministic SLA-threshold
logic: breach rates above 20% are a high-tier service failure, above 5% (or a deep
queue) warrant rebalancing, and a healthy floor keeps the current configuration.
Vulnerable-customer prioritisation always outranks generic throughput actions when
the desk is in a high-tier state.
"""

from __future__ import annotations

from typing import Any

from fintwinos.agents.base import TaskSpec
from fintwinos.agents.domain import (
    DomainAgent,
    flatten_numeric,
    match_numeric,
    max_matched,
)
from fintwinos.core.types import RiskTier

#: SLA breach rate above this is a high-tier service failure.
SLA_BREACH_HIGH = 0.20
#: SLA breach rate above this warrants rebalancing (medium tier).
SLA_BREACH_MEDIUM = 0.05
#: Total queue depth above this is itself a medium-tier signal.
QUEUE_DEPTH_MEDIUM = 100.0


class CustomerOpsAgent(DomainAgent):
    """Queue / complaint / SLA / customer-service domain owner."""

    name = "customer_ops"
    domain = "customer_ops"
    observe_patterns = (
        "observe_*queue*",
        "observe_*ops*",
        "observe_*complaint*",
        "observe_*sla*",
        "observe_*service*",
    )
    simulate_patterns = (
        "simulate_*queue*",
        "simulate_*staffing*",
        "simulate_*ops*",
        "simulate_*service*",
    )
    propose_patterns = (
        "propose_*queue*",
        "propose_*staffing*",
        "propose_*ops*",
        "propose_*service*",
    )

    def _metrics(
        self,
        task: TaskSpec,
        observations: dict[str, Any],
        simulation: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Queue depth, SLA breach rate and complaint stock plus rehearsal metrics."""
        metrics: dict[str, float] = {}
        flat = flatten_numeric(observations)
        depth = sum(match_numeric(flat, ("queue", "backlog", "depth")).values())
        if depth:
            metrics["queue_depth_total"] = round(depth, 2)
        breach_rate = max_matched(flat, ("sla", "breach_rate"))
        if breach_rate is not None:
            metrics["sla_breach_rate"] = round(breach_rate, 6)
        complaints = sum(match_numeric(flat, ("complaint",)).values())
        if complaints:
            metrics["complaints_open"] = round(complaints, 2)
        if simulation:
            for key, value in (simulation.get("metrics") or {}).items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metrics[f"sim_{key}"] = float(value)
        return metrics

    def _breach_signal(self, metrics: dict[str, float]) -> float:
        """Worst SLA breach rate visible across observed and simulated metrics."""
        rates = match_numeric(metrics, ("sla", "breach"))
        return max(rates.values()) if rates else 0.0

    def _implied_tier(
        self,
        task: TaskSpec,
        metrics: dict[str, float],
        simulation: dict[str, Any] | None,
    ) -> RiskTier:
        """SLA-threshold logic; deep queues alone are a medium-tier signal."""
        breach = self._breach_signal(metrics)
        if breach > SLA_BREACH_HIGH:
            return RiskTier.high
        if breach > SLA_BREACH_MEDIUM:
            return RiskTier.medium
        if metrics.get("queue_depth_total", 0.0) > QUEUE_DEPTH_MEDIUM:
            return RiskTier.medium
        return RiskTier.low

    def _candidate_actions(
        self,
        task: TaskSpec,
        metrics: dict[str, float],
        simulation: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Staffing/queue ladder ranked by the SLA breach signal."""
        breach = self._breach_signal(metrics)
        depth = metrics.get("queue_depth_total", 0.0)
        signal = f"SLA breach rate {breach:.3f}, queue depth {depth:.0f}"
        if breach > SLA_BREACH_HIGH:
            return [
                {
                    "action": "surge_staffing",
                    "band": "propose",
                    "arguments": {"extra_fte": 8, "duration_days": 5},
                    "score": 0.9,
                    "requires_approval": True,
                    "rationale": f"service failure in progress: {signal}",
                },
                {
                    "action": "prioritise_vulnerable_customers",
                    "band": "propose",
                    "arguments": {"queue_policy": "vulnerability_first"},
                    "score": 0.8,
                    "requires_approval": False,
                    "rationale": "conduct duty: protect vulnerable customers during the backlog",
                },
                {
                    "action": "pause_non_critical_outbound",
                    "band": "propose",
                    "arguments": {},
                    "score": 0.5,
                    "requires_approval": False,
                    "rationale": "free capacity for inbound service recovery",
                },
            ]
        if breach > SLA_BREACH_MEDIUM or depth > QUEUE_DEPTH_MEDIUM:
            return [
                {
                    "action": "rebalance_queue_staffing",
                    "band": "propose",
                    "arguments": {"rebalance_fraction": 0.2},
                    "score": 0.8,
                    "requires_approval": False,
                    "rationale": f"pressure building: {signal}",
                },
                {
                    "action": "automate_simple_triage",
                    "band": "propose",
                    "arguments": {"categories": ["balance_enquiry", "card_reissue"]},
                    "score": 0.5,
                    "requires_approval": False,
                    "rationale": "deflect simple contacts to automated handling",
                },
            ]
        return [
            {
                "action": "maintain_current_staffing",
                "band": "propose",
                "arguments": {},
                "score": 0.8,
                "requires_approval": False,
                "rationale": f"service healthy ({signal}); no change needed",
            },
            {
                "action": "review_complaint_root_causes",
                "band": "propose",
                "arguments": {"lookback_days": 30},
                "score": 0.4,
                "requires_approval": False,
                "rationale": "routine root-cause review to prevent future backlogs",
            },
        ]
