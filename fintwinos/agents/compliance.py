"""The financial-crime compliance desk agent: alerts, AML rings, KYC, sanctions.

Observes open alerts and the entity graph neighbourhood through the registry,
rehearses an alert-triage scenario, then ranks investigative actions with
deterministic severity logic: any high/critical alert or a connected ring of three or
more entities escalates the implied tier to ``high`` so the case routes to a human.
SAR-adjacent actions are always flagged ``requires_approval`` — drafting a suspicious
activity report is never an autonomous act.
"""

from __future__ import annotations

from typing import Any

from fintwinos.agents.base import TaskSpec
from fintwinos.agents.domain import DomainAgent, flatten_numeric, severity_rank
from fintwinos.core.types import RiskTier

#: A connected ring at or above this size is treated as an organised typology.
RING_SIZE_HIGH = 3


def find_alerts(data: Any, depth: int = 0) -> list[dict[str, Any]]:
    """Recursively locate alert-shaped dicts (anything carrying a ``severity``)."""
    if depth > 5:
        return []
    found: list[dict[str, Any]] = []
    if isinstance(data, dict):
        if "severity" in data:
            found.append(data)
        else:
            for key in sorted(data, key=str):
                found.extend(find_alerts(data[key], depth + 1))
    elif isinstance(data, list | tuple):
        for item in data:
            found.extend(find_alerts(item, depth + 1))
    return found


def ring_size(data: Any, depth: int = 0) -> int:
    """Largest entity-collection size found under ring/entities/members keys."""
    if depth > 5:
        return 0
    largest = 0
    if isinstance(data, dict):
        for key in sorted(data, key=str):
            value = data[key]
            if str(key).lower() in {"ring", "ring_members", "entities", "members"} and isinstance(
                value, list | tuple
            ):
                largest = max(largest, len(value))
            largest = max(largest, ring_size(value, depth + 1))
    elif isinstance(data, list | tuple):
        for item in data:
            largest = max(largest, ring_size(item, depth + 1))
    return largest


class ComplianceAgent(DomainAgent):
    """Alert / AML / ring / KYC domain owner."""

    name = "compliance"
    domain = "compliance"
    observe_patterns = (
        "observe_*alert*",
        "observe_*case*",
        "observe_*graph*",
        "observe_*kyc*",
        "observe_*sanction*",
        "observe_*compliance*",
    )
    simulate_patterns = (
        "simulate_*triage*",
        "simulate_*alert*",
        "simulate_*aml*",
        "simulate_*compliance*",
        "simulate_*investigation*",
    )
    propose_patterns = (
        "propose_*case*",
        "propose_*investigation*",
        "propose_*compliance*",
        "propose_*sar*",
    )

    def _metrics(
        self,
        task: TaskSpec,
        observations: dict[str, Any],
        simulation: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Alert counts, severity ranks and ring size plus rehearsal metrics."""
        metrics: dict[str, float] = {}
        alerts = find_alerts(observations)
        metrics["open_alerts"] = float(len(alerts))
        if alerts:
            ranks = [severity_rank(a.get("severity")) for a in alerts]
            metrics["max_alert_severity_rank"] = float(max(ranks))
            metrics["high_severity_alerts"] = float(sum(1 for r in ranks if r >= 2))
        size = ring_size(observations)
        if size:
            metrics["ring_size"] = float(size)
        flat = flatten_numeric(observations)
        scores = [v for k, v in flat.items() if "score" in k.lower()]
        if scores:
            metrics["max_alert_score"] = round(max(scores), 6)
        if simulation:
            for key, value in (simulation.get("metrics") or {}).items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metrics[f"sim_{key}"] = float(value)
        return metrics

    def _implied_tier(
        self,
        task: TaskSpec,
        metrics: dict[str, float],
        simulation: dict[str, Any] | None,
    ) -> RiskTier:
        """High/critical alerts or an organised ring escalate to high tier."""
        if metrics.get("max_alert_severity_rank", 0.0) >= 2.0:
            return RiskTier.high
        if metrics.get("ring_size", 0.0) >= RING_SIZE_HIGH:
            return RiskTier.high
        if metrics.get("open_alerts", 0.0) > 0.0:
            return RiskTier.medium
        return RiskTier.low

    def _candidate_actions(
        self,
        task: TaskSpec,
        metrics: dict[str, float],
        simulation: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Investigative ladder ranked by alert severity and ring evidence."""
        open_alerts = int(metrics.get("open_alerts", 0))
        size = int(metrics.get("ring_size", 0))
        high_severity = metrics.get("max_alert_severity_rank", 0.0) >= 2.0
        organised = size >= RING_SIZE_HIGH
        if high_severity or organised:
            evidence = (
                f"{open_alerts} open alert(s), max severity rank "
                f"{metrics.get('max_alert_severity_rank', 0):.0f}, ring size {size}"
            )
            return [
                {
                    "action": "open_investigation_case",
                    "band": "propose",
                    "arguments": {"priority": "high", "ring_size": size},
                    "score": 0.9,
                    "requires_approval": False,
                    "rationale": f"organised typology indicators: {evidence}",
                },
                {
                    "action": "prepare_sar_draft",
                    "band": "propose",
                    "arguments": {"scope": "ring" if organised else "alert"},
                    "score": 0.7,
                    "requires_approval": True,
                    "rationale": "SAR drafting is approval-gated; never autonomous",
                },
                {
                    "action": "escalate_to_mlro",
                    "band": "propose",
                    "arguments": {},
                    "score": 0.6,
                    "requires_approval": False,
                    "rationale": "MLRO must be informed of organised-typology evidence",
                },
            ]
        if open_alerts > 0:
            return [
                {
                    "action": "triage_alerts",
                    "band": "propose",
                    "arguments": {"batch_size": min(open_alerts, 25)},
                    "score": 0.8,
                    "requires_approval": False,
                    "rationale": f"{open_alerts} open alert(s) below high severity",
                },
                {
                    "action": "request_kyc_refresh",
                    "band": "propose",
                    "arguments": {"scope": "flagged_customers"},
                    "score": 0.5,
                    "requires_approval": False,
                    "rationale": "refresh stale KYC on the alerted relationships",
                },
            ]
        return [
            {
                "action": "routine_screening_review",
                "band": "propose",
                "arguments": {"lookback_days": 30},
                "score": 0.7,
                "requires_approval": False,
                "rationale": "no open alerts; confirm screening coverage is intact",
            }
        ]
