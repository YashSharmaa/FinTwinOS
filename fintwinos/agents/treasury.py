"""The treasury desk agent: liquidity, funding and cash management.

Observes the liquidity ladder, cash balances and the funding profile through the
registry; rehearses a liquidity stress scenario; then ranks funding actions with
deterministic threshold logic on the simulated coverage metrics. The implied risk
tier ratchets up when simulated coverage (LCR-style ratios) drops below 1.0 or a
shortfall appears, so a strained balance sheet always routes to a human.
"""

from __future__ import annotations

from typing import Any

from fintwinos.agents.base import TaskSpec
from fintwinos.agents.domain import (
    DomainAgent,
    flatten_numeric,
    match_numeric,
    max_matched,
    sum_matched,
)
from fintwinos.core.types import RiskTier

#: Coverage ratios below this are a hard liquidity problem (high tier).
COVERAGE_HIGH_THRESHOLD = 1.0
#: Coverage ratios below this leave little headroom (medium tier).
COVERAGE_MEDIUM_THRESHOLD = 1.1


class TreasuryAgent(DomainAgent):
    """Liquidity / funding / cash domain owner."""

    name = "treasury"
    domain = "treasury"
    observe_patterns = (
        "observe_*liquidity*",
        "observe_*cash*",
        "observe_*funding*",
        "observe_*treasury*",
        "observe_*deposit*",
    )
    simulate_patterns = (
        "simulate_*liquidity*",
        "simulate_*funding*",
        "simulate_*deposit*",
        "simulate_*treasury*",
    )
    propose_patterns = (
        "propose_*funding*",
        "propose_*liquidity*",
        "propose_*treasury*",
    )

    def _metrics(
        self,
        task: TaskSpec,
        observations: dict[str, Any],
        simulation: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Cash totals and coverage signals from observations plus rehearsal metrics."""
        metrics: dict[str, float] = {}
        flat = flatten_numeric(observations)
        cash_total = sum_matched(flat, ("cash", "balance", "deposit"))
        if cash_total:
            metrics["observed_cash_total"] = round(cash_total, 4)
        ladder_total = sum_matched(flat, ("bucket", "ladder", "outflow", "inflow"))
        if ladder_total:
            metrics["observed_ladder_total"] = round(ladder_total, 4)
        observed_coverage = max_matched(flat, ("lcr", "coverage", "nsfr"))
        if observed_coverage is not None:
            metrics["observed_coverage_ratio"] = round(observed_coverage, 6)
        if simulation:
            for key, value in (simulation.get("metrics") or {}).items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metrics[f"sim_{key}"] = float(value)
        return metrics

    def _coverage_signal(self, metrics: dict[str, float]) -> float | None:
        """The most pessimistic coverage-style ratio visible in the metrics."""
        coverage = match_numeric(metrics, ("lcr", "coverage", "nsfr"))
        return min(coverage.values()) if coverage else None

    def _shortfall_signal(self, metrics: dict[str, float]) -> float:
        """Largest simulated/observed shortfall (0.0 when none reported)."""
        shortfalls = match_numeric(metrics, ("shortfall", "gap", "deficit"))
        return max(shortfalls.values()) if shortfalls else 0.0

    def _implied_tier(
        self,
        task: TaskSpec,
        metrics: dict[str, float],
        simulation: dict[str, Any] | None,
    ) -> RiskTier:
        """Threshold logic on coverage and shortfall: strained liquidity is high tier."""
        coverage = self._coverage_signal(metrics)
        if self._shortfall_signal(metrics) > 0.0:
            return RiskTier.high
        if coverage is not None:
            if coverage < COVERAGE_HIGH_THRESHOLD:
                return RiskTier.high
            if coverage < COVERAGE_MEDIUM_THRESHOLD:
                return RiskTier.medium
            return RiskTier.low
        # No coverage signal at all: cautious but not alarming.
        return RiskTier.medium if not metrics else RiskTier.low

    def _candidate_actions(
        self,
        task: TaskSpec,
        metrics: dict[str, float],
        simulation: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Deterministically ranked funding actions keyed off the coverage signal."""
        coverage = self._coverage_signal(metrics)
        shortfall = self._shortfall_signal(metrics)
        coverage_note = (
            f"worst coverage ratio {coverage:.3f}" if coverage is not None else "no coverage signal"
        )
        if shortfall > 0.0 or (coverage is not None and coverage < COVERAGE_HIGH_THRESHOLD):
            return [
                {
                    "action": "raise_short_term_funding",
                    "band": "propose",
                    "arguments": {"tenor_days": 30, "target_coverage": 1.2},
                    "score": 0.9,
                    "requires_approval": True,
                    "rationale": f"coverage breach: {coverage_note}, shortfall {shortfall:.2f}",
                },
                {
                    "action": "draw_committed_facility",
                    "band": "propose",
                    "arguments": {"fraction": 0.5},
                    "score": 0.7,
                    "requires_approval": True,
                    "rationale": "bridge the stress window while term funding is raised",
                },
                {
                    "action": "monitor_intraday_liquidity",
                    "band": "propose",
                    "arguments": {"frequency": "hourly"},
                    "score": 0.2,
                    "requires_approval": False,
                    "rationale": "insufficient on its own given the simulated breach",
                },
            ]
        if coverage is not None and coverage < COVERAGE_MEDIUM_THRESHOLD:
            return [
                {
                    "action": "rebalance_funding_mix",
                    "band": "propose",
                    "arguments": {"shift_to_term_fraction": 0.2},
                    "score": 0.7,
                    "requires_approval": False,
                    "rationale": f"thin headroom: {coverage_note}",
                },
                {
                    "action": "extend_term_funding",
                    "band": "propose",
                    "arguments": {"tenor_days": 90},
                    "score": 0.5,
                    "requires_approval": False,
                    "rationale": "lengthen the maturity profile before headroom erodes",
                },
            ]
        return [
            {
                "action": "monitor_liquidity_buffers",
                "band": "propose",
                "arguments": {"frequency": "daily"},
                "score": 0.8,
                "requires_approval": False,
                "rationale": f"buffers healthy ({coverage_note}); maintain monitoring",
            },
            {
                "action": "optimise_cash_deployment",
                "band": "propose",
                "arguments": {"max_tenor_days": 7},
                "score": 0.5,
                "requires_approval": False,
                "rationale": "deploy excess cash into short-tenor instruments",
            },
        ]
