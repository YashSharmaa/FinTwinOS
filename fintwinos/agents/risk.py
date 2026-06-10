"""The market/credit risk desk agent: exposures, VaR, shocks and hedging.

Observes positions and exposures through the registry, rehearses a market shock,
then ranks hedging actions with deterministic loss-ratio thresholds: the simulated
shock loss (or VaR) relative to gross exposure decides whether the desk holds,
partially hedges, or recommends a full hedge with human approval. Any explicit
breach flag in the rehearsal metrics escalates immediately.
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

#: Simulated loss above this fraction of gross exposure is a high-tier event.
LOSS_RATIO_HIGH = 0.08
#: Simulated loss above this fraction of gross exposure warrants action (medium).
LOSS_RATIO_MEDIUM = 0.03
#: VaR above this fraction of gross exposure is a high-tier event.
VAR_RATIO_HIGH = 0.12


class RiskAgent(DomainAgent):
    """Hedge / exposure / VaR / shock domain owner."""

    name = "risk"
    domain = "risk"
    observe_patterns = (
        "observe_*position*",
        "observe_*exposure*",
        "observe_*market*",
        "observe_*portfolio*",
        "observe_*risk*",
    )
    simulate_patterns = (
        "simulate_*shock*",
        "simulate_*stress*",
        "simulate_*var*",
        "simulate_*market*",
        "simulate_*risk*",
    )
    propose_patterns = (
        "propose_*hedge*",
        "propose_*risk*",
        "propose_*exposure*",
    )

    def _metrics(
        self,
        task: TaskSpec,
        observations: dict[str, Any],
        simulation: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Exposure aggregates from observations plus rehearsal loss metrics."""
        metrics: dict[str, float] = {}
        flat = flatten_numeric(observations)
        gross = max_matched(flat, ("gross_exposure", "gross", "notional"))
        if gross is not None:
            metrics["gross_exposure"] = round(gross, 4)
        net = max_matched(flat, ("net_exposure", "net"))
        if net is not None:
            metrics["net_exposure"] = round(net, 4)
        position_mass = sum(match_numeric(flat, ("position", "quantity")).values())
        if position_mass:
            metrics["observed_position_mass"] = round(position_mass, 4)
        if simulation:
            for key, value in (simulation.get("metrics") or {}).items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metrics[f"sim_{key}"] = float(value)
        loss_ratio = self._loss_ratio(metrics)
        if loss_ratio is not None:
            metrics["loss_to_gross_ratio"] = round(loss_ratio, 6)
        return metrics

    def _loss_ratio(self, metrics: dict[str, float]) -> float | None:
        """Simulated shock loss as a fraction of gross exposure (None when unknown)."""
        losses = match_numeric(metrics, ("loss", "drawdown", "shortfall"))
        if not losses:
            return None
        gross = metrics.get("gross_exposure")
        denominator = gross if gross and gross > 0 else None
        if denominator is None:
            return None
        return abs(max(losses.values(), key=abs)) / denominator

    def _var_ratio(self, metrics: dict[str, float]) -> float | None:
        """Simulated VaR as a fraction of gross exposure (None when unknown)."""
        var_values = match_numeric(metrics, ("var", "cvar", "expected_shortfall"))
        gross = metrics.get("gross_exposure")
        if not var_values or not gross or gross <= 0:
            return None
        return abs(max(var_values.values(), key=abs)) / gross

    def _implied_tier(
        self,
        task: TaskSpec,
        metrics: dict[str, float],
        simulation: dict[str, Any] | None,
    ) -> RiskTier:
        """Loss-ratio thresholds; explicit breach flags escalate unconditionally."""
        breaches = match_numeric(metrics, ("breach", "limit_exceeded"))
        if any(v != 0.0 for v in breaches.values()):
            return RiskTier.high
        loss_ratio = self._loss_ratio(metrics)
        var_ratio = self._var_ratio(metrics)
        if (loss_ratio is not None and loss_ratio > LOSS_RATIO_HIGH) or (
            var_ratio is not None and var_ratio > VAR_RATIO_HIGH
        ):
            return RiskTier.high
        if loss_ratio is not None and loss_ratio > LOSS_RATIO_MEDIUM:
            return RiskTier.medium
        if loss_ratio is None and var_ratio is None and not metrics:
            return RiskTier.medium  # flying blind is itself a (moderate) risk
        return RiskTier.low

    def _candidate_actions(
        self,
        task: TaskSpec,
        metrics: dict[str, float],
        simulation: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Hedge / reduce / hold ladder ranked by the simulated loss ratio."""
        loss_ratio = self._loss_ratio(metrics)
        var_ratio = self._var_ratio(metrics)
        signal = (
            f"loss ratio {loss_ratio:.4f}" if loss_ratio is not None
            else f"VaR ratio {var_ratio:.4f}" if var_ratio is not None
            else "no rehearsal loss signal"
        )
        severe = (loss_ratio is not None and loss_ratio > LOSS_RATIO_HIGH) or (
            var_ratio is not None and var_ratio > VAR_RATIO_HIGH
        )
        moderate = loss_ratio is not None and loss_ratio > LOSS_RATIO_MEDIUM
        if severe:
            return [
                {
                    "action": "hedge_with_index_futures",
                    "band": "propose",
                    "arguments": {"hedge_ratio": 0.6, "instrument_class": "index_future"},
                    "score": 0.9,
                    "requires_approval": True,
                    "rationale": f"simulated stress is material: {signal}",
                },
                {
                    "action": "reduce_gross_exposure",
                    "band": "propose",
                    "arguments": {"reduction_fraction": 0.25},
                    "score": 0.7,
                    "requires_approval": True,
                    "rationale": "de-gross to cut tail loss if hedging capacity is limited",
                },
                {
                    "action": "hold_position",
                    "band": "propose",
                    "arguments": {},
                    "score": 0.1,
                    "requires_approval": False,
                    "rationale": f"not defensible given {signal}",
                },
            ]
        if moderate:
            return [
                {
                    "action": "partial_hedge",
                    "band": "propose",
                    "arguments": {"hedge_ratio": 0.3, "instrument_class": "index_future"},
                    "score": 0.7,
                    "requires_approval": False,
                    "rationale": f"trim the tail at moderate stress: {signal}",
                },
                {
                    "action": "hold_and_monitor",
                    "band": "propose",
                    "arguments": {"review_in_days": 5},
                    "score": 0.5,
                    "requires_approval": False,
                    "rationale": "acceptable if volatility mean-reverts; re-test in 5 days",
                },
            ]
        return [
            {
                "action": "hold_position",
                "band": "propose",
                "arguments": {},
                "score": 0.8,
                "requires_approval": False,
                "rationale": f"stress within appetite ({signal}); no hedge needed",
            },
            {
                "action": "review_risk_limits",
                "band": "propose",
                "arguments": {"scope": "desk"},
                "score": 0.4,
                "requires_approval": False,
                "rationale": "routine confirmation that limits match current book size",
            },
        ]
