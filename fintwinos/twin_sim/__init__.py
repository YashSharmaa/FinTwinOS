"""twin_sim — calibrated simulators for the FinTwinOS federated twin.

Four domain simulators, all implementing the
:class:`~fintwinos.core.interfaces.Simulator` protocol (deterministic per seed,
confidence intervals from >= 200 bootstrap draws, a ``{"ks_stat", "calibrated"}``
calibration block on every result, plus ``calibration_report()``):

- :class:`~fintwinos.twin_sim.market.AgentBasedMarketSimulator` (``"market"``) —
  agent-based LOB-lite market with inventory-averse market makers, momentum chasers
  and noise traders.
- :class:`~fintwinos.twin_sim.treasury.LiquiditySimulator` (``"treasury"``) —
  multi-currency cash-ladder stress with counterbalancing-capacity drawdown.
- :class:`~fintwinos.twin_sim.compliance.FraudRingSimulator` (``"compliance"``) —
  transaction-graph growth with embedded laundering motifs and threshold-swept
  alerting.
- :class:`~fintwinos.twin_sim.customer_ops.ServiceQueueSimulator`
  (``"customer_ops"``) — multi-class M/M/c queue with priorities, abandonment, SLAs
  and staffing/routing policy levers.

Supporting pieces: :class:`~fintwinos.twin_sim.scenarios.ScenarioCatalog` (named
stress presets) and :class:`~fintwinos.twin_sim.calibration.PostHocCalibrator`
(quantile-mapping calibration, numpy-only KS and coverage diagnostics).

The contracted entry point is :func:`register_all`.

Part of FinTwinOS — created by Yash Sharma
(https://www.linkedin.com/in/yashsharmaa/), MIT License.
"""

from __future__ import annotations

from fintwinos.core.interfaces import TwinRuntime
from fintwinos.twin_sim.calibration import (
    PostHocCalibrator,
    bootstrap_ci,
    coverage_check,
    ks_statistic,
)
from fintwinos.twin_sim.compliance import FraudRingSimulator
from fintwinos.twin_sim.customer_ops import ServiceQueueSimulator
from fintwinos.twin_sim.market import AgentBasedMarketSimulator
from fintwinos.twin_sim.scenarios import ScenarioCatalog, default_catalog
from fintwinos.twin_sim.treasury import LiquiditySimulator

__all__ = [
    "AgentBasedMarketSimulator",
    "FraudRingSimulator",
    "LiquiditySimulator",
    "PostHocCalibrator",
    "ScenarioCatalog",
    "ServiceQueueSimulator",
    "bootstrap_ci",
    "coverage_check",
    "default_catalog",
    "ks_statistic",
    "register_all",
]


def register_all(runtime: TwinRuntime) -> None:
    """Instantiate the four twin_sim simulators and register them on the runtime.

    This is the contracted entry point consumed by tools, rl and demos (see
    CONTRACTS.md). It wires each simulator to the runtime where useful — the
    treasury simulator reads observed cash-ladder series from
    ``runtime.timeseries`` when present — registers them under their canonical
    names (``market``, ``treasury``, ``compliance``, ``customer_ops``) and appends
    a registration record to the shared audit trail.

    Idempotent: calling twice simply re-registers the same four names.

    Args:
        runtime: The assembled :class:`TwinRuntime` from
            ``fintwinos.twin_core.runtime.build_runtime``.
    """
    simulators = [
        AgentBasedMarketSimulator(),
        LiquiditySimulator(runtime=runtime),
        FraudRingSimulator(),
        ServiceQueueSimulator(),
    ]
    for simulator in simulators:
        runtime.register_simulator(simulator)
    runtime.audit.append(
        "twin_sim",
        "simulators.registered",
        {
            "simulators": [s.name for s in simulators],
            "count": len(simulators),
        },
    )
