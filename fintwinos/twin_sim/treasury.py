"""Multi-currency liquidity / cash-ladder simulator.

Evolves a per-currency cash ladder over ``horizon_days``: every day each currency
books stochastic inflows and outflows around a structural baseline, outflows are
scaled by a stress multiplier, and any aggregate funding gap is met by drawing a
counterbalancing-capacity (CBC) stack in a fixed drawdown order (cash first, then
progressively less liquid, more heavily haircut tranches). Funding costs widen with
the spread shock (``shock_bps``) and quadratically with CBC utilisation, mimicking
the market's response to a visibly stressed treasury.

If the simulator is wired to a :class:`~fintwinos.core.interfaces.TwinRuntime`, it
reads observed cash-ladder series from ``runtime.timeseries`` (keys
``treasury.cash_ladder.<CCY>`` or ``cash_ladder.<CCY>``) and uses them as the
baseline daily net flows; otherwise it falls back to a deterministic synthetic
baseline derived from a stable per-currency hash, so behaviour never depends on
process-level hash salting.

Scenario params (defaults shown):

``horizon_days`` (30), ``currencies`` (["USD", "EUR", "GBP"]),
``stress_outflow_multiplier`` (1.0; may also be a per-currency dict),
``shock_bps`` (0.0 — funding spread shock), ``opening_balances`` (per-currency dict),
``cbc_tranches`` (list of {name, amount, haircut} in drawdown order),
``flow_noise`` (0.08 lognormal sigma), ``n_reps`` (16), ``seed``.

Headline metrics: ``min_cumulative_net_position`` (worst aggregate cumulative net
flow, base-currency millions), ``survival_days`` (days the institution can meet
outflows from balances plus CBC), ``lcr_proxy`` (HQLA over 30-day net stressed
outflows with the regulatory 75% inflow cap) and ``peak_funding_cost_bps``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import Scenario, SimulationResult
from fintwinos.twin_sim.base import (
    CalibratedSimulatorMixin,
    resolve_seed,
    spawn_generators,
    stable_int_hash,
    summarise_replications,
)

_DEFAULT_CURRENCIES = ["USD", "EUR", "GBP"]
_DEFAULT_OPENING = {"USD": 1000.0, "EUR": 700.0, "GBP": 500.0}
_DEFAULT_FX = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27}
_DEFAULT_CBC = [
    {"name": "central_bank_cash", "amount": 300.0, "haircut": 0.0},
    {"name": "hqla_level1", "amount": 500.0, "haircut": 0.05},
    {"name": "hqla_level2", "amount": 300.0, "haircut": 0.15},
    {"name": "committed_lines", "amount": 200.0, "haircut": 0.30},
]
_BASE_FUNDING_SPREAD_BPS = 40.0
_UTILISATION_WIDENING_BPS = 120.0
_LCR_INFLOW_CAP = 0.75


@dataclass(frozen=True)
class _TreasuryParams:
    """Validated parameters for one liquidity run."""

    horizon_days: int
    currencies: tuple[str, ...]
    multipliers: dict[str, float]
    shock_bps: float
    opening: dict[str, float]
    fx: dict[str, float]
    cbc: tuple[dict[str, float | str], ...]
    flow_noise: float
    n_reps: int


def _parse_params(scenario: Scenario) -> tuple[_TreasuryParams, list[str]]:
    """Extract and validate treasury parameters, accumulating warnings."""
    p = scenario.params
    warnings: list[str] = []
    horizon = max(int(p.get("horizon_days", 30)), 2)
    currencies = tuple(str(c).upper() for c in p.get("currencies", _DEFAULT_CURRENCIES))
    if not currencies:
        warnings.append("empty currency list; defaulting to USD")
        currencies = ("USD",)

    raw_mult = p.get("stress_outflow_multiplier", 1.0)
    if isinstance(raw_mult, dict):
        multipliers = {c: float(raw_mult.get(c, raw_mult.get(c.upper(), 1.0))) for c in currencies}
    else:
        multipliers = dict.fromkeys(currencies, float(raw_mult))
    for c, m in multipliers.items():
        if m < 0:
            warnings.append(f"negative stress multiplier for {c}; clamped to 0")
            multipliers[c] = 0.0

    opening_raw = p.get("opening_balances", {})
    opening = {
        c: float(opening_raw.get(c, _DEFAULT_OPENING.get(c, 500.0))) for c in currencies
    }
    fx_raw = p.get("fx_rates", {})
    fx = {c: float(fx_raw.get(c, _DEFAULT_FX.get(c, 1.0))) for c in currencies}

    cbc_raw = p.get("cbc_tranches", _DEFAULT_CBC)
    cbc: list[dict[str, float | str]] = []
    for tranche in cbc_raw:
        cbc.append(
            {
                "name": str(tranche.get("name", f"tranche_{len(cbc)}")),
                "amount": max(float(tranche.get("amount", 0.0)), 0.0),
                "haircut": min(max(float(tranche.get("haircut", 0.0)), 0.0), 1.0),
            }
        )

    params = _TreasuryParams(
        horizon_days=horizon,
        currencies=currencies,
        multipliers=multipliers,
        shock_bps=float(p.get("shock_bps", 0.0)),
        opening=opening,
        fx=fx,
        cbc=tuple(cbc),
        flow_noise=max(float(p.get("flow_noise", 0.08)), 0.0),
        n_reps=max(int(p.get("n_reps", 16)), 4),
    )
    return params, warnings


class LiquiditySimulator(CalibratedSimulatorMixin):
    """Multi-currency cash-ladder stress simulator (Simulator protocol).

    Args:
        runtime: Optional :class:`TwinRuntime`. When provided, observed cash-ladder
            series in ``runtime.timeseries`` override the synthetic flow baseline.
    """

    name = "treasury"

    def __init__(self, runtime: TwinRuntime | None = None) -> None:
        super().__init__()
        self.runtime = runtime

    # -- reference -------------------------------------------------------------------

    def _reference_sample(self) -> np.ndarray:
        """Deterministic reference distribution of daily net flows (base ccy, mm).

        Mildly left-skewed Gaussian mixture standing in for an observed history of
        daily net cash movements until users supply their own reference.
        """
        rng = np.random.default_rng(20240617)
        core = rng.normal(-6.0, 9.0, size=3200)
        tail = rng.normal(-30.0, 14.0, size=400)
        return np.concatenate([core, tail])

    # -- baseline flows ------------------------------------------------------------------

    def _runtime_ladder(self, ccy: str, horizon: int) -> np.ndarray | None:
        """Read an observed cash-ladder series for ``ccy`` from the runtime, if any.

        Accepts keys ``treasury.cash_ladder.<CCY>`` and ``cash_ladder.<CCY>``
        (case-insensitive currency suffix). Series shorter than the horizon are
        tiled; longer ones are truncated. Returns ``None`` when unavailable.
        """
        if self.runtime is None:
            return None
        try:
            keys = list(self.runtime.timeseries.keys())
        except Exception:
            return None
        for key in keys:
            for prefix in ("treasury.cash_ladder.", "cash_ladder."):
                if key.lower() == f"{prefix}{ccy}".lower():
                    try:
                        values = [float(v) for _, v in self.runtime.timeseries.window(key)]
                    except Exception:
                        return None
                    if not values:
                        return None
                    arr = np.asarray(values, dtype=float)
                    reps = int(np.ceil(horizon / arr.size))
                    return np.tile(arr, reps)[:horizon]
        return None

    @staticmethod
    def _synthetic_baseline(ccy: str, opening: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic synthetic baseline ``(inflows, outflows)`` for a currency.

        Shapes are derived from a stable sha256-based hash of the currency code, so
        the structural profile (weekly seasonality, level) is identical across runs
        and machines, independent of the run seed.
        """
        shape_rng = np.random.default_rng(stable_int_hash(f"fintwinos.treasury.{ccy}"))
        days = np.arange(horizon)
        outflow_base = 0.06 * opening
        inflow_base = 0.9 * outflow_base
        seasonality = 1.0 + 0.15 * np.sin(2.0 * np.pi * (days % 7) / 7.0 + shape_rng.uniform(0, np.pi))
        outflows = outflow_base * seasonality * shape_rng.uniform(0.9, 1.1, size=horizon)
        inflows = inflow_base * shape_rng.uniform(0.85, 1.15, size=horizon)
        return inflows, outflows

    # -- core dynamics ---------------------------------------------------------------------

    def _simulate_rep(
        self, rng: np.random.Generator, p: _TreasuryParams
    ) -> tuple[dict[str, float], dict[str, np.ndarray], np.ndarray]:
        """One Monte-Carlo replication.

        Returns:
            ``(metrics, series, net_flows)`` where ``series`` holds per-currency
            cumulative net positions plus aggregate ladders, and ``net_flows`` is the
            pooled daily net-flow sample used for calibration.
        """
        horizon = p.horizon_days
        total_net = np.zeros(horizon)
        gross_out = np.zeros(horizon)
        gross_in = np.zeros(horizon)
        series: dict[str, np.ndarray] = {}

        for ccy in p.currencies:
            opening = p.opening[ccy]
            fx = p.fx[ccy]
            ladder = self._runtime_ladder(ccy, horizon)
            if ladder is not None:
                # Observed ladder = net flows; split into synthetic gross legs around it.
                inflows = np.clip(ladder, 0.0, None) + 0.02 * opening
                outflows = inflows - ladder
            else:
                inflows, outflows = self._synthetic_baseline(ccy, opening, horizon)
            mult = p.multipliers[ccy]
            noise_out = rng.lognormal(mean=0.0, sigma=p.flow_noise, size=horizon)
            noise_in = rng.lognormal(mean=0.0, sigma=p.flow_noise, size=horizon)
            out_stressed = outflows * mult * noise_out
            in_stressed = inflows * noise_in
            net = (in_stressed - out_stressed) * fx
            total_net += net
            gross_out += out_stressed * fx
            gross_in += in_stressed * fx
            series[f"cum_net_{ccy.lower()}"] = np.cumsum(in_stressed - out_stressed)

        cum_net = np.cumsum(total_net)
        opening_total = sum(p.opening[c] * p.fx[c] for c in p.currencies)
        cbc_amounts = np.array([float(t["amount"]) * (1.0 - float(t["haircut"])) for t in p.cbc])
        cbc_total = float(cbc_amounts.sum())

        # Walk the ladder: draw CBC tranches in order whenever balances go negative.
        cbc_remaining = np.empty(horizon)
        funding_cost = np.empty(horizon)
        survival = horizon
        breached = False
        for d in range(horizon):
            position = opening_total + cum_net[d]
            shortfall = max(-position, 0.0)
            drawn = min(shortfall, cbc_total)
            cbc_remaining[d] = cbc_total - drawn
            utilisation = drawn / cbc_total if cbc_total > 0 else (1.0 if shortfall > 0 else 0.0)
            funding_cost[d] = (
                _BASE_FUNDING_SPREAD_BPS
                + p.shock_bps
                + _UTILISATION_WIDENING_BPS * utilisation**2
            )
            if not breached and shortfall > cbc_total:
                survival = d  # survived d full days before resources ran out
                breached = True

        # LCR proxy: HQLA / 30-day net stressed outflows, inflows capped at 75%.
        window = min(horizon, 30)
        capped_in = np.minimum(gross_in[:window], _LCR_INFLOW_CAP * gross_out[:window])
        net_outflow_30d = float(np.sum(gross_out[:window] - capped_in))
        hqla = opening_total + cbc_total
        lcr_proxy = hqla / net_outflow_30d if net_outflow_30d > 1e-9 else float("inf")

        metrics = {
            "min_cumulative_net_position": float(np.min(cum_net)),
            "survival_days": float(survival),
            "lcr_proxy": float(min(lcr_proxy, 99.0)),
            "peak_funding_cost_bps": float(np.max(funding_cost)),
        }
        series["total_cum_net"] = cum_net
        series["cbc_remaining"] = cbc_remaining
        series["funding_cost_bps"] = funding_cost
        return metrics, series, total_net

    # -- Simulator protocol -------------------------------------------------------------

    def run(self, scenario: Scenario, *, seed: int | None = None) -> SimulationResult:
        """Run the liquidity stress simulation for ``scenario``.

        Args:
            scenario: Scenario whose ``params`` follow the module docstring.
            seed: Master seed; falls back to ``scenario.params['seed']`` then 7.

        Returns:
            A :class:`SimulationResult` with the four headline metrics (each with a
            bootstrap CI), per-currency cumulative-net series from the primary
            replication, aggregate ladder/CBC/funding-cost series, and a calibration
            block over the pooled daily net flows.
        """
        params, warnings = _parse_params(scenario)
        master_seed = resolve_seed(scenario, seed)
        rep_rngs, boot_rng = spawn_generators(master_seed, params.n_reps)

        per_rep: dict[str, list[float]] = {
            "min_cumulative_net_position": [],
            "survival_days": [],
            "lcr_proxy": [],
            "peak_funding_cost_bps": [],
        }
        pooled_flows: list[np.ndarray] = []
        primary_series: dict[str, np.ndarray] | None = None

        for i, rng in enumerate(rep_rngs):
            metrics_i, series_i, flows_i = self._simulate_rep(rng, params)
            if i == 0:
                primary_series = series_i
            for k, v in metrics_i.items():
                per_rep[k].append(v)
            pooled_flows.append(flows_i)

        metrics, confidence = summarise_replications(per_rep, boot_rng)
        calibration = self._calibration_block(
            np.concatenate(pooled_flows),
            extra={"headline_sample": "daily_net_flows_base_ccy"},
        )

        if metrics["survival_days"] < params.horizon_days:
            warnings.append(
                "counterbalancing capacity exhausted before the horizon: mean survival "
                f"{metrics['survival_days']:.1f} of {params.horizon_days} days"
            )
        if metrics["lcr_proxy"] < 1.0:
            warnings.append(f"LCR proxy below 1.0 ({metrics['lcr_proxy']:.2f}): HQLA shortfall")

        assert primary_series is not None
        series = {k: [float(v) for v in arr] for k, arr in primary_series.items()}
        return SimulationResult(
            simulator=self.name,
            scenario_name=scenario.name,
            metrics=metrics,
            series=series,
            confidence=confidence,
            calibration=calibration,
            warnings=warnings,
            seed=master_seed,
        )
