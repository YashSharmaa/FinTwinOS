"""Agent-based LOB-lite market simulator.

Three stylised agent populations interact through a single liquidity pool each step:

- **Market makers** quote around the mid and absorb the period's net order flow.
  They are inventory-averse: a growing (short) book skews their quotes down (up),
  producing mean reversion; in stressed regimes they post less depth, so the same
  flow moves price further.
- **Momentum traders** chase the trailing return with a saturating (tanh) demand
  curve, amplifying trends and fattening tails.
- **Noise traders** submit i.i.d. demand whose intensity scales with recent realised
  volatility (an ARCH-style feedback), generating volatility clustering.

The mid-price path follows ``log m_{t+1} = log m_t + r_t`` where ``r_t`` aggregates
flow impact, market-maker inventory skew, anchoring towards fundamental value, an
optional exogenous shock (``shock_bps`` at the shock step) and, in the stressed
regime, a jump mixture. The construction reproduces the classic stylised facts of
asset returns — heavy tails (kurtosis above the Gaussian 3) and positive lag-1
autocorrelation of absolute returns — which :meth:`AgentBasedMarketSimulator.stylised_facts`
checks explicitly and the test-suite asserts in the stressed regime.

Scenario params (all optional, defaults shown):

``n_steps`` (250), ``shock_bps`` (0.0), ``shock_step`` (n_steps // 3),
``regime`` ("calm" | "stressed"), ``n_agents`` (mix dict with keys
``market_makers``/``momentum``/``noise`` or an int total split 1:3:6),
``n_paths`` (16 Monte-Carlo replications), ``initial_price`` (100.0), ``seed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from fintwinos.core.types import Scenario, SimulationResult
from fintwinos.twin_sim.base import (
    CalibratedSimulatorMixin,
    resolve_seed,
    spawn_generators,
    summarise_replications,
)

_TRADING_DAYS = 252.0
_REGIMES: dict[str, dict[str, float]] = {
    # base_vol: per-step noise scale; arch_alpha: variance-feedback weight;
    # jump_prob/jump_scale: stress jump mixture; depth: relative MM depth;
    # momentum_gain: trend-chasing strength; reversion: pull to fundamental.
    "calm": {
        "base_vol": 0.008,
        "arch_alpha": 0.30,
        "jump_prob": 0.0,
        "jump_scale": 0.0,
        "depth": 1.0,
        "momentum_gain": 0.30,
        "reversion": 0.040,
    },
    "stressed": {
        "base_vol": 0.016,
        "arch_alpha": 0.55,
        "jump_prob": 0.035,
        "jump_scale": 4.0,
        "depth": 0.80,
        "momentum_gain": 0.60,
        "reversion": 0.030,
    },
}
#: Volatility cap (in multiples of the regime base vol) keeping the ARCH-style
#: feedback recursion bounded: liquidity withdrawal amplifies impact, which feeds
#: variance, so an explicit ceiling guarantees stationarity in every regime.
_VOL_CAP_MULTIPLE = 5.0
_DEFAULT_AGENT_MIX = {"market_makers": 4, "momentum": 12, "noise": 24}


@dataclass(frozen=True)
class _MarketParams:
    """Validated, fully-defaulted parameters for one market run."""

    n_steps: int
    shock_bps: float
    shock_step: int
    regime: str
    n_market_makers: int
    n_momentum: int
    n_noise: int
    n_paths: int
    initial_price: float


def _parse_params(scenario: Scenario) -> tuple[_MarketParams, list[str]]:
    """Extract and validate market parameters from a scenario, with warnings."""
    p = scenario.params
    warnings: list[str] = []
    n_steps = max(int(p.get("n_steps", 250)), 16)
    regime = str(p.get("regime", "calm")).lower()
    if regime not in _REGIMES:
        warnings.append(f"unknown regime '{regime}', falling back to 'calm'")
        regime = "calm"
    shock_bps = float(p.get("shock_bps", 0.0))
    shock_step = int(p.get("shock_step", n_steps // 3))
    if shock_bps != 0.0 and not (0 < shock_step < n_steps):
        warnings.append(
            f"shock_step {shock_step} outside (0, {n_steps}); shock moved to step {n_steps // 3}"
        )
        shock_step = n_steps // 3
    mix_raw = p.get("n_agents", dict(_DEFAULT_AGENT_MIX))
    if isinstance(mix_raw, dict):
        mix = {
            "market_makers": int(mix_raw.get("market_makers", _DEFAULT_AGENT_MIX["market_makers"])),
            "momentum": int(mix_raw.get("momentum", _DEFAULT_AGENT_MIX["momentum"])),
            "noise": int(mix_raw.get("noise", _DEFAULT_AGENT_MIX["noise"])),
        }
    else:
        total = max(int(mix_raw), 10)
        mix = {
            "market_makers": max(total // 10, 1),
            "momentum": max((3 * total) // 10, 1),
            "noise": max(total - total // 10 - (3 * total) // 10, 1),
        }
    for key in mix:
        if mix[key] < 1:
            warnings.append(f"n_agents.{key} < 1; clamped to 1")
            mix[key] = 1
    params = _MarketParams(
        n_steps=n_steps,
        shock_bps=shock_bps,
        shock_step=shock_step,
        regime=regime,
        n_market_makers=mix["market_makers"],
        n_momentum=mix["momentum"],
        n_noise=mix["noise"],
        n_paths=max(int(p.get("n_paths", 16)), 4),
        initial_price=float(p.get("initial_price", 100.0)),
    )
    return params, warnings


def _pearson_kurtosis(x: np.ndarray) -> float:
    """Raw (non-excess) kurtosis ``m4 / m2^2``; the Gaussian value is 3."""
    x = np.asarray(x, dtype=float)
    centred = x - x.mean()
    m2 = float(np.mean(centred**2))
    if m2 <= 0.0:
        return 3.0
    return float(np.mean(centred**4) / m2**2)


def _lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation; returns 0 for degenerate inputs."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    a, b = x[:-1], x[1:]
    sa, sb = a.std(), b.std()
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _max_drawdown(prices: np.ndarray) -> float:
    """Maximum peak-to-trough decline as a positive fraction of the peak."""
    peaks = np.maximum.accumulate(prices)
    drawdowns = 1.0 - prices / peaks
    return float(np.max(drawdowns))


def _expected_shortfall(returns: np.ndarray, level: float = 0.975) -> float:
    """Expected Shortfall (CVaR) of daily returns at ``level``, as a positive loss.

    Mean of the returns at or below the ``1 - level`` quantile (the loss tail),
    sign-flipped so larger values mean worse tail losses.
    """
    if returns.size == 0:
        return 0.0
    var_cut = float(np.quantile(returns, 1.0 - level))
    tail = returns[returns <= var_cut]
    if tail.size == 0:
        return -var_cut
    return float(-np.mean(tail))


class AgentBasedMarketSimulator(CalibratedSimulatorMixin):
    """Agent-based LOB-lite market simulator (Simulator protocol).

    See the module docstring for the price-formation mechanics. Each ``run`` performs
    ``n_paths`` independent replications, reports cross-path means as point estimates,
    percentile-bootstrap intervals (>= 200 draws) as confidence, and a calibration
    block with the KS distance between pooled standardised returns and a heavy-tailed
    reference distribution.
    """

    name = "market"

    def __init__(self) -> None:
        super().__init__()

    # -- reference -------------------------------------------------------------------

    def _reference_sample(self) -> np.ndarray:
        """Deterministic heavy-tailed reference for standardised daily returns.

        A Student-t(5) sample rescaled to unit variance, generated from a fixed seed —
        a stand-in for an empirical daily-return history until users fit their own via
        :meth:`fit_calibration`.
        """
        rng = np.random.default_rng(20240613)
        df = 5.0
        draws = rng.standard_t(df, size=4000)
        return draws / math.sqrt(df / (df - 2.0))

    # -- dynamics --------------------------------------------------------------------

    def _simulate_path(
        self, rng: np.random.Generator, p: _MarketParams
    ) -> tuple[np.ndarray, np.ndarray]:
        """One mid-price path; returns ``(prices[n_steps+1], returns[n_steps])``."""
        cfg = _REGIMES[p.regime]
        base_vol = cfg["base_vol"]
        base_var = base_vol**2
        alpha = cfg["arch_alpha"]
        depth_factor = cfg["depth"]
        gain = cfg["momentum_gain"]
        lam = cfg["reversion"]
        jump_prob = cfg["jump_prob"]
        jump_scale = cfg["jump_scale"]
        kappa = 0.05

        log_p0 = math.log(p.initial_price)
        log_m = np.empty(p.n_steps + 1)
        log_m[0] = log_p0
        returns = np.empty(p.n_steps)

        sqrt_noise = math.sqrt(p.n_noise)
        mm_scale = p.n_market_makers / _DEFAULT_AGENT_MIX["market_makers"]
        mom_scale = p.n_momentum / _DEFAULT_AGENT_MIX["momentum"]
        inventory = 0.0
        ewma_var = base_var
        window: list[float] = []

        # Pre-draw the per-step randomness in bulk for speed and reproducibility.
        noise_draws = rng.normal(0.0, 1.0, size=(p.n_steps, p.n_noise))
        jump_uniform = rng.uniform(size=p.n_steps)
        jump_draws = rng.normal(0.0, 1.0, size=p.n_steps)

        vol_cap = _VOL_CAP_MULTIPLE * base_vol
        for t in range(p.n_steps):
            sigma_t = min(math.sqrt((1.0 - alpha) * base_var + alpha * ewma_var), vol_cap)
            # Liquidity provided by market makers; thinner when volatility spikes
            # (liquidity withdrawal) and in the stressed regime.
            liquidity = max(mm_scale * depth_factor * sqrt_noise / max(sigma_t, 1e-9), 1e-9)

            noise_flow = float(noise_draws[t].sum())
            if window:
                signal = float(np.mean(window))
            else:
                signal = 0.0
            momentum_flow = gain * mom_scale * sqrt_noise * math.tanh(signal / base_vol)
            net_flow = noise_flow + momentum_flow

            impact_noise = noise_flow / liquidity
            impact_momentum = momentum_flow / liquidity
            # Market makers take the other side and lay residual inventory off
            # externally at ~10%/step, so the book mean-reverts instead of trending.
            inventory = 0.90 * inventory - net_flow
            skew = -kappa * inventory / liquidity
            anchor = lam * (log_p0 - log_m[t])

            r = impact_noise + impact_momentum + skew + anchor
            if jump_prob > 0.0 and jump_uniform[t] < jump_prob:
                r += jump_scale * sigma_t * float(jump_draws[t])
            if p.shock_bps != 0.0 and t == p.shock_step:
                r -= p.shock_bps / 1e4

            returns[t] = r
            log_m[t + 1] = log_m[t] + r
            # Feed only the noise-impact component back into the variance state and
            # clamp it: jumps, shocks and momentum drift raise realised vol but
            # cannot detonate the recursion (alpha/depth^2 stays below 1, and the
            # cap bounds the state in all regimes).
            feedback = min(impact_noise * impact_noise, vol_cap * vol_cap)
            ewma_var = (1.0 - alpha) * ewma_var + alpha * feedback
            window.append(r)
            if len(window) > 5:
                window.pop(0)

        return np.exp(log_m), returns

    @staticmethod
    def _shock_half_life(log_prices: np.ndarray, p: _MarketParams) -> float:
        """Steps until the post-shock deviation from the pre-shock level halves.

        Measured on the log-price path: ``dev_t = |log m_t - log m_pre|`` with
        ``m_pre`` the price just before the shock step. Capped at the number of
        remaining steps when the deviation never halves.
        """
        if p.shock_bps == 0.0:
            return 0.0
        s = p.shock_step
        pre = log_prices[s]  # log price entering the shock step
        dev0 = abs(log_prices[s + 1] - pre)
        if dev0 <= 1e-12:
            return 0.0
        remaining = len(log_prices) - (s + 1)
        for k in range(1, remaining):
            if abs(log_prices[s + 1 + k] - pre) <= dev0 / 2.0:
                return float(k)
        return float(remaining)

    # -- Simulator protocol ------------------------------------------------------------

    def run(self, scenario: Scenario, *, seed: int | None = None) -> SimulationResult:
        """Run the market simulation for ``scenario`` and return a calibrated result.

        Args:
            scenario: A :class:`~fintwinos.core.types.Scenario`; see the module
                docstring for recognised ``params``.
            seed: Master seed; falls back to ``scenario.params['seed']`` then 7.

        Returns:
            A :class:`~fintwinos.core.types.SimulationResult` with metrics
            ``realised_vol`` (annualised), ``max_drawdown``, ``expected_shortfall_97_5``
            (daily-return CVaR as a positive loss), ``kurtosis`` (Pearson),
            ``autocorr_abs_lag1``, ``shock_half_life`` and ``terminal_price``; the
            primary path's ``mid_price`` and ``returns`` series; bootstrap CIs for
            every metric; and a KS-based calibration block.
        """
        params, warnings = _parse_params(scenario)
        master_seed = resolve_seed(scenario, seed)
        rep_rngs, boot_rng = spawn_generators(master_seed, params.n_paths)

        per_rep: dict[str, list[float]] = {
            "realised_vol": [],
            "max_drawdown": [],
            "expected_shortfall_97_5": [],
            "kurtosis": [],
            "autocorr_abs_lag1": [],
            "shock_half_life": [],
            "terminal_price": [],
        }
        pooled_returns: list[np.ndarray] = []
        primary_prices: np.ndarray | None = None
        primary_returns: np.ndarray | None = None

        for i, rng in enumerate(rep_rngs):
            prices, rets = self._simulate_path(rng, params)
            if i == 0:
                primary_prices, primary_returns = prices, rets
            per_rep["realised_vol"].append(float(np.std(rets) * math.sqrt(_TRADING_DAYS)))
            per_rep["max_drawdown"].append(_max_drawdown(prices))
            per_rep["expected_shortfall_97_5"].append(_expected_shortfall(rets, 0.975))
            per_rep["kurtosis"].append(_pearson_kurtosis(rets))
            per_rep["autocorr_abs_lag1"].append(_lag1_autocorr(np.abs(rets)))
            per_rep["shock_half_life"].append(self._shock_half_life(np.log(prices), params))
            per_rep["terminal_price"].append(float(prices[-1]))
            pooled_returns.append(rets)

        metrics, confidence = summarise_replications(per_rep, boot_rng)

        pooled = np.concatenate(pooled_returns)
        pooled_std = float(np.std(pooled))
        standardised = pooled / pooled_std if pooled_std > 0 else pooled
        calibration = self._calibration_block(
            standardised,
            extra={"headline_sample": "standardised_returns", "regime": params.regime},
        )

        assert primary_prices is not None and primary_returns is not None
        series = {
            "mid_price": [float(v) for v in primary_prices],
            "returns": [float(v) for v in primary_returns],
        }
        if metrics["max_drawdown"] > 0.25:
            warnings.append(
                f"severe drawdown regime: mean max drawdown {metrics['max_drawdown']:.1%}"
            )
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

    # -- diagnostics ---------------------------------------------------------------------

    def stylised_facts(self, result: SimulationResult) -> dict[str, Any]:
        """Check the canonical stylised facts of asset returns on a run's metrics.

        Returns a dict of named boolean checks plus an aggregate ``all_pass``:

        - ``heavy_tails``: Pearson kurtosis above the Gaussian value of 3
          (expected to hold in the stressed regime).
        - ``volatility_clustering``: positive lag-1 autocorrelation of |returns|.
        - ``drawdown_in_unit_interval``: max drawdown is a sane fraction.
        - ``vol_positive``: realised volatility is strictly positive.
        """
        m = result.metrics
        checks = {
            "heavy_tails": bool(m.get("kurtosis", 0.0) > 3.0),
            "volatility_clustering": bool(m.get("autocorr_abs_lag1", -1.0) > 0.0),
            "drawdown_in_unit_interval": bool(0.0 <= m.get("max_drawdown", -1.0) <= 1.0),
            "vol_positive": bool(m.get("realised_vol", 0.0) > 0.0),
        }
        checks["all_pass"] = all(checks.values())
        return checks
