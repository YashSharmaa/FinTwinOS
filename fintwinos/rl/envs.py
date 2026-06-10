"""Bounded, deterministic decision environments over twin dynamics.

Three discrete-action environments mirror the twin simulators' dynamics so RL
policies can be trained and shadow-tested entirely offline:

- :class:`AlertTriageEnv` — dismiss / investigate / escalate a stream of
  AML-style alerts; reward trades catches against analyst minutes, with a
  recall-floor penalty so a policy can never buy efficiency by silently
  dropping true positives.
- :class:`LiquidityActionEnv` — manage a funding-ladder cash buffer with
  do-nothing / draw-facility / sell-liquids / term-funding actions; reward is
  ``-funding_cost`` minus an expected-shortfall penalty whenever the buffer
  breaches its floor.
- :class:`QueueRoutingEnv` — choose a staffing level for an operations queue;
  reward is ``-SLA breaches - staffing cost``.

Every environment is *bounded* (rewards clipped to a declared
``reward_range``, episodes capped at ``max_steps``) and *deterministic*: the
same ``reset(seed)`` plus the same action sequence reproduces the trajectory
bit-for-bit, because each step consumes a fixed number of draws from a single
``numpy.random.default_rng(seed)`` stream regardless of the action chosen
(common random numbers — which also makes shadow A/B comparisons fair).

The factory functions look up calibration overrides from
``fintwinos.twin_sim`` *lazily and optionally*: when the twin_sim module (built
in parallel) is importable and exposes ``calibration_defaults(name)``, any
matching numeric parameters override the local defaults; otherwise the
self-contained dynamics below are used as-is. Environment construction never
fails because twin_sim is absent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fintwinos.rl.rewards import expected_shortfall

__all__ = [
    "ENV_NAMES",
    "AlertTriageEnv",
    "BoundedDiscreteEnv",
    "LiquidityActionEnv",
    "QueueRoutingEnv",
    "make_alert_triage_env",
    "make_env",
    "make_liquidity_action_env",
    "make_queue_routing_env",
]


class BoundedDiscreteEnv:
    """Base class: discrete actions, capped horizon, clipped rewards.

    Subclasses implement ``_initial_state()`` (set up internal state and
    return nothing) plus ``_transition(action) -> (raw_reward, done, info)``
    and ``_observe() -> np.ndarray``. The base class owns the rng lifecycle,
    step counting, action validation and reward clipping.
    """

    name: str = "bounded_env"
    action_names: tuple[str, ...] = ()
    state_dim: int = 0
    max_steps: int = 1
    reward_range: tuple[float, float] = (-1.0, 1.0)

    def __init__(self, seed: int = 7, max_steps: int | None = None, **params: float):
        self._seed = int(seed)
        if max_steps is not None:
            self.max_steps = int(max_steps)
        self.params: dict[str, float] = dict(params)
        self._rng = np.random.default_rng(self._seed)
        self._t = 0
        self._done = True  # require reset() before step()

    # -- public API --------------------------------------------------------------

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return len(self.action_names)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Re-seed and restart the episode; returns the initial observation."""
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._t = 0
        self._done = False
        self._initial_state()
        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply one action; returns ``(state, reward, done, info)``.

        Raises ``ValueError`` for out-of-range actions and ``RuntimeError``
        when called on a finished episode without an intervening ``reset``.
        """
        if self._done:
            raise RuntimeError(f"{self.name}: episode finished; call reset() first")
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(
                f"{self.name}: action {action} out of range [0, {self.n_actions})"
            )
        raw_reward, done, info = self._transition(action)
        self._t += 1
        if self._t >= self.max_steps:
            done = True
        self._done = done
        lo, hi = self.reward_range
        reward = float(np.clip(raw_reward, lo, hi))
        info = dict(info)
        info.update(
            {"raw_reward": float(raw_reward), "t": self._t, "action_name": self.action_names[action]}
        )
        return self._observe(), reward, done, info

    # -- hooks ---------------------------------------------------------------------

    def _initial_state(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _transition(self, action: int) -> tuple[float, bool, dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def _observe(self) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _param(self, key: str, default: float) -> float:
        """Calibrated parameter lookup with a local default."""
        return float(self.params.get(key, default))


class AlertTriageEnv(BoundedDiscreteEnv):
    """Triage a stream of alerts: dismiss, investigate or escalate.

    Each step presents one alert whose features are correlated with a hidden
    ``is_true_positive`` flag. Investigating catches a true positive with
    probability ``p_catch_investigate`` for 30 analyst minutes; escalating
    catches with ``p_catch_escalate`` for 90 minutes; dismissing costs 2
    minutes but forfeits the catch.

    Reward per step::

        catch_gain * caught - miss_cost * missed_tp - minutes / 60
        - recall_penalty   (when rolling recall < recall_floor after warm-up)

    Observation (8 dims, all roughly in [0, 1]): alert score, KYC risk,
    severity, normalised amount, prior-alert count, rolling recall, normalised
    minutes spent, episode progress.
    """

    name = "alert_triage"
    action_names = ("dismiss", "investigate", "escalate")
    state_dim = 8
    max_steps = 64
    reward_range = (-20.0, 20.0)

    def _initial_state(self) -> None:
        self._tp_seen = 0
        self._tp_caught = 0
        self._minutes = 0.0
        self._draw_alert()

    def _draw_alert(self) -> None:
        rng = self._rng
        tp_rate = self._param("true_positive_rate", 0.35)
        self._is_tp = bool(rng.random() < tp_rate)
        if self._is_tp:
            self._score = float(rng.beta(5.0, 2.0))
            self._kyc = float(rng.beta(4.0, 2.0))
        else:
            self._score = float(rng.beta(2.0, 5.0))
            self._kyc = float(rng.beta(2.0, 4.0))
        self._severity = float(rng.integers(0, 4)) / 3.0
        self._amount = float(np.clip(rng.normal(0.45 + 0.25 * self._is_tp, 0.15), 0.0, 1.0))
        self._priors = float(min(rng.poisson(1.0 + 2.0 * self._is_tp), 10)) / 10.0

    def _recall(self) -> float:
        return self._tp_caught / self._tp_seen if self._tp_seen else 1.0

    def _transition(self, action: int) -> tuple[float, bool, dict[str, Any]]:
        catch_draw = float(self._rng.random())  # always consumed: common random numbers
        minutes_by_action = (2.0, 30.0, 90.0)
        p_catch = (
            0.0,
            self._param("p_catch_investigate", 0.8),
            self._param("p_catch_escalate", 0.95),
        )[action]
        minutes = minutes_by_action[action]
        caught = self._is_tp and catch_draw < p_catch
        missed = self._is_tp and not caught

        if self._is_tp:
            self._tp_seen += 1
            if caught:
                self._tp_caught += 1
        self._minutes += minutes

        reward = (
            self._param("catch_gain", 10.0) * caught
            - self._param("miss_cost", 4.0) * missed
            - minutes / 60.0
        )
        recall_floor = self._param("recall_floor", 0.7)
        floor_breached = self._tp_seen >= 5 and self._recall() < recall_floor
        if floor_breached:
            reward -= self._param("recall_penalty", 3.0)

        info = {
            "is_true_positive": self._is_tp,
            "caught": bool(caught),
            "missed": bool(missed),
            "minutes": minutes,
            "recall": self._recall(),
            "recall_floor_breached": bool(floor_breached),
        }
        self._draw_alert()
        return reward, False, info

    def _observe(self) -> np.ndarray:
        return np.asarray(
            [
                self._score,
                self._kyc,
                self._severity,
                self._amount,
                self._priors,
                self._recall(),
                min(self._minutes / (90.0 * self.max_steps), 1.0),
                self._t / self.max_steps,
            ],
            dtype=np.float64,
        )


class LiquidityActionEnv(BoundedDiscreteEnv):
    """Manage a cash buffer against stochastic funding-ladder outflows.

    Actions: ``do_nothing``, ``draw_facility`` (raise 15 at the facility
    spread), ``sell_liquids`` (raise 12 from the HQLA pool at a haircut),
    ``term_funding`` (raise 20 at the term premium; durably damps the stress
    factor that drives future outflows).

    Reward per step is never positive::

        -funding_cost
        - es_weight * ES_alpha(projected outflows) / es_scale   (on breach)
        - insolvency_cost                                        (cash floored at 0)

    Observation (7 dims): cash buffer, facility headroom, HQLA pool, and the
    projected overnight / one-week / one-month cumulative outflow buckets (the
    ladder summary), plus episode progress — each normalised by a fixed scale.
    """

    name = "liquidity_action"
    action_names = ("do_nothing", "draw_facility", "sell_liquids", "term_funding")
    state_dim = 7
    max_steps = 32
    reward_range = (-40.0, 0.0)

    def _initial_state(self) -> None:
        self._cash = self._param("initial_cash", 60.0)
        self._headroom = self._param("initial_headroom", 50.0)
        self._hqla = self._param("initial_hqla", 80.0)
        self._stress = self._param("initial_stress", 1.0)

    def _ladder(self) -> tuple[float, float, float]:
        """Projected cumulative net outflows over o/n, 1w and 1m horizons."""
        base = self._param("base_outflow", 8.0) * self._stress
        return base, base * 5.0, base * 18.0

    def _transition(self, action: int) -> tuple[float, bool, dict[str, Any]]:
        rng = self._rng
        # Fixed draw schedule per step (common random numbers across policies).
        outflow_shock = float(rng.lognormal(mean=0.0, sigma=0.45))
        stress_shock = float(rng.normal(0.0, 0.08))
        es_samples = rng.lognormal(mean=0.0, sigma=0.45, size=64)

        base = self._param("base_outflow", 8.0)
        outflow = base * self._stress * outflow_shock

        raised, cost = 0.0, 0.0
        if action == 1:  # draw_facility
            draw = min(15.0, self._headroom)
            self._headroom -= draw
            raised = draw
            cost = draw * self._param("facility_spread", 0.02)
        elif action == 2:  # sell_liquids
            sale = min(12.0, self._hqla)
            self._hqla -= sale
            haircut = self._param("liquids_haircut", 0.03)
            raised = sale * (1.0 - haircut)
            cost = sale * haircut
        elif action == 3:  # term_funding
            raised = 20.0
            cost = 20.0 * self._param("term_premium", 0.04)
            self._stress = max(0.5, self._stress * 0.95)  # stable funding damps stress

        self._cash = self._cash + raised - outflow
        self._stress = float(np.clip(self._stress + stress_shock, 0.5, 2.5))

        floor = self._param("buffer_floor", 20.0)
        breach = self._cash < floor
        penalty = 0.0
        if breach:
            projected = base * self._stress * es_samples  # one-step-ahead outflow dist
            es = expected_shortfall(projected, alpha=0.95)
            penalty = self._param("es_weight", 2.0) * es / self._param("es_scale", 10.0)
        insolvency = self._cash <= 0.0
        if insolvency:
            penalty += self._param("insolvency_cost", 10.0)
            self._cash = 0.0

        reward = -cost - penalty
        info = {
            "outflow": outflow,
            "raised": raised,
            "funding_cost": cost,
            "breach": bool(breach),
            "insolvency": bool(insolvency),
            "cash": self._cash,
            "stress": self._stress,
        }
        return reward, False, info

    def _observe(self) -> np.ndarray:
        on, week, month = self._ladder()
        return np.asarray(
            [
                self._cash / 200.0,
                self._headroom / 50.0,
                self._hqla / 100.0,
                on / 20.0,
                week / 100.0,
                month / 400.0,
                self._t / self.max_steps,
            ],
            dtype=np.float64,
        )


class QueueRoutingEnv(BoundedDiscreteEnv):
    """Choose a staffing level for an operations queue under stochastic load.

    Actions index the staffing ladder ``(2, 4, 6, 8)`` agents. Arrivals are
    Poisson around a mean-reverting arrival rate; each agent serves
    ``service_rate`` items per step. Items queued beyond ``sla_buffer`` breach
    the SLA.

    Reward per step::

        -breach_weight * breaches - staff_cost * staff
    """

    name = "queue_routing"
    action_names = ("staff_2", "staff_4", "staff_6", "staff_8")
    state_dim = 6
    max_steps = 48
    reward_range = (-30.0, 0.0)
    staffing_levels = (2, 4, 6, 8)

    def _initial_state(self) -> None:
        self._queue = self._param("initial_queue", 6.0)
        self._rate = self._param("initial_rate", 12.0)
        self._breaches = 0.0

    def _transition(self, action: int) -> tuple[float, bool, dict[str, Any]]:
        rng = self._rng
        arrivals = float(rng.poisson(self._rate))
        rate_shock = float(rng.normal(0.0, 1.5))

        staff = self.staffing_levels[action]
        capacity = staff * self._param("service_rate", 3.0)
        served = min(self._queue + arrivals, capacity)
        self._queue = self._queue + arrivals - served

        sla_buffer = self._param("sla_buffer", 10.0)
        breaches = max(0.0, self._queue - sla_buffer)
        self._breaches += breaches

        # Mean-reverting arrival intensity.
        target = self._param("mean_rate", 12.0)
        self._rate = float(np.clip(self._rate + 0.25 * (target - self._rate) + rate_shock, 4.0, 25.0))

        reward = (
            -self._param("breach_weight", 0.3) * breaches
            - self._param("staff_cost", 0.5) * staff
        )
        info = {
            "arrivals": arrivals,
            "served": served,
            "queue": self._queue,
            "staff": staff,
            "breaches": breaches,
        }
        return reward, False, info

    def _observe(self) -> np.ndarray:
        return np.asarray(
            [
                min(self._queue / 50.0, 2.0),
                self._rate / 25.0,
                self._param("service_rate", 3.0) / 5.0,
                self._staff_fraction(),
                min(self._breaches / 100.0, 2.0),
                self._t / self.max_steps,
            ],
            dtype=np.float64,
        )

    def _staff_fraction(self) -> float:
        # The previous staffing choice is not state the policy controls between
        # steps in this formulation; expose the queue/SLA pressure ratio instead.
        sla_buffer = self._param("sla_buffer", 10.0)
        return float(min(self._queue / max(sla_buffer, 1.0), 2.0))


# -----------------------------------------------------------------------------
# Factories with optional twin_sim calibration
# -----------------------------------------------------------------------------

_SIMULATOR_NAMES = {
    "alert_triage": "alert_stream",
    "liquidity_action": "liquidity_ladder",
    "queue_routing": "ops_queue",
}


def _twin_sim_calibration(env_name: str) -> dict[str, float]:
    """Best-effort calibration lookup from ``fintwinos.twin_sim``.

    Imported lazily so that this module never fails when twin_sim is missing
    (it is built in parallel). Any exception — absent module, absent hook,
    incompatible return shape — degrades to the local defaults.
    """
    try:
        import fintwinos.twin_sim as twin_sim  # noqa: PLC0415 - deliberate lazy import
    except Exception:
        return {}
    hook = getattr(twin_sim, "calibration_defaults", None)
    if hook is None:
        return {}
    try:
        raw = hook(_SIMULATOR_NAMES.get(env_name, env_name))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _build(cls: type[BoundedDiscreteEnv], seed: int, overrides: dict[str, float]) -> Any:
    params = _twin_sim_calibration(cls.name)
    params.update({k: float(v) for k, v in overrides.items()})
    return cls(seed=seed, **params)


def make_alert_triage_env(seed: int = 7, **params: float) -> AlertTriageEnv:
    """Build an :class:`AlertTriageEnv`, merging twin_sim calibration if available."""
    return _build(AlertTriageEnv, seed, params)


def make_liquidity_action_env(seed: int = 7, **params: float) -> LiquidityActionEnv:
    """Build a :class:`LiquidityActionEnv`, merging twin_sim calibration if available."""
    return _build(LiquidityActionEnv, seed, params)


def make_queue_routing_env(seed: int = 7, **params: float) -> QueueRoutingEnv:
    """Build a :class:`QueueRoutingEnv`, merging twin_sim calibration if available."""
    return _build(QueueRoutingEnv, seed, params)


_FACTORIES = {
    "alert_triage": make_alert_triage_env,
    "liquidity_action": make_liquidity_action_env,
    "queue_routing": make_queue_routing_env,
}

#: Names accepted by :func:`make_env` (and the training CLI).
ENV_NAMES: tuple[str, ...] = tuple(sorted(_FACTORIES))


def make_env(name: str, seed: int = 7, **params: float) -> BoundedDiscreteEnv:
    """Build a bounded decision env by name.

    Args:
        name: One of :data:`ENV_NAMES`.
        seed: Determinism seed forwarded to the environment.
        **params: Calibration overrides (take precedence over twin_sim values).

    Raises:
        KeyError: For unknown environment names.
    """
    if name not in _FACTORIES:
        raise KeyError(f"unknown env '{name}'; available: {', '.join(ENV_NAMES)}")
    return _FACTORIES[name](seed=seed, **params)
