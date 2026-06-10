"""Multi-class service-queue simulator for customer operations.

A continuous-time, event-driven M/M/c-style queue with three customer classes
(``vip`` > ``priority`` > ``standard``), exponential arrivals, service and patience
(abandonment) clocks, SLA tracking and post-handling escalations that keep the agent
busy for extra rework time. The two operational policy levers are exposed directly as
scenario parameters:

- **staffing** — ``n_agents`` (number of concurrently serving staff);
- **routing** — ``"priority"`` (highest class first, FIFO within class; default) or
  ``"fifo"`` (strict arrival order regardless of class).

Customers abandon when their patience expires before service starts; abandoned
contacts always count as SLA breaches. Arrivals stop at the horizon but the event
loop runs to completion so every customer is resolved (served or abandoned) and the
metrics are not censored.

Scenario params (defaults shown): ``arrival_rate_per_hr`` (60), ``horizon_hours``
(8.0), ``n_agents`` (8), ``sla_minutes`` (15.0), ``escalation_prob`` (0.08),
``routing`` ("priority"), ``classes`` (override dict of
``{name: {weight, priority, handle_mean, patience_mean}}``), ``n_reps`` (10),
``seed``.

Headline metrics: ``avg_handling_time`` (minutes, includes escalation rework),
``sla_breach_rate``, ``abandonment_rate`` and ``p90_wait`` (minutes).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from fintwinos.core.types import Scenario, SimulationResult
from fintwinos.twin_sim.base import (
    CalibratedSimulatorMixin,
    resolve_seed,
    spawn_generators,
    summarise_replications,
)

_DEFAULT_CLASSES: dict[str, dict[str, float]] = {
    # priority: lower = served first under the "priority" routing policy.
    "vip": {"weight": 0.10, "priority": 0, "handle_mean": 8.0, "patience_mean": 25.0},
    "priority": {"weight": 0.20, "priority": 1, "handle_mean": 7.0, "patience_mean": 18.0},
    "standard": {"weight": 0.70, "priority": 2, "handle_mean": 6.0, "patience_mean": 12.0},
}
_ESCALATION_REWORK_MEAN = 8.0
_QUEUE_SAMPLE_MINUTES = 15.0


@dataclass(frozen=True)
class _QueueParams:
    """Validated parameters for one queue run."""

    arrival_rate_per_hr: float
    horizon_minutes: float
    n_agents: int
    sla_minutes: float
    escalation_prob: float
    routing: str
    class_names: tuple[str, ...]
    weights: tuple[float, ...]
    priorities: tuple[int, ...]
    handle_means: tuple[float, ...]
    patience_means: tuple[float, ...]
    n_reps: int


def _parse_params(scenario: Scenario) -> tuple[_QueueParams, list[str]]:
    """Extract and validate queue parameters, accumulating warnings."""
    p = scenario.params
    warnings: list[str] = []
    routing = str(p.get("routing", "priority")).lower()
    if routing not in {"priority", "fifo"}:
        warnings.append(f"unknown routing policy '{routing}'; falling back to 'priority'")
        routing = "priority"

    classes_raw = p.get("classes", _DEFAULT_CLASSES)
    names: list[str] = []
    weights: list[float] = []
    priorities: list[int] = []
    handles: list[float] = []
    patiences: list[float] = []
    for name in sorted(classes_raw, key=lambda c: classes_raw[c].get("priority", 9)):
        cfg = classes_raw[name]
        names.append(str(name))
        weights.append(max(float(cfg.get("weight", 1.0)), 0.0))
        priorities.append(int(cfg.get("priority", len(names))))
        handles.append(max(float(cfg.get("handle_mean", 6.0)), 0.1))
        patiences.append(max(float(cfg.get("patience_mean", 12.0)), 0.1))
    total_w = sum(weights)
    if total_w <= 0:
        warnings.append("class weights sum to zero; using uniform weights")
        weights = [1.0] * len(names)
        total_w = float(len(names))
    weights = [w / total_w for w in weights]

    params = _QueueParams(
        arrival_rate_per_hr=max(float(p.get("arrival_rate_per_hr", 60.0)), 0.1),
        horizon_minutes=max(float(p.get("horizon_hours", 8.0)), 0.25) * 60.0,
        n_agents=max(int(p.get("n_agents", 8)), 1),
        sla_minutes=max(float(p.get("sla_minutes", 15.0)), 0.1),
        escalation_prob=min(max(float(p.get("escalation_prob", 0.08)), 0.0), 1.0),
        routing=routing,
        class_names=tuple(names),
        weights=tuple(weights),
        priorities=tuple(priorities),
        handle_means=tuple(handles),
        patience_means=tuple(patiences),
        n_reps=max(int(p.get("n_reps", 10)), 4),
    )
    return params, warnings


class ServiceQueueSimulator(CalibratedSimulatorMixin):
    """Multi-class M/M/c queue with priorities, abandonment and SLAs."""

    name = "customer_ops"

    def __init__(self) -> None:
        super().__init__()

    # -- reference ----------------------------------------------------------------------

    def _reference_sample(self) -> np.ndarray:
        """Deterministic reference distribution of wait times (minutes).

        A 70/30 mixture of short exponential waits and a long-wait tail, standing in
        for an observed contact-centre wait-time history.
        """
        rng = np.random.default_rng(20240621)
        short = rng.exponential(2.0, size=2800)
        long_tail = rng.exponential(15.0, size=1200)
        return np.concatenate([short, long_tail])

    # -- core dynamics ---------------------------------------------------------------------

    def _simulate_rep(
        self, rng: np.random.Generator, p: _QueueParams
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        """One replication of the event-driven queue.

        Returns:
            ``(metrics, queue_length_series, waits)`` where the series samples the
            number waiting every 15 simulated minutes across the horizon and
            ``waits`` is the per-customer wait sample (service start minus arrival
            for served customers; patience for abandoners) used for calibration.
        """
        rate_per_min = p.arrival_rate_per_hr / 60.0
        # Generate Poisson arrivals over the horizon.
        arrival_times: list[float] = []
        t = 0.0
        while True:
            t += float(rng.exponential(1.0 / rate_per_min))
            if t > p.horizon_minutes:
                break
            arrival_times.append(t)
        n = len(arrival_times)
        if n == 0:
            empty = np.zeros(int(p.horizon_minutes // _QUEUE_SAMPLE_MINUTES) + 1)
            return (
                {
                    "avg_handling_time": 0.0,
                    "sla_breach_rate": 0.0,
                    "abandonment_rate": 0.0,
                    "p90_wait": 0.0,
                },
                empty,
                np.zeros(0),
            )

        cls_idx = rng.choice(len(p.class_names), size=n, p=np.asarray(p.weights))
        patience = np.array(
            [float(rng.exponential(p.patience_means[c])) for c in cls_idx]
        )
        service = np.array([float(rng.exponential(p.handle_means[c])) for c in cls_idx])
        escalate = rng.uniform(size=n) < p.escalation_prob
        rework = np.where(escalate, rng.exponential(_ESCALATION_REWORK_MEAN, size=n), 0.0)
        handling = service + rework

        served = np.zeros(n, dtype=bool)
        abandoned = np.zeros(n, dtype=bool)
        waits = np.zeros(n)

        busy: list[float] = []  # min-heap of completion times
        waiting: list[list[int]] = [[] for _ in p.class_names]  # FIFO per class
        event_t: list[float] = [0.0]
        event_q: list[int] = [0]

        def queue_size() -> int:
            return sum(len(q) for q in waiting)

        def next_customer(now: float) -> int | None:
            """Pop the next live customer per routing policy, marking expired ones."""
            order = range(len(waiting))
            while True:
                candidate: int | None = None
                candidate_class: int | None = None
                if p.routing == "fifo":
                    best_arrival = np.inf
                    for ci in order:
                        if waiting[ci] and arrival_times[waiting[ci][0]] < best_arrival:
                            best_arrival = arrival_times[waiting[ci][0]]
                            candidate, candidate_class = waiting[ci][0], ci
                else:  # priority: classes are already sorted highest-priority first
                    for ci in order:
                        if waiting[ci]:
                            candidate, candidate_class = waiting[ci][0], ci
                            break
                if candidate is None:
                    return None
                assert candidate_class is not None
                waiting[candidate_class].pop(0)
                if arrival_times[candidate] + patience[candidate] <= now:
                    abandoned[candidate] = True
                    waits[candidate] = patience[candidate]
                    continue
                return candidate

        i = 0
        while i < n or busy:
            t_arrival = arrival_times[i] if i < n else np.inf
            t_completion = busy[0] if busy else np.inf
            if t_arrival <= t_completion:
                now = t_arrival
                if len(busy) < p.n_agents:
                    served[i] = True
                    waits[i] = 0.0
                    heapq.heappush(busy, now + handling[i])
                else:
                    waiting[int(cls_idx[i])].append(i)
                i += 1
            else:
                now = heapq.heappop(busy)
                cust = next_customer(now)
                if cust is not None:
                    served[cust] = True
                    waits[cust] = now - arrival_times[cust]
                    heapq.heappush(busy, now + handling[cust])
            event_t.append(now)
            event_q.append(queue_size())

        total = float(n)
        breaches = int(np.sum(served & (waits > p.sla_minutes))) + int(abandoned.sum())
        metrics = {
            "avg_handling_time": float(np.mean(handling[served])) if served.any() else 0.0,
            "sla_breach_rate": breaches / total,
            "abandonment_rate": float(abandoned.sum()) / total,
            "p90_wait": float(np.percentile(waits, 90.0)),
        }

        grid = np.arange(0.0, p.horizon_minutes + _QUEUE_SAMPLE_MINUTES, _QUEUE_SAMPLE_MINUTES)
        ev_t = np.asarray(event_t)
        ev_q = np.asarray(event_q, dtype=float)
        idx = np.clip(np.searchsorted(ev_t, grid, side="right") - 1, 0, len(ev_q) - 1)
        return metrics, ev_q[idx], waits

    # -- Simulator protocol -------------------------------------------------------------------

    def run(self, scenario: Scenario, *, seed: int | None = None) -> SimulationResult:
        """Run the service-queue simulation for ``scenario``.

        Args:
            scenario: Scenario whose ``params`` follow the module docstring.
            seed: Master seed; falls back to ``scenario.params['seed']`` then 7.

        Returns:
            A :class:`SimulationResult` with the four headline metrics (each with a
            bootstrap CI), a 15-minute ``queue_length`` series from the primary
            replication, and a calibration block over the pooled wait-time sample.
        """
        params, warnings = _parse_params(scenario)
        master_seed = resolve_seed(scenario, seed)
        rep_rngs, boot_rng = spawn_generators(master_seed, params.n_reps)

        per_rep: dict[str, list[float]] = {
            "avg_handling_time": [],
            "sla_breach_rate": [],
            "abandonment_rate": [],
            "p90_wait": [],
        }
        pooled_waits: list[np.ndarray] = []
        primary_queue: np.ndarray | None = None

        for i, rng in enumerate(rep_rngs):
            metrics_i, queue_series, waits = self._simulate_rep(rng, params)
            if i == 0:
                primary_queue = queue_series
            for k, v in metrics_i.items():
                per_rep[k].append(v)
            pooled_waits.append(waits)

        metrics, confidence = summarise_replications(per_rep, boot_rng)
        pooled = np.concatenate(pooled_waits) if pooled_waits else np.zeros(0)
        calibration = self._calibration_block(
            pooled, extra={"headline_sample": "wait_times_minutes"}
        )

        mean_handle_hr = float(np.mean(params.handle_means)) / 60.0
        offered_load = params.arrival_rate_per_hr * mean_handle_hr / params.n_agents
        if offered_load > 0.95:
            warnings.append(
                f"offered load {offered_load:.2f} per agent: the queue is at or beyond "
                "saturation; expect heavy SLA breaches and abandonment"
            )

        assert primary_queue is not None
        return SimulationResult(
            simulator=self.name,
            scenario_name=scenario.name,
            metrics=metrics,
            series={"queue_length": [float(v) for v in primary_queue]},
            confidence=confidence,
            calibration=calibration,
            warnings=warnings,
            seed=master_seed,
        )
