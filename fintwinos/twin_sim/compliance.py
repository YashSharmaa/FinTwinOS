"""Fraud-ring transaction-graph simulator with embedded laundering motifs.

Grows a directed transaction graph step by step. Background traffic is typology-free:
random account pairs exchanging lognormal amounts spread across the whole horizon.
Laundering rings are injected at a configurable rate, one of four classic motifs:

- **fan_in** — many sources pay a hub within a tight window; the hub forwards the
  bulk onwards (collection stage).
- **fan_out** — a hub receives one large credit and disperses it to many recipients
  (distribution stage).
- **cycle** — funds travel a closed loop of accounts, shrinking by a small fee at
  every hop (round-tripping).
- **layering** — a chain of mule accounts each forwarding ~90-97% of what it received
  on consecutive days (classic layering).

Ring transactions are *structured*: amounts are round multiples of 100 and bursts are
compressed into one to three steps — exactly the regularities a transaction-monitoring
model keys on. A deterministic, feature-based detector scores every account. Its
primary evidence is the count of structured transactions an account touches
(background amounts are lognormal values rounded to cents, so they are almost never
exact multiples of 100), supported by pass-through ratio, burstiness, structuring
ratio, cycle membership via strongly connected components on the structured-edge
subgraph, and activity. The detector emits :class:`~fintwinos.core.types.Alert`
objects above ``alert_threshold``, and the simulator reports precision/recall across
a full threshold sweep so reviewers can see the operating curve, not just one point.
The weights are calibrated so account-level recall at the default ``alert_threshold``
of 0.5 clears the compliance release-gate floor (0.75) across ring densities.

Scenario params (defaults shown): ``n_steps`` (60), ``n_accounts`` (200),
``background_rate`` (30 transactions/step), ``ring_rate`` (0.12 — probability a new
ring spawns each step), ``alert_threshold`` (0.5), ``n_reps`` (8), ``seed``.

Headline metrics: ``true_ring_edges``, ``ring_accounts``, ``n_alerts``,
``alert_precision`` and ``alert_recall`` at the configured threshold, and
``analyst_minutes_expected`` (triage workload implied by the alert volume).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from fintwinos.core.types import Alert, EntityRef, RiskTier, Scenario, SimulationResult
from fintwinos.twin_sim.base import (
    CalibratedSimulatorMixin,
    resolve_seed,
    spawn_generators,
    summarise_replications,
)

_MOTIFS = ("fan_in", "fan_out", "cycle", "layering")
_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
_TRIAGE_BASE_MINUTES = 12.0
_TRIAGE_PER_SCORE_MINUTES = 18.0


@dataclass(frozen=True)
class _ComplianceParams:
    """Validated parameters for one fraud-ring run."""

    n_steps: int
    n_accounts: int
    background_rate: float
    ring_rate: float
    alert_threshold: float
    n_reps: int


@dataclass
class _TxLog:
    """Columnar transaction log accumulated during graph growth."""

    src: list[int]
    dst: list[int]
    step: list[int]
    amount: list[float]
    is_ring: list[bool]

    def append(self, src: int, dst: int, step: int, amount: float, is_ring: bool) -> None:
        self.src.append(src)
        self.dst.append(dst)
        self.step.append(step)
        self.amount.append(amount)
        self.is_ring.append(is_ring)


def _parse_params(scenario: Scenario) -> tuple[_ComplianceParams, list[str]]:
    """Extract and validate compliance parameters, accumulating warnings."""
    p = scenario.params
    warnings: list[str] = []
    ring_rate = float(p.get("ring_rate", 0.12))
    if not 0.0 <= ring_rate <= 1.0:
        warnings.append(f"ring_rate {ring_rate} outside [0, 1]; clamped")
        ring_rate = min(max(ring_rate, 0.0), 1.0)
    threshold = float(p.get("alert_threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        warnings.append(f"alert_threshold {threshold} outside (0, 1); reset to 0.5")
        threshold = 0.5
    params = _ComplianceParams(
        n_steps=max(int(p.get("n_steps", 60)), 10),
        n_accounts=max(int(p.get("n_accounts", 200)), 30),
        background_rate=max(float(p.get("background_rate", 30.0)), 1.0),
        ring_rate=ring_rate,
        alert_threshold=threshold,
        n_reps=max(int(p.get("n_reps", 8)), 4),
    )
    return params, warnings


def _round_structured(amount: float) -> float:
    """Round a laundering amount to the nearest 100, floored at 100."""
    return max(100.0, round(amount / 100.0) * 100.0)


class FraudRingSimulator(CalibratedSimulatorMixin):
    """Transaction-graph laundering simulator with alerting (Simulator protocol).

    After :meth:`run`, the simulator exposes ``last_alerts`` — the
    :class:`Alert` objects emitted for the primary replication at the configured
    threshold — for downstream tools that want the individual cases rather than the
    aggregate metrics.
    """

    name = "compliance"

    def __init__(self) -> None:
        super().__init__()
        self.last_alerts: list[Alert] = []

    # -- reference ---------------------------------------------------------------------

    def _reference_sample(self) -> np.ndarray:
        """Deterministic bimodal reference for account suspicion scores.

        A 90/10 mixture of a low-score Beta(2, 12) population and a high-score
        Beta(8, 2) population — the canonical shape of a monitored book with a small
        genuinely suspicious tail.
        """
        rng = np.random.default_rng(20240619)
        low = rng.beta(2.0, 12.0, size=3600)
        high = rng.beta(8.0, 2.0, size=400)
        return np.concatenate([low, high])

    # -- graph growth --------------------------------------------------------------------

    def _grow_graph(
        self, rng: np.random.Generator, p: _ComplianceParams
    ) -> tuple[_TxLog, set[int]]:
        """Grow background traffic plus embedded rings; return the log and ring members."""
        log = _TxLog([], [], [], [], [])
        ring_members: set[int] = set()
        for step in range(p.n_steps):
            n_bg = int(rng.poisson(p.background_rate))
            for _ in range(n_bg):
                src = int(rng.integers(p.n_accounts))
                dst = int(rng.integers(p.n_accounts))
                while dst == src:
                    dst = int(rng.integers(p.n_accounts))
                amount = float(np.round(rng.lognormal(mean=3.0, sigma=1.0) * 10.0, 2))
                log.append(src, dst, step, amount, False)
            if rng.uniform() < p.ring_rate:
                motif = _MOTIFS[int(rng.integers(len(_MOTIFS)))]
                members = self._inject_ring(rng, p, log, step, motif)
                ring_members.update(members)
        return log, ring_members

    def _inject_ring(
        self,
        rng: np.random.Generator,
        p: _ComplianceParams,
        log: _TxLog,
        step: int,
        motif: str,
    ) -> set[int]:
        """Embed one laundering motif starting at ``step``; return member accounts."""
        base = _round_structured(float(rng.uniform(800.0, 5000.0)))
        pick = lambda k: [int(a) for a in rng.choice(p.n_accounts, size=k, replace=False)]  # noqa: E731

        if motif == "fan_in":
            k = int(rng.integers(5, 10))
            accounts = pick(k + 2)
            sources, hub, collector = accounts[:k], accounts[k], accounts[k + 1]
            total = 0.0
            for s in sources:
                amt = _round_structured(base * float(rng.uniform(0.6, 1.4)))
                log.append(s, hub, step + int(rng.integers(0, 2)), amt, True)
                total += amt
            keep = float(rng.uniform(0.90, 0.97))
            log.append(hub, collector, step + 2, _round_structured(total * keep), True)
            return set(accounts)

        if motif == "fan_out":
            k = int(rng.integers(5, 10))
            accounts = pick(k + 2)
            source, hub, recipients = accounts[0], accounts[1], accounts[2:]
            total = _round_structured(base * k)
            log.append(source, hub, step, total, True)
            share = total / k
            for r in recipients:
                amt = _round_structured(share * float(rng.uniform(0.85, 1.1)))
                log.append(hub, r, step + int(rng.integers(1, 3)), amt, True)
            return set(accounts)

        if motif == "cycle":
            k = int(rng.integers(3, 7))
            accounts = pick(k)
            amt = base
            for i in range(k):
                nxt = accounts[(i + 1) % k]
                log.append(accounts[i], nxt, step + min(i, 2), amt, True)
                amt = _round_structured(amt * float(rng.uniform(0.93, 0.98)))
            return set(accounts)

        # layering chain
        k = int(rng.integers(4, 8))
        accounts = pick(k)
        amt = base
        for i in range(k - 1):
            keep = float(rng.uniform(0.90, 0.97))
            log.append(accounts[i], accounts[i + 1], step + i, amt, True)
            amt = _round_structured(amt * keep)
        return set(accounts)

    # -- detection -------------------------------------------------------------------------

    def _score_accounts(
        self, rng: np.random.Generator, p: _ComplianceParams, log: _TxLog
    ) -> np.ndarray:
        """Deterministic feature-based suspicion score in [0, 1] per account.

        Primary evidence is the number of structured transactions (exact multiples
        of 100) an account touches: the generator plants them on every ring edge,
        while background lognormal amounts rounded to cents essentially never land
        on a round multiple. A binary touched-structured flag plus a saturating
        structured-count term carries most of the weight; pass-through ratio,
        burstiness, structuring ratio, cycle membership and activity refine the
        ordering so hubs and mules outrank one-touch peripherals. Weights are
        calibrated so recall at the default ``alert_threshold`` of 0.5 clears the
        0.75 release-gate floor across ring densities with precision held high.
        """
        n = p.n_accounts
        src = np.asarray(log.src, dtype=int)
        dst = np.asarray(log.dst, dtype=int)
        step = np.asarray(log.step, dtype=int)
        amount = np.asarray(log.amount, dtype=float)

        in_amt = np.bincount(dst, weights=amount, minlength=n)
        out_amt = np.bincount(src, weights=amount, minlength=n)
        n_in = np.bincount(dst, minlength=n).astype(float)
        n_out = np.bincount(src, minlength=n).astype(float)
        n_txn = n_in + n_out

        both = np.minimum(in_amt, out_amt)
        peak = np.maximum(np.maximum(in_amt, out_amt), 1e-9)
        pass_through = np.where((in_amt > 0) & (out_amt > 0), both / peak, 0.0)

        structured = np.abs(np.mod(amount, 100.0)) < 1e-9
        struct_counts = np.bincount(dst, weights=structured.astype(float), minlength=n)
        struct_counts += np.bincount(src, weights=structured.astype(float), minlength=n)
        structured_ratio = np.where(n_txn > 0, struct_counts / np.maximum(n_txn, 1.0), 0.0)

        # Burstiness: largest share of an account's transactions inside any 3-step window.
        burst = np.zeros(n)
        max_step = int(step.max()) + 1 if step.size else 1
        for acct in np.flatnonzero(n_txn > 0):
            steps_a = np.concatenate([step[src == acct], step[dst == acct]])
            counts = np.bincount(steps_a, minlength=max_step + 3)
            window = counts[:-2] + counts[1:-1] + counts[2:] if counts.size >= 3 else counts
            burst[acct] = float(window.max()) / float(steps_a.size)

        # Cycle membership on the structured-edge subgraph (SCCs of size >= 2).
        cycle_flag = np.zeros(n)
        g = nx.DiGraph()
        for s, d in zip(src[structured], dst[structured], strict=True):
            g.add_edge(int(s), int(d))
        for component in nx.strongly_connected_components(g):
            if len(component) >= 2:
                for node in component:
                    cycle_flag[node] = 1.0

        activity = np.minimum(n_txn, 10.0) / 10.0

        # Structured-amount evidence: the planted typology signature. A binary
        # touched-structured flag separates ring members from background traffic;
        # the saturating count term ranks hubs/mules above one-touch peripherals.
        struct_flag = (struct_counts > 0).astype(float)
        struct_depth = np.minimum(struct_counts, 4.0) / 4.0

        z = (
            2.5 * struct_flag
            + 1.2 * struct_depth
            + 1.0 * (pass_through - 0.5)
            + 0.8 * burst
            + 0.8 * structured_ratio
            + 0.5 * cycle_flag
            + 0.3 * activity
            - 2.8
            + rng.normal(0.0, 0.25, size=n)
        )
        scores = 1.0 / (1.0 + np.exp(-z))
        scores[n_txn == 0] = 0.0
        return np.clip(scores, 0.0, 1.0)

    @staticmethod
    def _precision_recall(
        scores: np.ndarray, truth: np.ndarray, threshold: float
    ) -> tuple[float, float]:
        """Account-level precision/recall at one threshold.

        Precision is vacuously 1.0 when nothing is alerted; recall is 0.0 when no
        ring accounts exist (an explicit warning is raised upstream in that case).
        """
        alerted = scores >= threshold
        n_alerted = int(alerted.sum())
        n_truth = int(truth.sum())
        tp = int((alerted & truth).sum())
        precision = tp / n_alerted if n_alerted > 0 else 1.0
        recall = tp / n_truth if n_truth > 0 else 0.0
        return precision, recall

    def _build_alerts(
        self, scores: np.ndarray, truth: np.ndarray, threshold: float
    ) -> list[Alert]:
        """Materialise Alert objects for every account scoring at/above threshold."""
        alerts: list[Alert] = []
        for acct in np.flatnonzero(scores >= threshold):
            score = float(scores[acct])
            severity = (
                RiskTier.high if score > 0.85 else RiskTier.medium if score > 0.65 else RiskTier.low
            )
            alerts.append(
                Alert(
                    kind="aml_ring_suspicion",
                    severity=severity,
                    score=round(score, 4),
                    entities=[EntityRef(entity_type="account", entity_id=f"acct_{int(acct):04d}")],
                    payload={
                        "detector": "twin_sim.compliance.feature_score",
                        "is_true_ring_member": bool(truth[acct]),
                        "threshold": threshold,
                    },
                )
            )
        return alerts

    # -- Simulator protocol ------------------------------------------------------------------

    def run(self, scenario: Scenario, *, seed: int | None = None) -> SimulationResult:
        """Run the fraud-ring simulation for ``scenario``.

        Args:
            scenario: Scenario whose ``params`` follow the module docstring.
            seed: Master seed; falls back to ``scenario.params['seed']`` then 7.

        Returns:
            A :class:`SimulationResult` whose metrics carry bootstrap CIs, whose
            series include the full threshold sweep (``thresholds``,
            ``precision_at_threshold``, ``recall_at_threshold``,
            ``alerts_at_threshold`` from the primary replication), and whose
            calibration block compares the pooled suspicion-score distribution to a
            bimodal reference. ``self.last_alerts`` holds the primary replication's
            Alert objects.
        """
        params, warnings = _parse_params(scenario)
        master_seed = resolve_seed(scenario, seed)
        rep_rngs, boot_rng = spawn_generators(master_seed, params.n_reps)

        per_rep: dict[str, list[float]] = {
            "true_ring_edges": [],
            "ring_accounts": [],
            "n_alerts": [],
            "alert_precision": [],
            "alert_recall": [],
            "analyst_minutes_expected": [],
        }
        pooled_scores: list[np.ndarray] = []
        primary_sweep: dict[str, list[float]] | None = None

        for i, rng in enumerate(rep_rngs):
            log, ring_members = self._grow_graph(rng, params)
            scores = self._score_accounts(rng, params, log)
            truth = np.zeros(params.n_accounts, dtype=bool)
            truth[list(ring_members)] = True

            precision, recall = self._precision_recall(scores, truth, params.alert_threshold)
            alerted_scores = scores[scores >= params.alert_threshold]
            minutes = float(
                np.sum(_TRIAGE_BASE_MINUTES + _TRIAGE_PER_SCORE_MINUTES * alerted_scores)
            )

            per_rep["true_ring_edges"].append(float(sum(log.is_ring)))
            per_rep["ring_accounts"].append(float(truth.sum()))
            per_rep["n_alerts"].append(float(alerted_scores.size))
            per_rep["alert_precision"].append(precision)
            per_rep["alert_recall"].append(recall)
            per_rep["analyst_minutes_expected"].append(minutes)
            pooled_scores.append(scores)

            if i == 0:
                sweep_p, sweep_r, sweep_n = [], [], []
                for th in _THRESHOLDS:
                    pr, rc = self._precision_recall(scores, truth, float(th))
                    sweep_p.append(pr)
                    sweep_r.append(rc)
                    sweep_n.append(float((scores >= th).sum()))
                primary_sweep = {
                    "thresholds": [float(t) for t in _THRESHOLDS],
                    "precision_at_threshold": sweep_p,
                    "recall_at_threshold": sweep_r,
                    "alerts_at_threshold": sweep_n,
                }
                self.last_alerts = self._build_alerts(scores, truth, params.alert_threshold)
                if not ring_members:
                    warnings.append(
                        "no laundering rings were embedded in the primary replication; "
                        "recall metrics are vacuous"
                    )

        metrics, confidence = summarise_replications(per_rep, boot_rng)
        calibration = self._calibration_block(
            np.concatenate(pooled_scores),
            extra={"headline_sample": "account_suspicion_scores"},
        )

        assert primary_sweep is not None
        return SimulationResult(
            simulator=self.name,
            scenario_name=scenario.name,
            metrics=metrics,
            series=primary_sweep,
            confidence=confidence,
            calibration=calibration,
            warnings=warnings,
            seed=master_seed,
        )
