"""Shared scaffolding for FinTwinOS calibrated simulators.

Every simulator in this package follows the same replication discipline:

1. Resolve a deterministic master seed (explicit ``seed`` argument beats the
   scenario's ``params["seed"]`` beats the module default).
2. Spawn independent child seed sequences — one per Monte-Carlo replication plus one
   reserved for bootstrap resampling — via ``numpy.random.SeedSequence.spawn``.
3. Run the replications, collect one value per replication for every headline metric.
4. Report the mean across replications as the point estimate and a percentile
   bootstrap interval (>= 200 draws, see :mod:`fintwinos.twin_sim.calibration`) as
   ``confidence[metric] = [lo, hi]``.
5. Attach a calibration block ``{"ks_stat": ..., "calibrated": bool, ...}`` comparing
   a headline sample distribution against the simulator's reference sample, applying
   the fitted :class:`~fintwinos.twin_sim.calibration.PostHocCalibrator` when present.

Centralising this here keeps the four domain simulators focused on their dynamics
while guaranteeing the contract of CONTRACTS.md hard rule 3: reproducible results with
confidence intervals and a calibration block, never bare point estimates.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from fintwinos.core.types import Scenario
from fintwinos.twin_sim.calibration import (
    DEFAULT_BOOTSTRAP_DRAWS,
    PostHocCalibrator,
    bootstrap_ci,
    ks_statistic,
)

#: Default master seed when neither the caller nor the scenario supplies one.
DEFAULT_SEED = 7


def resolve_seed(scenario: Scenario, seed: int | None, default: int = DEFAULT_SEED) -> int:
    """Resolve the master seed for a run.

    Precedence: explicit ``seed`` argument, then ``scenario.params["seed"]``, then
    the module default. The result is always a plain ``int`` so it can be stored on
    :class:`~fintwinos.core.types.SimulationResult.seed` verbatim.
    """
    if seed is not None:
        return int(seed)
    scenario_seed = scenario.params.get("seed")
    if scenario_seed is not None:
        return int(scenario_seed)
    return int(default)


def stable_int_hash(text: str, modulus: int = 2**31 - 1) -> int:
    """Process-independent integer hash of a string (sha256-based).

    Python's builtin ``hash`` is salted per interpreter; this helper gives a stable
    value for deriving structural randomness (e.g. a currency's baseline flow shape)
    that must not change between runs or machines.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def spawn_generators(seed: int, n_reps: int) -> tuple[list[np.random.Generator], np.random.Generator]:
    """Spawn ``n_reps`` independent replication generators plus a bootstrap generator.

    Uses ``numpy.random.SeedSequence.spawn`` so the replication streams are
    statistically independent and the whole tree is reproducible from one seed.

    Returns:
        ``(rep_rngs, boot_rng)`` where ``rep_rngs`` has length ``n_reps``.
    """
    children = np.random.SeedSequence(seed).spawn(n_reps + 1)
    rep_rngs = [np.random.default_rng(child) for child in children[:n_reps]]
    boot_rng = np.random.default_rng(children[-1])
    return rep_rngs, boot_rng


def summarise_replications(
    per_rep_metrics: dict[str, list[float]],
    boot_rng: np.random.Generator,
    *,
    n_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    alpha: float = 0.10,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Collapse per-replication metric samples into point estimates and CIs.

    Args:
        per_rep_metrics: ``metric name -> one value per replication``.
        boot_rng: Generator dedicated to bootstrap resampling.
        n_draws: Bootstrap draws (>= 200 enforced downstream).
        alpha: Two-sided tail mass for the percentile interval.

    Returns:
        ``(metrics, confidence)`` where ``metrics[name]`` is the cross-replication
        mean and ``confidence[name] = [lo, hi]``. Every metric receives an interval.
    """
    metrics: dict[str, float] = {}
    confidence: dict[str, list[float]] = {}
    for name in sorted(per_rep_metrics):
        values = np.asarray(per_rep_metrics[name], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            metrics[name] = float("nan")
            confidence[name] = [float("nan"), float("nan")]
            continue
        metrics[name] = float(np.mean(finite))
        confidence[name] = bootstrap_ci(finite, boot_rng, n_draws=n_draws, alpha=alpha)
    return metrics, confidence


class CalibratedSimulatorMixin:
    """Calibration bookkeeping shared by all four simulators.

    Subclasses must provide ``name`` and a ``_reference_sample()`` method returning a
    deterministic 1-d numpy array that stands in for the historical distribution of
    the simulator's headline sample (returns, net flows, suspicion scores, waits).
    Users may overwrite the reference and fit a quantile-mapping calibrator with
    :meth:`fit_calibration`.
    """

    name: str = "simulator"

    def __init__(self) -> None:
        self.calibrator: PostHocCalibrator | None = None
        self._last_calibration: dict[str, Any] = {}
        self._reference_override: np.ndarray | None = None

    # -- reference management -----------------------------------------------------

    def _reference_sample(self) -> np.ndarray:  # pragma: no cover - abstract-ish
        """Deterministic stand-in reference distribution; subclasses override."""
        raise NotImplementedError

    def reference(self) -> np.ndarray:
        """The active reference sample (user-supplied override wins)."""
        if self._reference_override is not None:
            return self._reference_override
        return self._reference_sample()

    def fit_calibration(self, reference_sample: Any, sim_sample: Any) -> PostHocCalibrator:
        """Fit a quantile-mapping calibrator and adopt the reference for future runs.

        Args:
            reference_sample: Observed/historical draws the simulator should match.
            sim_sample: Draws produced by this simulator under comparable conditions.

        Returns:
            The fitted :class:`PostHocCalibrator` (also stored on ``self``).
        """
        ref = np.asarray(reference_sample, dtype=float).ravel()
        self._reference_override = ref
        self.calibrator = PostHocCalibrator().fit(sim_sample, ref)
        return self.calibrator

    # -- calibration block ----------------------------------------------------------

    def _calibration_block(self, headline_sample: np.ndarray, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build the per-run calibration dict required on every SimulationResult.

        Computes the KS distance between the run's headline sample and the active
        reference. When a calibrator is fitted, the headline sample is passed through
        the quantile map first and both raw and calibrated KS values are reported.
        """
        sample = np.asarray(headline_sample, dtype=float).ravel()
        sample = sample[np.isfinite(sample)]
        block: dict[str, Any] = {
            "calibrated": bool(self.calibrator is not None and self.calibrator.fitted),
            "method": "quantile_mapping" if self.calibrator else "none",
            "reference_size": int(self.reference().size),
            "sample_size": int(sample.size),
        }
        if sample.size == 0:
            block["ks_stat"] = None
            block["note"] = "empty headline sample; KS not computable"
        else:
            raw_ks = ks_statistic(sample, self.reference())
            if self.calibrator is not None and self.calibrator.fitted:
                mapped = self.calibrator.transform(sample)
                block["ks_stat"] = round(ks_statistic(mapped, self.reference()), 6)
                block["ks_stat_raw"] = round(raw_ks, 6)
            else:
                block["ks_stat"] = round(raw_ks, 6)
        if extra:
            block.update(extra)
        self._last_calibration = dict(block)
        return block

    def calibration_report(self) -> dict[str, Any]:
        """Calibration state of this simulator (Simulator protocol requirement).

        Returns:
            A dict with the simulator name, the last run's calibration block, the
            calibrator's own fit diagnostics (or ``None``) and reference metadata.
        """
        return {
            "simulator": self.name,
            "last_run": dict(self._last_calibration),
            "calibrator": self.calibrator.report() if self.calibrator else None,
            "reference_size": int(self.reference().size),
            "reference_overridden": self._reference_override is not None,
        }
