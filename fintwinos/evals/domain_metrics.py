"""Domain release-gate suites computed from simulator outputs.

Four suites, one per founding-brief domain gate, all driven by the simulators that
``fintwinos.twin_sim.register_all`` attaches to the twin runtime:

- **risk** (``domain_risk``) — preset stress scenarios must run cleanly
  (*scenario coverage*), and every covered scenario must report an Expected
  Shortfall metric with confidence intervals (never a bare point estimate).
- **treasury** (``domain_treasury``) — survival days under preset deposit-runoff
  stresses must stay above the floor.
- **compliance** (``domain_compliance``) — detection recall on synthetic AML
  typologies, measured at one fixed alert threshold, must clear the floor.
- **customer ops** (``domain_customer_ops``) — the SLA breach rate under baseline
  staffing must stay below the ceiling.

Every threshold lives in the single :data:`GATES` dict so the brief's release gates
are encoded in exactly one reviewable place; :func:`fintwinos.evals.harness.builtin_gates`
reads the same dict, so reports and assertions can never drift apart.

Simulators are looked up by fuzzy name (e.g. any simulator whose name mentions
``liquidity`` or ``treasury`` serves the treasury suite) and metrics by fuzzy key, so
the suites bind to the twin_sim module without importing its internals — per the
module contract. When the runtime or a needed simulator is absent the suite raises
:class:`SuiteUnavailable` and the runner reports it as skipped (a skipped suite fails
its gates: no evidence, no promotion).
"""

from __future__ import annotations

import re
from typing import Any

from fintwinos.core.types import Scenario, SimulationResult
from fintwinos.evals.harness import EvalCase, EvalResult, Suite, SuiteUnavailable

#: The single source of truth for every release-gate threshold in FinTwinOS.
GATES: dict[str, float] = {
    # tool-calling gates (consumed by harness.builtin_gates)
    "function_calls.pass_rate_min": 0.95,
    "function_calls.hallucinated_tool_rate_max": 0.02,
    # agent-behaviour gates
    "consistency.mean_score_min": 0.90,
    # domain gates
    "risk.scenario_coverage_min": 0.80,
    "risk.es_present_rate_min": 1.00,
    "treasury.survival_days_floor": 30.0,
    "compliance.recall_floor": 0.75,
    "compliance.alert_threshold": 0.50,  # fixed threshold recall is measured at
    "customer_ops.sla_breach_ceiling": 0.10,
}

_ES_KEY_RE = re.compile(r"(?:^|_)(?:es|cvar)(?:_|$)|expected_shortfall")


def _find_metric(metrics: dict[str, float], *substrings: str) -> tuple[str, float] | None:
    """Find the first metric whose key contains any substring (exact key wins first)."""
    for sub in substrings:
        if sub in metrics:
            return sub, float(metrics[sub])
    for sub in substrings:
        for key in sorted(metrics):
            if sub in key:
                return key, float(metrics[key])
    return None


def _es_metric(metrics: dict[str, float]) -> str | None:
    """Key of an Expected Shortfall-style metric (es_*, *_es, cvar, expected_shortfall)."""
    for key in sorted(metrics):
        if _ES_KEY_RE.search(key.lower()):
            return key
    return None


class SimulatorSuite(Suite):
    """Base for suites that score :class:`SimulationResult` outputs of one simulator.

    Subclasses set ``simulator_keywords`` (fuzzy name match against the runtime's
    simulator registry) and implement :meth:`score_result`.
    """

    simulator_keywords: tuple[str, ...] = ()

    def __init__(self, name: str, cases: list[EvalCase], runtime: Any | None, seed: int = 7):
        super().__init__(name, cases)
        self.runtime = runtime
        self.seed = seed

    # -- simulator resolution ---------------------------------------------------

    def resolve_simulator(self) -> Any:
        if self.runtime is None:
            raise SuiteUnavailable(f"{self.name}: twin runtime not available")
        simulators: dict[str, Any] = getattr(self.runtime, "simulators", None) or {}
        for key in sorted(simulators):
            haystack = f"{key} {getattr(simulators[key], 'name', '')}".lower()
            if any(kw in haystack for kw in self.simulator_keywords):
                return simulators[key]
        raise SuiteUnavailable(
            f"{self.name}: no simulator matching {self.simulator_keywords} registered "
            f"(have: {sorted(simulators) or 'none'})"
        )

    def preflight(self, subject: Any) -> None:
        self.resolve_simulator()

    # -- evaluation ---------------------------------------------------------------

    def scenario_for(self, case: EvalCase) -> Scenario:
        return Scenario(
            name=str(case.input.get("scenario_name", case.id)),
            kind="stress",
            params=dict(case.input.get("params", {})),
        )

    async def evaluate_case(self, case: EvalCase, subject: Any) -> EvalResult:
        simulator = self.resolve_simulator()
        result = simulator.run(self.scenario_for(case), seed=self.seed)
        if not isinstance(result, SimulationResult):
            result = SimulationResult.model_validate(result)
        return self.score_result(case, result)

    def score_result(self, case: EvalCase, result: SimulationResult) -> EvalResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Risk: scenario coverage + Expected Shortfall presence
# ---------------------------------------------------------------------------

#: Preset stresses in the market simulator's documented param vocabulary
#: (``shock_bps``, ``regime``); ``asset_class`` is a descriptive label.
RISK_STRESS_PRESETS: list[dict[str, Any]] = [
    {"scenario_name": "rates_up_200bp",
     "params": {"asset_class": "rates", "shock_bps": 200, "regime": "stressed"}},
    {"scenario_name": "equity_down_20pct",
     "params": {"asset_class": "equity", "shock_bps": -2000, "regime": "stressed"}},
    {"scenario_name": "credit_spreads_plus_150bp",
     "params": {"asset_class": "credit", "shock_bps": 150, "regime": "stressed"}},
    {"scenario_name": "fx_usd_up_8pct",
     "params": {"asset_class": "fx", "shock_bps": 800, "regime": "calm"}},
    {"scenario_name": "liquidity_squeeze",
     "params": {"asset_class": "rates", "shock_bps": 75, "regime": "stressed"}},
]


class RiskScenarioSuite(SimulatorSuite):
    """Coverage of preset stress scenarios with ES + confidence intervals present."""

    simulator_keywords = ("market", "risk", "stress")

    def __init__(self, runtime: Any | None = None, seed: int = 7, name: str = "domain_risk"):
        cases = [
            EvalCase(id=f"risk-{preset['scenario_name']}", suite=name, input=preset)
            for preset in RISK_STRESS_PRESETS
        ]
        super().__init__(name, cases, runtime, seed)

    def score_result(self, case: EvalCase, result: SimulationResult) -> EvalResult:
        covered = bool(result.ok and result.metrics)
        es_key = _es_metric(result.metrics) if covered else None
        has_confidence = bool(result.confidence)
        score = 0.4 * covered + 0.3 * (es_key is not None) + 0.3 * has_confidence
        return EvalResult(
            case_id=case.id,
            passed=covered and es_key is not None and has_confidence,
            score=round(score, 6),
            details={
                "covered": covered,
                "es_metric": es_key,
                "has_confidence": has_confidence,
                "warnings": list(result.warnings),
            },
        )

    def extra_metrics(self, results: list[EvalResult]) -> dict[str, float]:
        if not results:
            return {}
        n = len(results)
        covered = sum(1 for r in results if r.details.get("covered"))
        es = sum(1 for r in results if r.details.get("es_metric"))
        return {"scenario_coverage": covered / n, "es_present_rate": es / n}


# ---------------------------------------------------------------------------
# Treasury: survival-days floor under preset runoff stresses
# ---------------------------------------------------------------------------

#: LCR-style preset stresses: the institution must survive at least the floor under
#: baseline, mild (1.25x) and moderate (1.5x) outflow multipliers over a 90-day
#: horizon. Severe (2x+) stresses belong in exploratory risk analysis, not the floor
#: gate. Param names follow the treasury simulator's documented vocabulary.
TREASURY_STRESS_PRESETS: list[dict[str, Any]] = [
    {"scenario_name": "baseline_outflows",
     "params": {"stress_outflow_multiplier": 1.0, "horizon_days": 90}},
    {"scenario_name": "mild_stress_125pct_outflows",
     "params": {"stress_outflow_multiplier": 1.25, "horizon_days": 90}},
    {"scenario_name": "moderate_stress_150pct_outflows",
     "params": {"stress_outflow_multiplier": 1.5, "horizon_days": 90}},
]


class TreasurySurvivalSuite(SimulatorSuite):
    """Survival days under deposit-runoff stress must clear the brief's floor."""

    simulator_keywords = ("liquidity", "treasury", "funding")

    def __init__(self, runtime: Any | None = None, seed: int = 7, name: str = "domain_treasury"):
        cases = [
            EvalCase(id=f"treasury-{preset['scenario_name']}", suite=name, input=preset)
            for preset in TREASURY_STRESS_PRESETS
        ]
        super().__init__(name, cases, runtime, seed)

    def score_result(self, case: EvalCase, result: SimulationResult) -> EvalResult:
        floor = GATES["treasury.survival_days_floor"]
        found = _find_metric(result.metrics, "survival_days", "survival")
        if not result.ok or found is None:
            return EvalResult(
                case_id=case.id, passed=False, score=0.0,
                details={"survival_days": None,
                         "error": "no survival metric in simulator output"},
            )
        key, survival = found
        return EvalResult(
            case_id=case.id,
            passed=survival >= floor,
            score=round(min(1.0, survival / floor), 6) if floor > 0 else 1.0,
            details={"survival_days": survival, "metric_key": key, "floor": floor},
        )

    def extra_metrics(self, results: list[EvalResult]) -> dict[str, float]:
        values = [
            float(r.details["survival_days"]) for r in results
            if r.details.get("survival_days") is not None
        ]
        return {"min_survival_days": min(values) if values else 0.0}


# ---------------------------------------------------------------------------
# Compliance: recall floor at one fixed alert threshold
# ---------------------------------------------------------------------------

#: Ring-density presets at the one fixed alert threshold. ``alert_threshold`` is the
#: compliance simulator's documented param; ``threshold`` is kept as a mirror for
#: alternative implementations. Recall is gated on the *worst* preset.
COMPLIANCE_PRESETS: list[dict[str, Any]] = [
    {"scenario_name": "sparse_rings",
     "params": {"ring_rate": 0.08,
                "alert_threshold": GATES["compliance.alert_threshold"],
                "threshold": GATES["compliance.alert_threshold"]}},
    {"scenario_name": "baseline_rings",
     "params": {"ring_rate": 0.12,
                "alert_threshold": GATES["compliance.alert_threshold"],
                "threshold": GATES["compliance.alert_threshold"]}},
    {"scenario_name": "dense_rings",
     "params": {"ring_rate": 0.20,
                "alert_threshold": GATES["compliance.alert_threshold"],
                "threshold": GATES["compliance.alert_threshold"]}},
]


class ComplianceRecallSuite(SimulatorSuite):
    """Detection recall at the fixed alert threshold across ring-density presets."""

    simulator_keywords = ("aml", "compliance", "detection", "typology")

    def __init__(self, runtime: Any | None = None, seed: int = 7,
                 name: str = "domain_compliance"):
        cases = [
            EvalCase(id=f"compliance-{preset['scenario_name']}", suite=name, input=preset)
            for preset in COMPLIANCE_PRESETS
        ]
        super().__init__(name, cases, runtime, seed)

    @staticmethod
    def _recall_from(metrics: dict[str, float]) -> float | None:
        found = _find_metric(metrics, "recall")
        if found is not None:
            return found[1]
        tp = _find_metric(metrics, "true_positives", "true_positive")
        fn = _find_metric(metrics, "false_negatives", "false_negative")
        if tp is not None and fn is not None and (tp[1] + fn[1]) > 0:
            return tp[1] / (tp[1] + fn[1])
        return None

    def score_result(self, case: EvalCase, result: SimulationResult) -> EvalResult:
        floor = GATES["compliance.recall_floor"]
        recall = self._recall_from(result.metrics) if result.ok else None
        if recall is None:
            return EvalResult(
                case_id=case.id, passed=False, score=0.0,
                details={"recall": None,
                         "error": "no recall (or TP/FN) metric in simulator output"},
            )
        return EvalResult(
            case_id=case.id,
            passed=recall >= floor,
            score=round(min(1.0, recall / floor), 6) if floor > 0 else 1.0,
            details={"recall": round(recall, 6), "floor": floor,
                     "threshold": GATES["compliance.alert_threshold"]},
        )

    def extra_metrics(self, results: list[EvalResult]) -> dict[str, float]:
        values = [
            float(r.details["recall"]) for r in results if r.details.get("recall") is not None
        ]
        # conservative: the gate sees the *worst* typology, not the average
        return {"recall": min(values) if values else 0.0}


# ---------------------------------------------------------------------------
# Customer ops: SLA breach ceiling under baseline staffing
# ---------------------------------------------------------------------------

#: Baseline staffing in the customer-ops simulator's documented vocabulary.
CUSTOMER_OPS_PRESETS: list[dict[str, Any]] = [
    {"scenario_name": "baseline_staffing",
     "params": {"n_agents": 8, "arrival_rate_per_hr": 60, "horizon_hours": 8.0,
                "sla_minutes": 15.0}},
]


class CustomerOpsSlaSuite(SimulatorSuite):
    """SLA breach rate under baseline staffing must stay under the ceiling."""

    simulator_keywords = ("ops", "customer", "backlog", "sla")

    def __init__(self, runtime: Any | None = None, seed: int = 7,
                 name: str = "domain_customer_ops"):
        cases = [
            EvalCase(id=f"ops-{preset['scenario_name']}", suite=name, input=preset)
            for preset in CUSTOMER_OPS_PRESETS
        ]
        super().__init__(name, cases, runtime, seed)

    def score_result(self, case: EvalCase, result: SimulationResult) -> EvalResult:
        ceiling = GATES["customer_ops.sla_breach_ceiling"]
        found = _find_metric(result.metrics, "sla_breach_rate", "sla_breach", "breach_rate")
        if not result.ok or found is None:
            return EvalResult(
                case_id=case.id, passed=False, score=0.0,
                details={"sla_breach_rate": None,
                         "error": "no SLA breach metric in simulator output"},
            )
        key, rate = found
        score = 1.0 if rate <= ceiling else round(min(1.0, ceiling / rate), 6)
        return EvalResult(
            case_id=case.id,
            passed=rate <= ceiling,
            score=score,
            details={"sla_breach_rate": round(rate, 6), "metric_key": key,
                     "ceiling": ceiling},
        )

    def extra_metrics(self, results: list[EvalResult]) -> dict[str, float]:
        values = [
            float(r.details["sla_breach_rate"]) for r in results
            if r.details.get("sla_breach_rate") is not None
        ]
        # conservative: the gate sees the worst observed breach rate
        return {"sla_breach_rate": max(values) if values else 1.0}


def domain_suites(runtime: Any | None, seed: int = 7) -> list[Suite]:
    """All four domain suites bound to one runtime, in report order."""
    return [
        RiskScenarioSuite(runtime, seed),
        TreasurySurvivalSuite(runtime, seed),
        ComplianceRecallSuite(runtime, seed),
        CustomerOpsSlaSuite(runtime, seed),
    ]
