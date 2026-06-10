# Model card: simulators

| | |
|---|---|
| **Artefact** | The FinTwinOS simulator family: agent-based market, treasury liquidity, compliance/fraud-ring, and customer-ops queue simulators |
| **Modules** | `fintwinos/twin_sim/` (implementations), registered onto a runtime by `fintwinos.twin_sim.register_all(runtime)`; contract in `fintwinos/core/interfaces.py` (`Simulator` protocol) |
| **Type** | Hybrid rules + agent-based + statistical simulators, implemented in numpy/networkx/pandas — chosen over a single ML world-model for auditability and per-component calibration |
| **Version** | Every scenario and result carries an explicit `assumptions_version`; bumped on any recalibration or structural change |
| **Owner** | One model owner per simulator; validation by an independent reviewer (effective challenge) |

## The honesty contract

One sentence governs this entire family: **simulators are never treated as
faithful worlds.** Concretely, the `Simulator` protocol makes dishonesty a
contract violation:

- `run(scenario, *, seed)` returns a `SimulationResult` with point `metrics`,
  full `series`, **confidence intervals for every headline metric**, a
  `calibration` block carrying current diagnostics, `warnings`, the `seed` and
  the `assumptions_version` — bare point estimates are structurally impossible.
- `calibration_report()` exposes live KS-distance and interval-coverage
  diagnostics ([calibration runbook](../runbooks/calibration.md)).
- All randomness flows through `numpy.random.default_rng(seed)`: identical
  scenario + seed reproduces identical results, so simulation evidence in the
  audit chain is regenerable on demand.
- Out-of-tolerance diagnostics raise `CalibrationError` and remove the simulator
  from decision support.

## The family

### Agent-based market simulator

- **Simulates:** price paths, venue conditions and market-impact responses from
  interacting order-flow agents; stress scenarios for risk and rehearsal for
  trading decisions.
- **Validation anchor:** reproduction of stylised facts (fat tails, volatility
  clustering, impact concavity) and stressed-regime behaviour, following the 2024
  work on RL in agent-based market simulation
  ([research basis](../research-basis.md#simulation-and-sim-to-real)).
- **Feeds:** `simulate_*` tools for shock simulation and hedging-candidate
  rehearsal; the RL layer's environment for bounded trading-adjacent decisions.

### Treasury liquidity simulator

- **Simulates:** cash ladders, intraday liquidity usage, funding-spread dynamics
  and collateral movements under stress scenarios.
- **Headline metrics:** liquidity-coverage trajectories, funding-cost deltas,
  time-to-breach under stress — each with confidence intervals.
- **Feeds:** liquidity stress demos (`FINTWIN_OFFLINE=1 fintwinos demo liquidity`),
  contingency-action prioritisation, Expected Shortfall-constrained RL rewards.

### Compliance / fraud-ring simulator

- **Simulates:** evolution of entity graphs containing laundering-style rings and
  benign traffic; alert-volume and false-positive trade-offs under policy
  changes.
- **Feeds:** the false-positive/recall trade-off analyses for compliance policy
  proposals; synthetic structure for the
  [AML subgraph scorer](aml-subgraph-scorer.md); the `aml_triage` demo.

### Customer-ops queue simulator

- **Simulates:** case arrival, routing, handling-time and escalation dynamics
  across SLA queues, with calibrated persona stress tests for journey outcomes.
- **Headline metrics:** SLA breach probabilities, average handling time,
  escalation rates — with intervals, since queue tails dominate the risk.
- **Feeds:** service policy what-ifs, staffing and routing decisions in the RL
  layer's bounded set, the `customer_ops` demo.

## Data and calibration

- **Calibration corpus:** historical replay episodes (`ReplayEngine`) and offline
  series from the `TimeSeriesStore`; parameter *distributions* are fitted to
  offline data (offline domain-randomisation approach, 2025) rather than
  hand-tuned point values.
- **Bundled data:** deterministic synthetic generators (`fintwinos/datasets/`)
  and the demo twin (`build_runtime(seed=7, with_demo_data=True)`) make every
  simulator runnable and testable fully offline.
- **Diagnostics:** Kolmogorov–Smirnov distance per calibration target and
  empirical interval coverage, computed in numpy, with the tolerance tables and
  recalibration triggers defined in the
  [calibration runbook](../runbooks/calibration.md).
- **Versioning:** any refit or structural change bumps `assumptions_version`;
  results across versions are never silently comparable.

## Metrics

- Per-simulator calibration health: KS distances, coverage rates, time since
  last recalibration — surfaced in `calibration_report()` and eval reports.
- Scenario coverage (risk/treasury gate metric): the proportion of the approved
  scenario library a release has exercised.
- Determinism checks: fixed-seed reproducibility asserted in CI
  ([worked check](../runbooks/calibration.md#worked-check-offline-deterministic)).
- The [simulate-before-act ablation](../evaluation.md#3-simulate-before-act-ablation)
  quantifies the decision-quality value of the family as a whole — simulation has
  to keep earning its place in the loop.

## Limitations

- **Stylised, not faithful.** Each simulator reproduces selected statistical
  properties of its domain. Outside those properties — novel market
  microstructure, unprecedented liquidity spirals, unseen fraud typologies,
  atypical customer behaviour — output reverts to assumption, and the
  `calibration` block is the only honest signal of how far to trust it.
- **Regime lag.** Calibration is backward-looking; diagnostics breach *after*
  reality moves. Regime events are therefore an explicit recalibration trigger
  rather than something the simulators self-detect.
- **Gaming surface.** Anything used to justify decisions invites optimisation
  against its quirks — by people or by RL policies. Controls: recorded seeds,
  versioned assumptions, out-of-sample diagnostics, and shadow-mode comparison
  against reality as the final arbiter
  ([threat T6](../threat-model.md#t6--simulator-gaming)).
- **Synthetic demo data** exercises mechanics, not market truth: demo-twin
  results demonstrate the platform, never a business case.
- **Interaction effects** between domains (e.g. liquidity stress driving customer
  complaints) are only captured where scenarios explicitly couple simulators;
  cross-domain contagion modelling is roadmap work, not a current claim.

## Governance hooks

- Simulators are reachable only through `simulate_*` tools: side-effect free by
  registry invariant, audited per call, with simulation branches recorded in the
  audit chain.
- Every result is reproducible from its recorded seed and assumptions version —
  simulation evidence presented to an approver can be regenerated exactly during
  review or forensics.
- `CalibrationError` is a hard removal from decision support; restoration
  requires the recalibration procedure with independent sign-off.
- Scenario libraries (`Scenario`, kinds `stress` / `replay` / `counterfactual` /
  `what_if`) are versioned artefacts in the documentation pack expected by
  BoE/FCA and SR 11-7 ([controls map](../governance-controls-map.md)).

---

FinTwinOS — created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/) — MIT License.
