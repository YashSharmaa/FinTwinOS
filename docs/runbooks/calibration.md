# Calibration runbook

Simulators are never treated as faithful worlds. Every FinTwinOS simulator must
expose confidence intervals and calibration diagnostics, never bare point
estimates, and this runbook describes how those diagnostics are computed, when a
simulator must be recalibrated, and how a recalibration is executed and signed off.

Audience: the model owners of `fintwinos/twin_sim/` simulators and the validation
function that challenges them. Companions: the
[simulators model card](../model-cards/simulators.md) (assumptions and
limitations), the [evaluation page](../evaluation.md) (gates that depend on
calibration), and [threat T6](../threat-model.md#t6-simulator-gaming) (why
calibration is also a security control).

## The contract

Defined by the `Simulator` protocol (`fintwinos/core/interfaces.py`) and
`SimulationResult` (`fintwinos/core/types.py`):

- `run(scenario, *, seed)` returns a `SimulationResult` carrying point `metrics`,
  full `series`, **`confidence` intervals per headline metric** (`name → [lo, hi]`),
  a `calibration` block, `warnings`, the `seed` and the `assumptions_version`.
- `calibration_report()` returns the simulator's current diagnostics on demand.
- All randomness flows through `numpy.random.default_rng(seed)`: the same scenario
  and seed reproduce the same result bit-for-bit, so every piece of simulation
  evidence in the audit trail can be regenerated.
- Diagnostics outside tolerance are the signal to pull a simulator from decision
  support until recalibrated. `simulator.run()` reports the diagnostics in the
  result's `calibration` block but does not raise on a breach; the
  `CalibrationError` type (`fintwinos/core/errors.py`) exists for deployments to
  wire that enforcement in, and the procedure below is operator-driven.

## The two core diagnostics

Both are implemented in numpy (no external stats dependency) and reported in the
`calibration` block of every result.

### 1. Kolmogorov–Smirnov distance (distributional fidelity)

For each calibration target (e.g. daily return distribution, queue wait times,
intraday liquidity drawdowns), compare the simulator's output distribution with
the realised distribution from historical replay episodes:

`D = sup_x | F_sim(x) − F_real(x) |`

computed over the pooled sample grid of the two empirical CDFs. Recommended
interpretation guidance (operator-driven; not enforced automatically in code):

| KS distance `D` | Reading | Action |
|---|---|---|
| `D < 0.05` | Excellent agreement | None |
| `0.05 ≤ D < 0.10` | Acceptable | Monitor; note in next review |
| `0.10 ≤ D < 0.20` | Degraded | Recalibrate before next decision-support use |
| `D ≥ 0.20` | Failed | Operator action: pull from decision support immediately and recalibrate |

Tolerances are per-simulator and per-target; the table above is the default
posture, and the simulator's model owner may only tighten it, not loosen it,
without an effective-challenge review.

### 2. Interval coverage (uncertainty honesty)

A simulator that nails the mean but lies about uncertainty is more dangerous than
one that is visibly noisy. For nominal `1 − α` confidence intervals (default
nominal level 90%), measure the empirical fraction of realised outcomes that fell
inside the simulator's stated intervals over the evaluation window:

`coverage = #(realised inside [lo, hi]) / #(observations)`

| Empirical coverage at nominal 90% | Reading | Action |
|---|---|---|
| 85% – 95% | Honest | None |
| 80% – 85% or 95% – 98% | Mildly off (over/under-confident) | Recalibrate interval widths |
| < 80% | **Overconfident, worst failure mode** | Operator pulls from decision support immediately |
| > 98% | Uselessly wide | Recalibrate; intervals carry no information |

Overconfidence is treated as the critical failure because downstream consumers,
human approvers and the policy gate alike, weight simulation evidence by its
stated uncertainty.

The KS diagnostic (`ks_stat`, with `calibrated` / `method` and reference metadata
such as `reference_size` / `sample_size`) is stamped into the `calibration` block of
every `SimulationResult` and surfaced by `simulator.calibration_report()`, so any
decision's simulation evidence carries the distributional-fidelity health of the
instrument that produced it. Interval coverage is computed by the standalone
`coverage_check` helper (`fintwinos/twin_sim/calibration.py`) during recalibration
sign-off, rather than embedded in each per-result `calibration` block.

## When to recalibrate

Recalibration is triggered by any of:

1. **Drift breach**, any diagnostic crossing its tolerance in scheduled
   monitoring (run at least weekly in shadow and production; daily for market
   simulators).
2. **Assumption change**, any edit to simulator parameters or structure. This
   *is* a recalibration, and it bumps `assumptions_version` by definition.
3. **Regime events**, a market, liquidity, fraud-pattern or operational regime
   shift in the real data, even if diagnostics have not yet breached (they lag).
4. **Cadence**, a hard ceiling per simulator class even if nothing triggered:
   quarterly at minimum, monthly for market and liquidity simulators.
5. **Upstream data change**, a new connector, a changed source schema, or a
   repaired poisoning incident touching the simulator's calibration targets.

## Recalibration procedure

1. **Freeze and announce.** Mark the simulator out of decision support (its
   results' `warnings` must say so; consumers in the agent layer treat such
   results as advisory only). Record the freeze in the audit trail.
2. **Assemble the calibration corpus.** Pull the relevant replay episodes
   (`runtime.replay.episodes()`) and realised series from the
   `TimeSeriesStore`. The corpus must cover the regime the simulator will be used
   in, fitting calm-period data and deploying into stress is the canonical
   failure (see [offline domain randomisation](../research-basis.md#simulation-and-sim-to-real)
   in the research basis).
3. **Refit.** Fit simulator parameter distributions to the offline corpus (not
   hand-tuned point values, fit the *distributions*, per the offline
   domain-randomisation approach). All fitting code is numpy against fixed seeds;
   record the seed set.
4. **Re-diagnose out-of-sample.** Compute KS and coverage on a held-out slice of
   episodes that played no part in fitting. In-sample diagnostics are reported but
   never used for sign-off.
5. **Bump `assumptions_version`.** The new version string appears on every
   subsequent `Scenario` and `SimulationResult`, so results from before and after
   the recalibration are never silently comparable.
6. **Re-run dependent suites.** `fintwinos eval all` plus the
   [simulate-before-act ablation](../evaluation.md#3-simulate-before-act-ablation);
   any RL policy trained against the old simulator re-enters off-policy
   evaluation ([staged ladder](../architecture.md#6-the-rl-layer-staged-bounded-constraint-first)).
7. **Sign off and unfreeze.** Model owner proposes; a reviewer independent of the
   fitting work approves (effective challenge, per the
   [controls map](../governance-controls-map.md)). The sign-off, the diagnostics
   and the version bump are all audit-trail records.

## Worked check (offline, deterministic)

A minimal end-to-end health check that runs anywhere, including air-gapped:

```python
from fintwinos.core.types import Scenario
from fintwinos.twin_core.runtime import build_runtime
from fintwinos.twin_sim import register_all

runtime = build_runtime(seed=7, with_demo_data=True)
register_all(runtime)

for name, sim in runtime.simulators.items():
    result = sim.run(Scenario(name="calibration-check", kind="stress"), seed=7)
    assert result.confidence, f"{name}: missing confidence intervals"
    assert result.calibration, f"{name}: missing calibration block"
    repeat = sim.run(Scenario(name="calibration-check", kind="stress"), seed=7)
    assert result.metrics == repeat.metrics, f"{name}: not deterministic under fixed seed"
    print(name, "ok,", sim.calibration_report())
```

Any assertion failure here is a contract violation, not a tuning issue, report
it as a bug on the issue tracker.

## Anti-gaming notes

Calibration is also a control against simulator gaming
([threat T6](../threat-model.md#t6-simulator-gaming)): seeds and
`assumptions_version` are recorded on every result, so seed-shopping and silent
assumption edits are visible in the audit chain; out-of-sample diagnostics prevent
fitting the test; and shadow-mode comparison against reality remains the final
arbiter above any simulator, however well it scores its own diagnostics.

---

FinTwinOS, created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/), MIT License.
