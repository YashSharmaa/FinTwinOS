# Evaluation stack and release gates

FinTwinOS's posture, aggressive on simulation and evaluation, conservative on
autonomy, only means something if evaluation is layered, continuous and tied to
hard release gates. This page defines the evaluation stack, the gates, the four
headline experiments, the public benchmark plan, and how the `fintwinos eval`
command maps onto all of it.

The motivating evidence is blunt: on the Finance Agent Benchmark's 537 real SEC
research tasks, the best model scored 46.8%; on BlueFin's real finance-spreadsheet
tasks, the strongest models stayed under 50%. Frontier models are not yet
dependable on hard financial work, which is why every claim in FinTwinOS must be
measured, not assumed. Sources are named in the [research basis](research-basis.md).

## The evaluation stack

Each layer has its own metrics and its own gate. A layer's gate must hold before
anything above it is trusted.

| Layer | Core metrics | Release gate |
|---|---|---|
| Function calls | Tool selection exactness, argument correctness, schema validity, hallucinated-tool rate | No write-enabled tools until exactness is stable |
| Agent runs | Task success, multi-run consistency, critic catch rate, human-review rate, latency, unit cost | No production use if reliability collapses across repeats |
| Risk & treasury | PnL impact, Expected Shortfall, limit breaches, funding-cost delta, scenario coverage | No autonomy unless hard-risk constraints never violated in replay and shadow |
| Compliance | Alert precision/recall, analyst minutes saved, policy citation accuracy, narrative completeness | No case-closure automation unless recall floors and policy traceability met |
| Customer ops | First-contact resolution, average handling time, complaint escalation rate, SLA breaches | No autonomous regulated actions without strong human-oversight evidence |
| Platform | Drift detection lag, trace completeness, approval latency, break-glass usage, incident recovery time | No scale-out without observable control effectiveness |

## Release gates

The gates are conjunctive, all applicable gates must pass, and they bind to the
[deployment rule](governance-controls-map.md#the-deployment-rule) (replay → shadow
→ human review → control testing). In code terms:

1. **Function-call gate.** Measured by the function-call suite over the typed
   registry (`fintwinos/tools/`): selection exactness and argument correctness on
   held-out task sets, schema-rejection rate (`tool.schema_rejected` audit events),
   and hallucinated-tool rate (`tool.unknown` events). Until these are stable
   across releases, `FINTWIN_EXECUTE_TOOLS_ENABLED` stays `0`, which is also its
   default.
2. **Agent-run gate.** `handle_case` runs are repeated with fixed seeds and
   compared for consistency of status, decision content and tool-call traces.
   Reliability that collapses across repeats blocks promotion regardless of
   single-run quality. Critic catch rate is measured by injecting known-flawed
   drafts and counting red-team detections.
3. **Domain gates (risk/treasury, compliance, customer ops).** Hard constraints
   are absolute: a single Expected Shortfall breach or recall-floor violation in
   replay or shadow fails the gate. Soft metrics (funding-cost delta, analyst
   minutes, handling time) are improvements to report, never licences to bypass
   hard constraints.
4. **Platform gate.** Trace completeness (every consequential event present in the
   audit chain and `AuditTrail.verify()` true), approval latency, and incident
   recovery time from kill-switch drills must be observable and within tolerance
   before any scale-out.

## The four headline experiments

These are the standing experiments the project maintains to keep its own
architecture honest. Each is runnable offline and reported by the eval runner.

### 1. Single agent vs swarm

Same objectives, two configurations: one generalist agent with the full tool
catalog versus the hierarchical-and-debating swarm (planner, sensing, domain and
critic, with the execute decision enforced by the policy gate). Measures task
success, consistency across repeats, critic
catch rate, latency and unit cost. The swarm must *earn* its coordination
overhead; multi-agent research (see the [research basis](research-basis.md))
shows more agents is not automatically better.

### 2. Prompt-only vs durable orchestration

The same workflows run under prompt-only self-orchestration versus the durable
external orchestration graph (`fintwinos/agents/runtime.py`). A 2026 controlled
study found prompt-only self-orchestration can win on procedural tasks, so
FinTwinOS measures where that holds, and reserves prompt-only mode for low-risk,
read-only work where it demonstrably does.

### 3. Simulate-before-act ablation

Identical decision tasks with and without the mandatory `simulate_*` rehearsal
step before proposals. Measures decision quality deltas, hard-constraint breach
rates, and the cost of simulation. This experiment is the empirical justification
for the platform's central promise; it is rerun whenever simulators are
recalibrated ([calibration runbook](runbooks/calibration.md)).

### 4. Rules vs offline RL on bounded control tasks

For each bounded control decision (queue routing, simulation budgets, hedging
candidate selection, escalation thresholds, staffing, scenario prioritisation):
the deterministic rule baseline versus the conservative offline-RL policy
(`fintwinos/rl/`), compared by off-policy evaluation first and shadow-mode
comparison second. An RL policy that cannot beat the rules under hard constraints
does not advance up the [staged ladder](architecture.md#6-the-rl-layer-staged-bounded-constraint-first).

## Benchmark plan

Public benchmarks anchor FinTwinOS to external reality. Adapters live in
`fintwinos/evals/`; datasets are fetched by the adopter (licences permitting) and
the suites degrade to bundled synthetic equivalents offline.

| Capability | Benchmark | What it tests in FinTwinOS |
|---|---|---|
| Function-call layer | BFCL V4 | Multi-turn tool selection, memory, hallucination and format sensitivity over the typed registry |
| Financial tool use | FinMCP-Bench | Real-world financial tool invocation through the MCP-style edge (`fintwinos serve-tools`) |
| Filings research | Finance Agent Benchmark | Hard SEC research questions against the document store and EDGAR connector |
| Filing QA | SECQUE | Comparison, ratio, risk and insight tasks for analyst agents |
| Spreadsheets | BlueFin | Real finance spreadsheet tasks for tabular reasoning |
| Graph compliance | Elliptic2 | Subgraph-level suspicious-pattern detection for the [AML scorer](model-cards/aml-subgraph-scorer.md) |
| Trading / treasury | Calibrated market simulator | Scenario engines validated against stylised facts and replay ([simulators card](model-cards/simulators.md)) |
| Customer ops | Historical case replay + synthetic persona stress tests | Queue and journey simulation with calibration diagnostics |
| Forecasting | MIRAI | Event forecasting with tools |

Benchmark numbers are reported with the benchmark's name and version, never as
bare percentages, and no FinTwinOS result is quoted in these docs until it is
reproducible from a released eval report.

## How `fintwinos eval` maps to the stack

The CLI entry point is:

```bash
fintwinos eval all                       # every suite
fintwinos eval <suite>                   # one suite
fintwinos eval all --report .fintwinos/eval-report
```

which calls `fintwinos.evals.runner.run_suites(suite, report_prefix)` and prints a
summary. Mapping:

- **Suites ↔ layers.** Each evaluation-stack layer is a suite family in
  `fintwinos/evals/`: function-call exactness suites exercise the registry
  directly; agent suites drive `handle_case` over fixed cases with fixed seeds;
  domain suites compute the risk/treasury, compliance and customer-ops metrics
  over replay episodes; the platform suite checks trace completeness and audit
  integrity.
- **Reports ↔ gates.** `run_suites` writes machine-readable reports under the
  `--report` prefix; the release gates above are predicates over those reports, so
  gate decisions are reproducible artefacts (see
  [evidence generation](governance-controls-map.md#evidence-generation)).
- **Offline determinism.** With `FINTWIN_OFFLINE=1` every suite runs on the
  bundled demo twin (`build_runtime(seed=7, with_demo_data=True)`) and synthetic
  datasets with fixed seeds, CI never needs a key, and two runs of the same
  commit produce the same report.
- **Headline experiments.** The four experiments are expressed as paired suites
  (e.g. swarm-on vs swarm-off configurations of the same cases), so their deltas
  appear in the same report format as everything else.

## Operational cadence

- **Every commit:** function-call and agent-run suites, offline, in CI.
- **Every release:** full `fintwinos eval all`, gates evaluated, report archived
  alongside the release.
- **Every recalibration:** domain suites plus the simulate-before-act ablation
  rerun ([calibration runbook](runbooks/calibration.md)).
- **Continuously in shadow:** domain metrics computed on live-parallel decisions,
  feeding the [deployment rule](governance-controls-map.md#the-deployment-rule)
  evidence.

---

FinTwinOS, created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/), MIT License.
