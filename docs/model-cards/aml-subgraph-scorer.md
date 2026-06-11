# Model card: AML subgraph scorer

| | |
|---|---|
| **Artefact** | Graph-based suspicious-pattern scorer for AML alert prioritisation at the *subgraph* level |
| **Modules** | `fintwinos/models/graph/` (feature extraction + scorer), scoring account-transfer subgraphs extracted from the `GraphStore` in `fintwinos/twin_core/`; exercised by the evals harness and tests |
| **Type** | Classical graph model — numpy/networkx feature extraction with a deterministic scoring head; no deep-learning dependency |
| **Version** | Tracks the FinTwinOS release; scoring assumptions versioned alongside simulator `assumptions_version` conventions |
| **Owner** | Compliance domain (model risk sign-off required before any production use) |

## Intended use

- **Prioritise**, never adjudicate: rank open AML alerts so analysts spend their
  minutes on the likeliest-suspicious entity neighbourhoods first.
- Operate at the **subgraph level**, following the framing established by the
  Elliptic2 dataset (2024): money laundering is a pattern over sets of entities
  and flows — rings, layering chains, fan-in/fan-out motifs — not a property of a
  single transaction.
- Feed the compliance agent swarm: scores arrive as `Alert.score` with the
  contributing subgraph features attached for narrative drafting, and analysts see
  *why* a neighbourhood scored high (degree patterns, density, motif counts,
  flow asymmetries), not a bare number.

**Out of scope / misuse — hard lines:**

- **No case-closure automation.** The [release gate](../evaluation.md#the-evaluation-stack)
  is explicit: no closure automation unless recall floors and policy traceability
  are met, and even then closure is a human act through the approval flow.
- No SAR/STR filing decisions; no customer exit decisions; no use as a standalone
  surveillance system. The scorer orders a queue.
- Scores are not risk ratings: `Customer.risk_rating` is a governed attribute and
  is never overwritten by this model.

## Data

- **Public benchmark anchor:** Elliptic2 (2024), the large-scale AML dataset with
  subgraph-level suspicious-pattern labels — used through the eval adapter for
  benchmark reporting ([benchmark plan](../evaluation.md#benchmark-plan)).
- **Bundled synthetic data:** deterministic fraud-ring generators in
  `fintwinos/datasets/` (seeded via `numpy.random.default_rng`) provide offline
  training/testing structure for demos, CI and the `aml_triage` demo
  (`FINTWIN_OFFLINE=1 fintwinos demo aml_triage`).
- **Deployment-local data:** an institution's own transaction/entity graph stays
  in-perimeter (privileged class in the data plan). Lineage requirements apply:
  every ingested edge carries `Provenance`, so a score is traceable to the
  envelopes that built its neighbourhood.
- No customer data is bundled with FinTwinOS, and nothing is sent to an LLM
  provider by the scorer itself (it is a classical model; narrative drafting by
  LLM agents is governed separately by the
  [LLM routing card](llm-routing.md)).

## Metrics and gates

Reported by the compliance eval suite (`fintwinos eval`):

- **Alert precision / recall**, with **recall floors as hard constraints** — a
  recall-floor violation in replay or shadow fails the gate outright, whatever
  the precision gain (missing true laundering is the asymmetric harm).
- **Analyst minutes saved** (queue-ordering efficiency against the incumbent
  ordering) — a soft metric, reported but never traded against the floor.
- **Policy citation accuracy and narrative completeness** for the downstream
  drafting the scores feed.
- Score calibration: ranked buckets are checked against realised confirmation
  rates in replay; miscalibrated buckets trigger refit per the
  [calibration discipline](../runbooks/calibration.md).

No quantitative performance claims are published in these docs until they are
reproducible from a released eval report.

## Limitations

- **Synthetic-to-real gap:** bundled rings are generated; structure learned from
  them transfers imperfectly to real laundering typologies. Deployment requires
  local replay evaluation before the scorer orders a real queue.
- **Subgraph blindness outside the graph:** patterns invisible in the ingested
  entity/flow graph (cash intensity, off-channel relationships) are invisible to
  the scorer. Coverage is bounded by connector coverage.
- **Adversarial adaptation:** launderers probe thresholds; a static scorer
  decays. Drift monitoring on score distributions and confirmation rates is
  mandatory, with refit cadence per the calibration triggers.
- **Poisoning surface:** ingestion poisoning ([threat T1](../threat-model.md#t1--ingestion-poisoning))
  can suppress or inflate neighbourhood scores; provenance checks and episode
  replay are the detection path.
- **Fairness:** neighbourhood features can proxy for geography or segment.
  Fairness guardrails apply to the queue-ordering reward in any learned
  re-ranking ([RL layer](../architecture.md#6-the-rl-layer-staged-bounded-constraint-first)),
  and disparate-impact review belongs in the institution's effective challenge.

## Governance hooks

- The scorer is read-only and side-effect free: it ranks subgraph suspicion and
  never mutates the twin. When surfaced to investigators it belongs in the
  `observe_*`/`propose_*` bands (a documented extension point alongside the
  shipped `observe_entity_graph`/`propose_case_narrative` compliance tools);
  nothing the scorer does can touch a case file without a human.
- Every scoring run that informs a case is auditable: tool calls, the alert's
  entity refs, and the provenance of contributing edges are in the audit chain.
- Status changes on alerts/cases flow through `CaseRecord`/`Alert` state machines
  driven by the agent layer's `handle_case`, with `awaiting_human` as the resting
  state for anything consequential.
- Model changes (features, thresholds, refits) are model-risk events: re-enter
  the [deployment rule](../governance-controls-map.md#the-deployment-rule) at
  replay, with effective-challenge sign-off.

---

FinTwinOS — created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/) — MIT License.
