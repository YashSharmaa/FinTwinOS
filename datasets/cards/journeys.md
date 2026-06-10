# Data card — journeys

## Description

Multi-channel customer journeys with complaint-escalation paths. Each journey starts
with onboarding stages (`onboarding -> kyc_check -> first_deposit -> regular_usage`);
with calibrated probabilities an issue occurs (45%), a complaint is raised (70% given
an issue), is acknowledged (80%), escalates internally (40% given a complaint) and
reaches the ombudsman (30% given escalation), before resolving or closing unresolved.
Touchpoints carry channels (mobile app, web, branch, call centre, email, chat) with
complaint stages biased toward the call centre, a sentiment score that deteriorates
along the complaint path, and channel-dependent wait times.

Generator: `fintwinos.datasets.synthetic.gen_customer_journeys(n, seed)`.

## Schema

- `envelopes`: one `EventEnvelope` per touchpoint, `kind="journey.touchpoint"`,
  `source="synthetic.journeys"`, payload `journey_id`, `customer_id`, `stage`,
  `detail` (issue type), `channel`, `step_index`, `sentiment` (−1..1),
  `wait_minutes`.
- `entities`: canonical `Customer` list and a `CaseRecord` (kind `complaint`) for
  every complaint journey, with status reflecting escalation/resolution.
- `labels`: `journeys[journey_id] -> {customer_id, had_issue, issue_type, complained,
  escalated, ombudsman, resolved, stages, channels}` plus aggregate counts.

## Generation / provenance

Fully synthetic and deterministic per seed. Stage transitions are sampled from fixed
probabilities via `numpy.random.default_rng(seed)`; timestamps advance 2–96 hours per
touchpoint from a fixed epoch. Ground truth lives only in the labels dict — envelope
payloads contain nothing a production event stream would not.

## Licence and pass-through terms

MIT (part of FinTwinOS). Fully synthetic; customer names are generated from fixed
fictional name lists. No pass-through terms.

## Intended use

Complaint-handling and conduct-risk demos, escalation-prediction experiments,
customer-operations agent testing, and journey-analytics visualisations.

## Limitations

- Stage grammar is linear; real journeys branch, repeat stages and interleave
  concurrent products.
- Sentiment is a deterministic function of stages plus the starting draw, not
  text-derived.
- Escalation probabilities are plausible round numbers, not calibrated to any
  institution's complaint statistics (e.g. FCA complaints data).
