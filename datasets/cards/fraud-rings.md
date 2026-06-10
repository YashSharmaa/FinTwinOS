# Data card — fraud-rings

## Description

One layered laundering ring generated as a layered DAG: layer 0 holds placement
accounts where illicit cash enters, intermediate layers hold mules, and the final
layer holds integration accounts. Every account forwards its incoming funds to one or
two accounts in the next layer, skimming a 1–3% fee per hop, so value provably flows
from placement to integration and every mule receives funds (no orphans).

Generator: `fintwinos.datasets.synthetic.gen_fraud_ring(size, layers, seed)`.

## Schema

- `graph`: `networkx.DiGraph`; node attributes `layer`, `role`
  (`source` / `mule` / `integrator`), `ring_id`; edge attributes `txn_id`, `amount`,
  `hop`.
- `envelopes`: one `EventEnvelope` per edge, `kind="transaction.posted"`,
  `source="synthetic.fraud_ring"`, payload `transaction_id`, `src_account`,
  `dst_account`, `amount`, `currency`, `channel`, `is_cross_border`.
- `entities`: canonical `Customer` and `Account` lists for all ring members.
- `labels`: `ring_id`, `layers` (layer index → account ids), `roles`
  (account → role), `total_placed`, `total_integrated`, `txn_ids`.

## Generation / provenance

Fully synthetic and deterministic per seed (`numpy.random.default_rng(seed)`, fixed
epoch timestamps, arithmetic identifiers). Accounts are distributed across layers with
extra capacity toward early layers; placement amounts are drawn in 15,000–60,000 per
source. Transfers in layer *l* are timestamped on day *l*, so the layering sequence is
visible in event time.

## Licence and pass-through terms

MIT (part of FinTwinOS). Fully synthetic; no real persons or institutions. No
pass-through terms.

## Intended use

Graph-analytics demos (community/flow detection), network visualisation, case-file
narratives for investigation agents, and unit testing of graph-store ingestion.

## Limitations

- A single isolated ring: no background population to hide in (combine with
  `synthetic-transactions` for in-population detection tasks).
- Flow conservation is exact up to the per-hop fee; real rings show partial
  withdrawals, currency conversion and timing jitter across institutions.
- All transfers are same-currency USD amounts.
