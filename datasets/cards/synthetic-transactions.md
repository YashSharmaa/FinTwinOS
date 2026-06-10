# Data card — synthetic-transactions

## Description

Seeded synthetic retail payment transactions with money-laundering motifs embedded at
a configurable rate. A fraction `ring_fraction` of transactions belongs to laundering
rings using three classic motifs, rotated in turn:

- **fan-in** — several dedicated source accounts each send one sub-threshold payment
  (structuring, amounts drawn in 8,200–9,850) into a single collector account;
- **fan-out** — one hub disperses sub-threshold payments to several mule accounts;
- **cycle** — funds traverse a closed loop `A1 -> A2 -> ... -> A1`, each hop retaining
  97–99% of the previous amount.

Generator: `fintwinos.datasets.synthetic.gen_transactions(n, ring_fraction, seed)`.

## Schema

Each transaction is one `EventEnvelope`:

| Field | Value |
|---|---|
| `kind` | `transaction.posted` |
| `source` | `synthetic.transactions` |
| `entities` | `account` refs for source and destination |
| `payload` | `transaction_id`, `src_account`, `dst_account`, `amount`, `currency`, `channel` (`wire`/`ach`/`card`/`internal`/`cash_deposit`), `is_cross_border` |

Returned alongside the envelopes:

- `entities`: canonical `Customer` and `Account` lists;
- `graph`: `networkx.MultiDiGraph` of account→account flows, edge attributes
  `txn_id`, `amount`, `is_ring`, `motif`, `ring_id`;
- `labels`: `transactions[txn_id] -> {is_ring, motif, ring_id}`, `rings` (with member
  accounts, hub and transaction ids), `ring_accounts`, plus counts.

## Generation / provenance

Fully synthetic. All randomness flows through `numpy.random.default_rng(seed)`;
timestamps derive from a fixed epoch (`2025-01-06T08:00Z`), identifiers are arithmetic
(`txn_{seed}_{i}`), so identical arguments produce byte-identical envelopes. Ring
accounts are dedicated (never used for background traffic), making the motif ground
truth exact; ring *customers* are drawn from the same attribute distributions as
everyone else so labels do not leak through entity attributes. Labels are returned in
a separate dict and never appear in envelope payloads.

## Licence and pass-through terms

MIT (part of FinTwinOS). The data is fully synthetic: no real persons, accounts,
institutions or transactions. No pass-through terms.

## Intended use

AML detection demos and evals, graph-feature engineering, alert-triage agent testing,
ingestion/replay exercises, and supervised experiments where exact ground truth is
required.

## Limitations

- Motif structure is idealised: real laundering rings overlap with legitimate
  activity, reuse accounts and span institutions; here ring accounts are dedicated.
- Background traffic is independent lognormal pairs with no merchant structure,
  salary cycles or seasonality.
- Amount thresholds reflect a generic 10,000 reporting threshold, not any specific
  jurisdiction's regime.
- Class balance is controlled by `ring_fraction`; real-world prevalence is far lower
  than typical demo settings.
