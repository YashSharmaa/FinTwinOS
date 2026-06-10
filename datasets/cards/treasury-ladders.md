# Data card — treasury-ladders

## Description

Daily treasury cash-flow ladders per currency. Each currency book starts from an
opening balance (40–160m) and accrues daily contractual inflows and outflows
(lognormal around per-book base levels), with occasional lumpy events — debt
maturities or collateral calls (8% of days, +10–30m outflow) and large asset
maturities (6% of days, +8–20m inflow) — that stress the cumulative position.

Generator: `fintwinos.datasets.synthetic.gen_treasury_ladder(currencies, days, seed)`.

## Schema

- `envelopes`: one `EventEnvelope` per currency-day, `kind="treasury.ladder_slot"`,
  `source="synthetic.treasury_ladder"`, entity `treasury_book:book_<ccy>`, payload
  `currency`, `day`, `date`, `opening_balance`, `inflow`, `outflow`, `net`,
  `cumulative`.
- `labels`: `opening_balance[ccy]`, `cumulative[ccy]` (running position per day),
  `first_negative_day[ccy]` (`null` if the book never goes negative),
  `worst_day[ccy] -> {day, cumulative}`.

## Generation / provenance

Fully synthetic and deterministic per seed (`numpy.random.default_rng(seed)`, fixed
epoch dates). The cumulative series in labels reconciles exactly with the per-day
`net` figures in the payloads (2-decimal rounding applied per day).

## Licence and pass-through terms

MIT (part of FinTwinOS). No real treasury positions or counterparties. No
pass-through terms.

## Intended use

Liquidity-risk demos (survival horizon, cumulative gap), treasury agent testing,
ladder visualisations, and stress-scenario inputs for the treasury simulators.

## Limitations

- Contractual view only: no behavioural run-off modelling, intraday liquidity or
  counterbalancing-capacity actions.
- Currencies are independent; no FX swap funding between books.
- Flow magnitudes are generic and not calibrated to any balance-sheet size or LCR
  regime.
