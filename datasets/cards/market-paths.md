# Data card — market-paths

## Description

Daily synthetic price paths for fictional instruments (`SYN01`, `SYN02`, ...)
combining geometric-Brownian drift, Poisson jumps and GARCH(1,1) volatility
clustering, implemented directly in numpy:

```
r_t     = mu*dt - v_t/2 + sqrt(v_t) * z_t + J_t
v_{t+1} = omega + alpha * (r_t - mu*dt)^2 + beta * v_t
```

with `z_t ~ N(0,1)`, jump counts Poisson at 2–6 per year, negative-mean jump sizes,
and `omega` set so long-run variance matches each instrument's drawn annual
volatility (12–35%). Prices start at 100.

Generator: `fintwinos.datasets.synthetic.gen_market_paths(n_instruments, n_steps, seed)`.

## Schema

- `envelopes`: one `EventEnvelope` per instrument per step, `kind="market.bar"`,
  `source="synthetic.market_paths"`, payload `symbol`, `step`, `date`, `close`,
  `log_return`.
- `entities`: canonical `Instrument` list (`asset_class="equity"`, USD).
- `labels`: `params[symbol] -> {mu, sigma_annual, alpha, beta,
  jump_intensity_per_year}`, `jump_steps[symbol]` (ground-truth jump dates),
  `prices[symbol]`, `realised_vol_annual[symbol]`.

## Generation / provenance

Fully synthetic and deterministic per seed (`numpy.random.default_rng(seed)`).
Parameters per instrument are drawn once, then paths are simulated with a vectorised
variance recursion. Jump indicators are ground truth held in labels only — payloads
expose just the observable bar (close, log return), as a market feed would.

## Licence and pass-through terms

MIT (part of FinTwinOS). Tickers are synthetic placeholders and do not reference real
listed instruments. No pass-through terms.

## Intended use

Market-risk simulator calibration tests, VaR/stress demo inputs, volatility-regime
detection experiments, and time-series store ingestion exercises.

## Limitations

- Single daily frequency; no intraday microstructure, spreads or volume.
- Instruments are mutually independent — no cross-sectional correlation matrix or
  common factors.
- GARCH(1,1) plus compound-Poisson jumps is a simplification; no leverage effect,
  term structure or regime switching.
