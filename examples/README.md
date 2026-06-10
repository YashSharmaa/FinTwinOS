# FinTwinOS examples

Thin, runnable scripts over the packaged demos. Every one of them works **fully
offline** — no OpenAI key, no network: the stack detects a missing key and
switches to deterministic rule-based fallbacks automatically. Set
`OPENAI_API_KEY` to see the same flows with live LLM enrichment.

| Script | What it shows |
|---|---|
| `01_quickstart.py` | Build the demo twin and print the governed tool catalog (bands, risk tiers, approval requirements). |
| `02_liquidity_stress.py` | USD liquidity squeeze: cash ladder, two stress presets with survival-day confidence intervals, and the rehearsed contingency-funding decision. |
| `03_aml_triage.py` | AML ring surge: trained alert scorer, precision/recall triage trade-off, proposed case narrative, and a case closure that is refused without — and executed with — a human `ApprovalToken`. |
| `04_full_day.py` | All four desks on one shared twin, closing with the governance roll-up: tool calls by band, audit-chain verification, LLM usage and cost. |

```bash
# from the repository root
pip install -e .
python examples/01_quickstart.py
python examples/02_liquidity_stress.py
python examples/03_aml_triage.py
python examples/04_full_day.py

# or via the CLI
fintwinos demo liquidity
fintwinos demo aml_triage
fintwinos demo analyst_research
fintwinos demo customer_ops
fintwinos demo day_in_the_life
```

To force offline mode explicitly (e.g. in CI): `FINTWIN_OFFLINE=1`.

Each demo also exposes `run(offline_ok=True, console=None, seed=7) -> dict`
returning structured results — see `fintwinos/demos/` for the result schemas.

---

FinTwinOS — created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/) · MIT License
