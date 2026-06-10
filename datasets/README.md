# FinTwinOS datasets

Data cards, fixtures and the rules for data handling in FinTwinOS. The Python
loaders and generators live in `fintwinos/datasets/`; this directory is
documentation and small test fixtures only.

## Data cards

Every dataset registered in `fintwinos.datasets.registry.DATASETS` has a card here
describing its schema, generation/provenance, licence (including pass-through
terms), intended use and limitations:

| Dataset | Card | Kind | Licence |
|---|---|---|---|
| `synthetic-transactions` | [cards/synthetic-transactions.md](cards/synthetic-transactions.md) | synthetic | MIT |
| `fraud-rings` | [cards/fraud-rings.md](cards/fraud-rings.md) | synthetic | MIT |
| `journeys` | [cards/journeys.md](cards/journeys.md) | synthetic | MIT |
| `market-paths` | [cards/market-paths.md](cards/market-paths.md) | synthetic | MIT |
| `treasury-ladders` | [cards/treasury-ladders.md](cards/treasury-ladders.md) | synthetic | MIT |
| `edgar-filings` | [cards/edgar-filings.md](cards/edgar-filings.md) | bundled sample + external cache | MIT (sample) / US public domain (filings) |
| `elliptic2` | [cards/elliptic2.md](cards/elliptic2.md) | external loader | Elliptic2's own research-use terms (not redistributed) |

List them programmatically:

```python
from fintwinos.datasets import list_datasets
for entry in list_datasets():
    print(entry["name"], "-", entry["licence"])
```

## Fixtures

- `fixtures/elliptic2_sample/` — a fully synthetic 14-node fixture mirroring the
  Elliptic2 CSV layout so the loader is testable offline. Contains no Elliptic2 data.

## The privileged-data rule

**Privileged, licensed or production data is deployment-local only and is never
committed to this repository.** That includes:

- real customer, account, transaction or case data of any kind;
- cached external corpora (e.g. the EDGAR connector cache) — keep these under your
  deployment's `settings.data_dir` (default `.fintwinos/`), which is local working
  storage;
- datasets with their own licences (e.g. Elliptic2) — download them from the
  official source, keep them outside the repo, and honour their terms.

Only two kinds of data belong in the repository: markdown documentation (these
cards) and small, fully synthetic fixtures written for FinTwinOS. Everything a test
needs must either be synthetic-and-bundled or generated deterministically at test
time from a seed.

---

FinTwinOS is MIT-licensed, created by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/).
