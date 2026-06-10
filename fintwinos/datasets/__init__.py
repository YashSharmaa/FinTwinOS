"""FinTwinOS datasets: seeded synthetic generators, loaders and the dataset registry.

Public surface:

- :mod:`fintwinos.datasets.synthetic` — deterministic generators emitting
  ingestion-ready ``EventEnvelope`` lists with ground-truth label dicts;
- :mod:`fintwinos.datasets.edgar_loader` — the EDGAR connector-cache loader and a
  bundled fictional filing corpus for offline document demos;
- :mod:`fintwinos.datasets.elliptic2_loader` — loader for the Elliptic2 AML graph
  dataset's published CSV layout (data not redistributed; fixture bundled);
- :mod:`fintwinos.datasets.registry` — the ``DATASETS`` registry mapping each name
  to its loader, markdown data card and licence.

Data cards live at the repo level under ``datasets/cards/``; see
``datasets/README.md`` for the index and the privileged-data rule.
"""

from fintwinos.datasets.edgar_loader import load_cached_filings, sample_filing_corpus
from fintwinos.datasets.elliptic2_loader import (
    Elliptic2Dataset,
    elliptic2_fixture_dir,
    load_elliptic2,
)
from fintwinos.datasets.registry import (
    DATASETS,
    DatasetEntry,
    get_dataset,
    list_datasets,
)
from fintwinos.datasets.synthetic import (
    BASE_TIME,
    SyntheticDataset,
    gen_customer_journeys,
    gen_fraud_ring,
    gen_market_paths,
    gen_transactions,
    gen_treasury_ladder,
)

__all__ = [
    "BASE_TIME",
    "DATASETS",
    "DatasetEntry",
    "Elliptic2Dataset",
    "SyntheticDataset",
    "elliptic2_fixture_dir",
    "gen_customer_journeys",
    "gen_fraud_ring",
    "gen_market_paths",
    "gen_transactions",
    "gen_treasury_ladder",
    "get_dataset",
    "list_datasets",
    "load_cached_filings",
    "load_elliptic2",
    "sample_filing_corpus",
]
