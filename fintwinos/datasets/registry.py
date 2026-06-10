"""Dataset registry: name -> (loader, data card, licence).

Every dataset that FinTwinOS ships, generates or knows how to load is registered
here, with a pointer to its markdown data card under the repo-level
``datasets/cards/`` directory and its licence (including pass-through terms for
external datasets). Demos, evals and the CLI discover datasets through
:func:`list_datasets` rather than importing loader modules directly.

Data cards live outside the Python package (they are repo documentation, not code),
so ``card_exists`` in :func:`list_datasets` output reports whether the card file is
present — it will be ``True`` in a repository checkout and may be ``False`` for a
bare wheel installation, which is expected and harmless.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fintwinos.datasets.edgar_loader import load_cached_filings, sample_filing_corpus
from fintwinos.datasets.elliptic2_loader import load_elliptic2
from fintwinos.datasets.synthetic import (
    SYNTHETIC_LICENCE,
    gen_customer_journeys,
    gen_fraud_ring,
    gen_market_paths,
    gen_transactions,
    gen_treasury_ladder,
)

#: Repo-level directory holding the markdown data cards.
CARDS_DIR = Path(__file__).resolve().parents[2] / "datasets" / "cards"


@dataclass(frozen=True)
class DatasetEntry:
    """One registered dataset.

    Attributes:
        name: Registry key, also the data-card filename stem.
        loader: Callable producing the dataset. Synthetic generators take
            ``(..., seed=...)``; external loaders take a ``data_dir``.
        data_card: Absolute path to the markdown data card.
        licence: Licence of the data this entry produces or loads, including
            pass-through terms for external datasets.
        description: One-line description for catalogues and the CLI.
        kind: ``"synthetic"``, ``"bundled"`` or ``"external"``.
    """

    name: str
    loader: Callable[..., Any]
    data_card: Path
    licence: str
    description: str
    kind: str = "synthetic"


def _card(name: str) -> Path:
    return CARDS_DIR / f"{name}.md"


DATASETS: dict[str, DatasetEntry] = {
    entry.name: entry
    for entry in [
        DatasetEntry(
            name="synthetic-transactions",
            loader=gen_transactions,
            data_card=_card("synthetic-transactions"),
            licence=SYNTHETIC_LICENCE,
            description=(
                "Retail transactions with embedded laundering motifs (fan-in, fan-out, "
                "cycles) and exact ground-truth labels."
            ),
            kind="synthetic",
        ),
        DatasetEntry(
            name="fraud-rings",
            loader=gen_fraud_ring,
            data_card=_card("fraud-rings"),
            licence=SYNTHETIC_LICENCE,
            description=(
                "A layered laundering ring (placement -> mules -> integration) as a "
                "networkx DAG plus transfer envelopes."
            ),
            kind="synthetic",
        ),
        DatasetEntry(
            name="journeys",
            loader=gen_customer_journeys,
            data_card=_card("journeys"),
            licence=SYNTHETIC_LICENCE,
            description=(
                "Multi-channel customer journeys with complaint and escalation paths, "
                "plus complaint CaseRecords."
            ),
            kind="synthetic",
        ),
        DatasetEntry(
            name="market-paths",
            loader=gen_market_paths,
            data_card=_card("market-paths"),
            licence=SYNTHETIC_LICENCE,
            description=(
                "Daily price paths: GBM drift, Poisson jumps and GARCH(1,1) volatility "
                "clustering, with full parameter ground truth."
            ),
            kind="synthetic",
        ),
        DatasetEntry(
            name="treasury-ladders",
            loader=gen_treasury_ladder,
            data_card=_card("treasury-ladders"),
            licence=SYNTHETIC_LICENCE,
            description=(
                "Daily treasury cash-flow ladders per currency with lumpy maturities "
                "and cumulative-position ground truth."
            ),
            kind="synthetic",
        ),
        DatasetEntry(
            name="edgar-filings",
            loader=sample_filing_corpus,
            data_card=_card("edgar-filings"),
            licence=(
                "Bundled sample corpus: MIT (fictional issuers, written for FinTwinOS). "
                "Real cached filings via load_cached_filings(): US public-domain SEC "
                "EDGAR material; observe SEC fair-access rules when fetching."
            ),
            description=(
                "Bundled fictional MD&A/risk-factor excerpts for offline document demos; "
                "load_cached_filings() reads the EDGAR connector cache."
            ),
            kind="bundled",
        ),
        DatasetEntry(
            name="elliptic2",
            loader=load_elliptic2,
            data_card=_card("elliptic2"),
            licence=(
                "Not redistributed. Elliptic2 is released by Elliptic / MIT-IBM Watson "
                "AI Lab for research use under its own terms — download from "
                "https://github.com/MITIBMxGraph/Elliptic2 and review the licence there."
            ),
            description=(
                "Loader for the Elliptic2 AML graph dataset (nodes/edges/component "
                "labels CSVs); synthetic same-layout fixture bundled for tests."
            ),
            kind="external",
        ),
    ]
}

# load_cached_filings is part of the registry surface even though the default
# edgar-filings loader is the offline sample corpus.
__all__ = [
    "CARDS_DIR",
    "DATASETS",
    "DatasetEntry",
    "get_dataset",
    "list_datasets",
    "load_cached_filings",
]


def get_dataset(name: str) -> DatasetEntry:
    """Look up a dataset entry by name.

    Raises:
        KeyError: With the list of known names when ``name`` is not registered.
    """
    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset '{name}'; known datasets: {known}") from None


def list_datasets() -> list[dict[str, Any]]:
    """List all registered datasets with card and licence metadata.

    Returns:
        One dict per dataset, sorted by name, with keys ``name``, ``description``,
        ``kind``, ``licence``, ``data_card`` (absolute path as a string),
        ``card_exists`` and ``loader`` (the dotted callable name).
    """
    out: list[dict[str, Any]] = []
    for name in sorted(DATASETS):
        entry = DATASETS[name]
        out.append(
            {
                "name": entry.name,
                "description": entry.description,
                "kind": entry.kind,
                "licence": entry.licence,
                "data_card": str(entry.data_card),
                "card_exists": entry.data_card.is_file(),
                "loader": f"{entry.loader.__module__}.{entry.loader.__qualname__}",
            }
        )
    return out
