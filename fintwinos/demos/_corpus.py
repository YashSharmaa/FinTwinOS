"""Sample filing corpus and small text utilities for the research demo.

The analyst-research demo prefers a corpus shipped by the data/connector
modules (probed via :func:`resolve_filing_corpus`); when none is available it
falls back to the bundled corpus below, fictional issuers, realistic Item 1A
risk-factor prose, so the demo and its citations work in any install.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

#: ``(module, attribute)`` pairs probed for a sibling-provided corpus loader.
_CORPUS_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("fintwinos.datasets", "sample_filing_corpus"),
    ("fintwinos.datasets.edgar_loader", "sample_filing_corpus"),
    ("fintwinos.connectors.filings", "sample_filing_corpus"),
    ("fintwinos.twin_core.demo_data", "sample_filing_corpus"),
)

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or our that the
    their this to was we were which with would could may might over under during""".split()
)


def sample_filing_corpus() -> list[dict[str, Any]]:
    """Return the bundled corpus: fictional 10-K risk-factor excerpts.

    Each entry is ``{"doc_id", "text", "metadata"}`` ready for
    ``DocumentStore.add``.
    """
    return [_doc(*row) for row in _RAW_CORPUS]


def resolve_filing_corpus() -> tuple[list[dict[str, Any]], str]:
    """Locate a filing corpus, preferring sibling data modules.

    Returns ``(documents, source)`` where ``source`` names the module that
    provided the corpus. Falls back to :func:`sample_filing_corpus`.
    """
    for module_name, attr in _CORPUS_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        loader = getattr(module, attr, None)
        if not callable(loader):
            continue
        try:
            docs = _normalize_corpus(loader())
        except Exception:
            continue
        if docs:
            return docs, f"{module_name}.{attr}"
    return sample_filing_corpus(), "fintwinos.demos._corpus.sample_filing_corpus"


def _normalize_corpus(raw: Any) -> list[dict[str, Any]]:
    """Coerce assorted corpus shapes into ``{"doc_id", "text", "metadata"}`` dicts.

    Accepts plain dicts (``doc_id``/``id``/``policy_id`` + ``text``/``body``),
    ``(doc_id, text[, metadata])`` tuples, and pydantic-style objects such as
    :class:`~fintwinos.core.types.PolicyDoc`. A ``title`` field, when present,
    is folded into the metadata.
    """
    if not isinstance(raw, list | tuple):
        return []
    docs: list[dict[str, Any]] = []
    for entry in raw:
        doc_id: Any = None
        text: Any = None
        title: Any = None
        metadata: dict[str, Any] = {}
        if isinstance(entry, dict):
            doc_id = entry.get("doc_id") or entry.get("id") or entry.get("policy_id")
            text = entry.get("text") or entry.get("body")
            title = entry.get("title")
            meta = entry.get("metadata")
            metadata = dict(meta) if isinstance(meta, dict) else {}
        elif isinstance(entry, list | tuple) and len(entry) >= 2:
            doc_id, text = entry[0], entry[1]
            if len(entry) >= 3 and isinstance(entry[2], dict):
                metadata = dict(entry[2])
        else:
            doc_id = getattr(entry, "doc_id", None) or getattr(entry, "policy_id", None)
            text = getattr(entry, "text", None) or getattr(entry, "body", None)
            title = getattr(entry, "title", None)
            meta = getattr(entry, "metadata", None)
            metadata = dict(meta) if isinstance(meta, dict) else {}
        if isinstance(title, str) and title and "title" not in metadata:
            metadata["title"] = title
        if isinstance(doc_id, str) and isinstance(text, str) and doc_id and text:
            docs.append({"doc_id": doc_id, "text": text, "metadata": metadata})
    return docs


# ---------------------------------------------------------------------------
# Text utilities (shared by the brief builder and the critic agent)
# ---------------------------------------------------------------------------


def tokens(text: str) -> set[str]:
    """Lowercase content-word token set, with a small stopword list removed."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def first_sentences(text: str, n: int = 2, max_chars: int = 340) -> str:
    """Extract the leading ``n`` sentences, capped at ``max_chars`` characters.

    Markdown-style heading lines are stripped first so excerpts of filings
    stored as markdown read as prose.
    """
    prose = " ".join(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    parts = re.split(r"(?<=[.!?])\s+", prose)
    excerpt = " ".join(parts[:n]).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return excerpt


def lexical_support(claim: str, sources: list[str]) -> float:
    """Fraction of a claim's content tokens found in the union of its sources."""
    claim_tokens = tokens(claim)
    if not claim_tokens:
        return 0.0
    source_tokens: set[str] = set()
    for source in sources:
        source_tokens |= tokens(source)
    return round(len(claim_tokens & source_tokens) / len(claim_tokens), 4)


# ---------------------------------------------------------------------------
# Bundled corpus (fictional issuers)
# ---------------------------------------------------------------------------


def _doc(doc_id: str, issuer: str, topic: str, text: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "text": text,
        "metadata": {
            "issuer": issuer,
            "form": "10-K",
            "section": "Item 1A, Risk Factors",
            "fiscal_year": 2025,
            "topic": topic,
            "synthetic": True,
        },
    }


_RAW_CORPUS: list[tuple[str, str, str, str]] = [
    (
        "fil_meridian_liquidity",
        "Meridian Holdings plc",
        "liquidity_funding",
        "Our liquidity position depends on continued access to wholesale funding markets and "
        "the stability of our corporate deposit base, of which approximately 38% is uninsured "
        "and concentrated among 120 treasury clients. A sustained outflow of uninsured deposits "
        "could compel us to draw on contingency funding sources, including repo of high-quality "
        "liquid assets at unfavourable haircuts and a $600 million committed credit facility "
        "that requires two business days' notice. During the March 2025 market disruption our "
        "30-day liquidity coverage ratio declined from 132% to 118%, and management cannot "
        "assure that comparable stress would not produce a larger decline in future periods.",
    ),
    (
        "fil_meridian_credit",
        "Meridian Holdings plc",
        "credit_concentration",
        "Our loan portfolio is concentrated in commercial real estate, which represented 41% of "
        "total loans at year end, including $2.3 billion of office exposure with weighted "
        "average debt service coverage of 1.21x. Deterioration in occupancy or refinancing "
        "conditions could require material increases to our allowance for credit losses. "
        "Criticised office loans increased 64% year over year, and a downgrade migration of one "
        "rating band across the portfolio would reduce our CET1 ratio by an estimated 80 basis "
        "points.",
    ),
    (
        "fil_northgate_cyber",
        "Northgate Bancorp",
        "cyber_operational",
        "We face persistent cybersecurity threats, including ransomware and supply-chain "
        "compromise of critical vendors. In November 2024 a third-party file-transfer utility "
        "used by our mortgage servicing unit was exploited, resulting in unauthorised access to "
        "records of approximately 94,000 customers and remediation costs of $18 million. A "
        "prolonged outage of our core banking platform, which is operated from two data centres "
        "with a four-hour recovery objective, could materially disrupt payments processing and "
        "damage customer confidence.",
    ),
    (
        "fil_northgate_aml",
        "Northgate Bancorp",
        "financial_crime",
        "In June 2025 we entered into a consent order with our primary regulator relating to "
        "deficiencies in our anti-money laundering programme, including transaction monitoring "
        "coverage of correspondent banking flows and the timeliness of suspicious activity "
        "reporting. The remediation programme requires us to re-screen 1.4 million historical "
        "alerts, enhance customer risk rating models, and retain an independent compliance "
        "monitor through 2027. We expect remediation expense of $45 to $60 million and failure "
        "to satisfy the order could result in growth restrictions or further enforcement "
        "action.",
    ),
    (
        "fil_atlas_intraday",
        "Atlas Clearing Group",
        "liquidity_funding",
        "As a clearing intermediary we are exposed to intraday liquidity risk arising from "
        "settlement timing mismatches across central securities depositories. Peak intraday "
        "usage reached $1.9 billion during the September 2025 quarterly expiry, against "
        "committed intraday lines of $2.4 billion. A failure by a major settlement bank, or "
        "loss of access to those lines, could force us to delay client settlements and could "
        "trigger margin calls from central counterparties at the point of maximum funding "
        "strain.",
    ),
    (
        "fil_harborline_irrbb",
        "Harborline Financial",
        "interest_rate",
        "Rapid changes in interest rates affect the value of our securities portfolio and the "
        "behaviour of our depositors. Unrealised losses on held-to-maturity securities were "
        "$1.1 billion at year end, equal to 27% of tangible common equity, and would be "
        "realised if liquidity needs compelled sales before maturity. Our deposit beta "
        "assumptions, calibrated on the 2022–2024 tightening cycle, may understate the speed of "
        "repricing in a renewed rising-rate environment, compressing net interest margin by up "
        "to 35 basis points per 100 basis point shock.",
    ),
    (
        "fil_harborline_vendor",
        "Harborline Financial",
        "third_party",
        "We rely on a concentrated set of third-party technology providers; our core deposit "
        "platform, card processing and cloud infrastructure are each supplied by a single "
        "vendor under long-term contracts. Termination, degradation or compromise of any of "
        "these arrangements could disrupt customer-facing services for an extended period. "
        "Substitution of our core platform vendor would require an estimated 18 to 24 months "
        "and material conversion expense.",
    ),
    (
        "fil_atlas_climate",
        "Atlas Clearing Group",
        "climate_transition",
        "Transition to a lower-carbon economy may reduce the value of collateral we accept "
        "from energy-sector counterparties and increase margin volatility in commodity "
        "derivatives clearing. Approximately 9% of initial margin held relates to carbon-"
        "intensive underlyings. Regulatory climate stress tests scheduled for 2026 may require "
        "additional capital or changes to our collateral haircut schedules, and disclosure "
        "obligations continue to expand across the jurisdictions in which we operate.",
    ),
]
