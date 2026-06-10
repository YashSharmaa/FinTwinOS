"""Loaders for SEC EDGAR filing material.

Two entry points:

- :func:`load_cached_filings` reads the connectors' on-disk EDGAR cache (one JSON
  document per filing, written by the EDGAR connector after fetching from the SEC
  APIs) and normalises each filing into a ``PolicyDoc``-like dictionary ready for the
  twin's document store.
- :func:`sample_filing_corpus` returns a bundled, fully fictional corpus of
  MD&A-style risk-factor excerpts so document-store demos, evals and tests run
  completely offline with zero network access and zero real-issuer text.

Cache format
------------

The cache directory contains one ``*.json`` file per filing (an ``index.json`` with a
``{"filings": [...]}`` list is also accepted). Each filing object carries::

    {
      "accession_no": "0001234567-24-000042",   # or accession_number / accession / id
      "cik": "0001234567",
      "company": "Example Corp",                # or company_name / issuer
      "form_type": "10-K",                      # or form
      "filed_at": "2024-02-15",                 # or filing_date / date_filed
      "title": "...",                           # optional
      "text": "...",                            # or body / content, or
      "sections": {"Risk Factors": "...", ...}, # section name -> text
      "url": "https://www.sec.gov/..."          # optional
    }

Key aliases are tolerated so the loader stays compatible as the connector evolves.
Real EDGAR text is public-domain US government material, but cached filings are
deployment-local working data: keep them under ``settings.data_dir`` and never commit
them to the repository (see ``datasets/README.md``).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Tags attached to every loaded filing document.
_BASE_TAGS = ["edgar", "filing"]


def _first(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-empty value among aliased keys."""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def _parse_filed_date(raw: Any) -> datetime | None:
    """Parse a filing date string into an aware UTC datetime (None if unparseable)."""
    if raw in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _body_from(record: dict[str, Any]) -> str:
    """Assemble the document body from ``text``-style keys or a ``sections`` dict."""
    text = _first(record, "text", "body", "content")
    if isinstance(text, str) and text.strip():
        return text.strip()
    sections = record.get("sections")
    if isinstance(sections, dict) and sections:
        parts = [f"## {name}\n\n{str(content).strip()}" for name, content in sections.items()]
        return "\n\n".join(parts)
    return ""


def _normalise_filing(record: dict[str, Any]) -> dict[str, Any] | None:
    """Map one cached filing record to a ``PolicyDoc``-like dict (None if not a filing)."""
    body = _body_from(record)
    if not body:
        return None
    accession = str(
        _first(record, "accession_no", "accession_number", "accession", "id", default="unknown")
    )
    company = str(_first(record, "company", "company_name", "issuer", default="Unknown Issuer"))
    form_type = str(_first(record, "form_type", "form", default="UNKNOWN"))
    filed_raw = _first(record, "filed_at", "filing_date", "date_filed")
    effective = _parse_filed_date(filed_raw)
    slug = re.sub(r"[^A-Za-z0-9]+", "", accession).lower() or "unknown"
    title = _first(record, "title")
    if not title:
        filed_str = effective.date().isoformat() if effective else "undated"
        title = f"{company} {form_type} ({filed_str})"
    tags = [*_BASE_TAGS, form_type.lower()]
    cik = _first(record, "cik")
    return {
        "policy_id": f"edgar_{slug}",
        "title": str(title),
        "body": body,
        "tags": tags,
        "version": "1.0",
        "effective_date": effective,
        "metadata": {
            "source": "sec-edgar",
            "accession_no": accession,
            "cik": str(cik) if cik is not None else None,
            "company": company,
            "form_type": form_type,
            "url": _first(record, "url"),
        },
    }


def load_cached_filings(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load every filing from a connectors' EDGAR cache directory.

    Accepts per-filing ``*.json`` files, JSON files containing a list of filings, and
    an ``index.json`` with a ``{"filings": [...]}`` envelope. Files that do not parse
    as JSON or that carry no filing text are skipped silently, so the loader tolerates
    partial caches and unrelated artefacts side by side.

    Args:
        data_dir: Directory the EDGAR connector caches into.

    Returns:
        ``PolicyDoc``-like dictionaries (``policy_id``, ``title``, ``body``, ``tags``,
        ``version``, ``effective_date``, plus a ``metadata`` block), sorted by filing
        date then ``policy_id`` for stable, reproducible ordering. Each dict validates
        with ``PolicyDoc.model_validate`` (extra keys are ignored by pydantic).

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist, with a pointer to the
            EDGAR connector that populates it.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"EDGAR cache directory not found: {directory}. Run the EDGAR connector "
            "(fintwinos.connectors) to populate a deployment-local cache under "
            "settings.data_dir, or use sample_filing_corpus() for a bundled offline corpus."
        )

    docs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(raw, dict) and isinstance(raw.get("filings"), list):
            records: list[Any] = raw["filings"]
        elif isinstance(raw, list):
            records = raw
        else:
            records = [raw]
        for record in records:
            if not isinstance(record, dict):
                continue
            doc = _normalise_filing(record)
            if doc is not None:
                docs.append(doc)

    docs.sort(key=lambda d: (d["effective_date"] or datetime.min.replace(tzinfo=UTC),
                             d["policy_id"]))
    return docs


# ---------------------------------------------------------------------------
# Bundled sample corpus — entirely fictional issuers, written for FinTwinOS
# ---------------------------------------------------------------------------

#: Fictional filing excerpts in the connectors' cache record format. Every issuer,
#: CIK and accession number below is invented; any resemblance to a real company is
#: coincidental. The text was written for FinTwinOS and ships under the MIT licence.
_SAMPLE_FILINGS: list[dict[str, Any]] = [
    {
        "accession_no": "0009100001-24-000011",
        "cik": "0009100001",
        "company": "Meridian Crest Bancorp",
        "form_type": "10-K",
        "filed_at": "2024-02-23",
        "sections": {
            "Risk Factors": (
                "Our deposit base is concentrated among small and mid-sized commercial "
                "customers in three metropolitan markets, and a significant portion of those "
                "balances exceeds insured limits. A loss of confidence among these depositors, "
                "whether triggered by events at the Company or by adverse developments at "
                "unrelated regional institutions, could produce rapid and correlated outflows "
                "that exceed our on-balance-sheet liquidity. While we maintain contingent "
                "funding capacity, including secured borrowing lines and a portfolio of "
                "high-quality liquid assets, the speed at which deposits can now move through "
                "digital channels may compress the time available to mobilise that capacity. "
                "In addition, a sustained inversion of the yield curve continues to pressure "
                "our net interest margin: a material share of our commercial real estate book "
                "reprices over the next eighteen months into a higher-rate environment, and "
                "borrower debt-service coverage ratios in the office segment have weakened. "
                "If criticised assets migrate faster than our current allowance assumptions "
                "contemplate, we may be required to increase provisions for credit losses, "
                "which would adversely affect our results of operations and regulatory "
                "capital ratios."
            ),
        },
        "url": "https://example.invalid/edgar/meridian-crest-10k",
    },
    {
        "accession_no": "0009100002-24-000027",
        "cik": "0009100002",
        "company": "Aurelia Payments Group",
        "form_type": "10-Q",
        "filed_at": "2024-05-09",
        "sections": {
            "Management's Discussion and Analysis": (
                "Total payment volume grew 14% year over year, driven by cross-border "
                "e-commerce corridors, while take rate declined three basis points as larger "
                "merchants migrated onto interchange-plus pricing. We continue to invest in "
                "our real-time fraud decisioning platform; model retraining cadence moved "
                "from weekly to daily during the quarter, and false-positive rates on card- "
                "not-present transactions improved 11%. These gains may not be sustainable "
                "as fraud patterns adapt. Our exposure to merchant insolvency risk increased "
                "with the expansion of delayed-delivery verticals such as travel and events: "
                "where a merchant fails before delivering goods, chargebacks may exceed the "
                "collateral we hold. We recognised a $9.3 million provision for merchant "
                "losses, up from $5.1 million in the prior-year quarter, principally related "
                "to a single tour operator placed into administration. Regulatory scrutiny "
                "of interchange, surcharging and buy-now-pay-later products continues to "
                "intensify across our principal markets, and adverse rulemaking could "
                "compress unit economics in future periods."
            ),
        },
        "url": "https://example.invalid/edgar/aurelia-payments-10q",
    },
    {
        "accession_no": "0009100003-24-000005",
        "cik": "0009100003",
        "company": "Northgate Custody Trust",
        "form_type": "10-K",
        "filed_at": "2024-03-01",
        "sections": {
            "Risk Factors": (
                "As a custodian for institutional clients we safekeep assets that are many "
                "multiples of our balance sheet, and our operational risk profile is "
                "dominated by settlement, corporate-action processing and cyber threats "
                "rather than credit exposure. A prolonged outage of our core settlement "
                "platform, whether caused by internal change-management failure or by a "
                "successful intrusion, could trigger client compensation claims, regulatory "
                "enforcement and accelerated client attrition. We rely on a small number of "
                "third-party providers for market data, messaging and cloud infrastructure; "
                "the failure of any of them, or the imposition of sanctions affecting a "
                "sub-custodian in our network, could interrupt service in affected markets. "
                "The transition of remaining client portfolios to T+1 settlement has reduced "
                "the time available to resolve exceptions, increasing the operational cost "
                "of fails. Our fee income is also sensitive to equity market levels: "
                "approximately 62% of assets-under-custody fees are charged on market "
                "values, so a sustained decline in global equity markets would directly "
                "reduce revenue without a commensurate reduction in our largely fixed "
                "operating cost base."
            ),
        },
        "url": "https://example.invalid/edgar/northgate-custody-10k",
    },
    {
        "accession_no": "0009100004-23-000064",
        "cik": "0009100004",
        "company": "Solent Maritime Finance plc",
        "form_type": "20-F",
        "filed_at": "2023-11-17",
        "sections": {
            "Risk Factors": (
                "We provide secured lending against ocean-going vessels, and the value of "
                "our collateral is exposed to charter-rate cycles that have historically "
                "been both deep and abrupt. A sustained downturn in dry-bulk or container "
                "charter markets would reduce vessel values and could leave portions of our "
                "loan book under-collateralised, particularly tranches originated near the "
                "2021–2022 cyclical peak. Environmental regulation presents a structural "
                "risk: tightening carbon-intensity rules may render older tonnage "
                "uneconomic faster than our residual-value assumptions anticipate, and "
                "borrowers may be unable to fund required retrofits. We are also exposed to "
                "sanctions and dark-fleet risk; although we screen counterparties and "
                "vessel movements against recognised databases, ship-to-ship transfers and "
                "flag-hopping techniques are designed to evade such screening, and an "
                "undetected breach involving a financed vessel could expose us to "
                "enforcement action, lender liability and reputational damage. Our funding "
                "is predominantly wholesale and denominated in US dollars, while a growing "
                "share of our lending is in other currencies, creating refinancing and "
                "cross-currency basis risk in stressed markets."
            ),
        },
        "url": "https://example.invalid/edgar/solent-maritime-20f",
    },
    {
        "accession_no": "0009100005-24-000018",
        "cik": "0009100005",
        "company": "Vantorre Asset Management",
        "form_type": "10-K",
        "filed_at": "2024-02-28",
        "sections": {
            "Risk Factors": (
                "A substantial majority of our revenue consists of management fees "
                "calculated as a percentage of assets under management, and 41% of AUM is "
                "concentrated in five institutional relationships. The redemption of any of "
                "these mandates would materially reduce revenue. Performance fees, which "
                "represented 18% of revenue in the most recent fiscal year, are inherently "
                "volatile and depend on exceeding high-water marks in our absolute-return "
                "strategies. Our expansion into semi-liquid private credit vehicles exposes "
                "us to liquidity mismatch risk: although the vehicles impose quarterly "
                "gates, a sustained period of elevated redemption requests could compel "
                "asset sales at unfavourable prices or reputational harm from gate "
                "imposition. We use quantitative models in portfolio construction and risk "
                "management; errors in model design, implementation or data inputs have in "
                "the past produced, and may in the future produce, unintended exposures. "
                "Increasing regulatory expectations regarding sustainability disclosures "
                "across jurisdictions create compliance complexity, and any finding that "
                "our marketing materials overstated environmental characteristics of our "
                "products could result in penalties and client losses."
            ),
        },
        "url": "https://example.invalid/edgar/vantorre-am-10k",
    },
    {
        "accession_no": "0009100006-24-000033",
        "cik": "0009100006",
        "company": "Kestrel Digital Markets Inc.",
        "form_type": "10-Q",
        "filed_at": "2024-08-08",
        "sections": {
            "Management's Discussion and Analysis": (
                "Average daily trading volume on our electronic fixed-income venues rose "
                "22% versus the prior-year quarter, with credit products accounting for "
                "most of the growth as portfolio trading adoption broadened. Net revenue "
                "per million traded declined modestly due to mix shift toward larger, "
                "lower-fee protocols. During the quarter we completed the migration of our "
                "matching engines to a new data-centre region; the migration was executed "
                "without client-facing downtime, although we recorded $4.2 million of "
                "duplicative infrastructure costs that will not recur. Clearing and "
                "settlement of our digital-asset pilot programme remains dependent on a "
                "single regulated custodian, and the programme's expansion is contingent "
                "on regulatory clarity that may not materialise on the timeline we "
                "anticipate. We note increasing concentration of liquidity provision among "
                "a small number of principal trading firms on our venues: the withdrawal "
                "of one or more of these firms during stressed conditions could widen "
                "spreads, reduce volumes and impair the perceived reliability of our "
                "markets, as occurred briefly during the March volatility event described "
                "in Note 14."
            ),
        },
        "url": "https://example.invalid/edgar/kestrel-digital-10q",
    },
    {
        "accession_no": "0009100007-24-000002",
        "cik": "0009100007",
        "company": "Bryndle & Howe Insurance Holdings",
        "form_type": "10-K",
        "filed_at": "2024-02-16",
        "sections": {
            "Risk Factors": (
                "Our specialty property book is geographically concentrated in coastal "
                "regions with elevated and rising catastrophe exposure. Although we "
                "purchase reinsurance to limit net losses, reinsurance pricing hardened "
                "materially at the most recent renewal, retention levels increased, and "
                "certain perils, notably severe convective storm, are now subject to "
                "tighter occurrence definitions. Model uncertainty is significant: the "
                "vendor catastrophe models we rely upon have historically under-estimated "
                "losses from secondary perils, and our own adjustments may prove "
                "insufficient. In our casualty lines, social inflation continues to drive "
                "settlement values above our booked reserve assumptions; we strengthened "
                "prior-year reserves by $112 million during the year, principally in "
                "commercial auto and excess liability, and further adverse development is "
                "possible. Our investment portfolio includes $640 million of commercial "
                "mortgage loans and CMBS with meaningful office exposure, where refinancing "
                "risk at maturity remains elevated. Finally, our reliance on managing "
                "general agents for 28% of gross written premium exposes us to "
                "underwriting decisions made outside our direct control, and termination "
                "or impairment of key MGA relationships could disrupt premium volume."
            ),
        },
        "url": "https://example.invalid/edgar/bryndle-howe-10k",
    },
]


def sample_filing_corpus() -> list[dict[str, Any]]:
    """Return the bundled offline corpus of fictional filing excerpts.

    The corpus contains seven MD&A / risk-factor style excerpts for fictional
    financial issuers (a regional bank, a payments processor, a custodian, a ship
    financier, an asset manager, an electronic trading venue and a specialty
    insurer). It exists so document-store demos, retrieval evals and tests can run
    fully offline without fetching or redistributing any real EDGAR text.

    Returns:
        ``PolicyDoc``-like dictionaries in the same shape as
        :func:`load_cached_filings` output, sorted by filing date then ``policy_id``.
        Deterministic: repeated calls return equal values (fresh copies each call, so
        callers may mutate safely).
    """
    docs = [_normalise_filing(dict(record)) for record in _SAMPLE_FILINGS]
    out = [doc for doc in docs if doc is not None]
    out.sort(key=lambda d: (d["effective_date"] or datetime.min.replace(tzinfo=UTC),
                            d["policy_id"]))
    return out
