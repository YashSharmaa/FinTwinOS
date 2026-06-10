"""SEC EDGAR connector: filings index and XBRL company facts, offline-first.

``EdgarConnector`` speaks to two public SEC endpoints:

- the submissions API ``https://data.sec.gov/submissions/CIK{cik:0>10}.json``,
  which lists an entity's recent filings as parallel arrays; and
- the companyfacts XBRL API
  ``https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:0>10}.json``.

All network access goes through a single :meth:`EdgarConnector.fetch` that

1. serves from a local JSON cache under ``settings.data_dir / "edgar"`` when a
   cached copy exists (so repeat calls are fully offline);
2. raises :class:`EdgarOfflineError` with a precise remediation message when the
   platform is offline (``FINTWIN_OFFLINE=1``) and no cache exists; and
3. otherwise performs a polite HTTPS GET with a descriptive User-Agent (the SEC
   requires one), read from ``FINTWIN_EDGAR_USER_AGENT`` or a sensible default,
   and writes the response to the cache atomically.

The stream emits one ``filing.indexed`` envelope per filing, with a payload shaped
for the twin's document store: ``doc_id``, ``title``, ``text``, ``url`` and a
``metadata`` block.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from fintwinos.connectors.base import BaseConnector, ConnectorError
from fintwinos.core.config import Settings, get_settings
from fintwinos.core.types import EntityRef, EventEnvelope, Provenance

DEFAULT_EDGAR_USER_AGENT = "FinTwinOS research contact@example.com"
EDGAR_USER_AGENT_ENV = "FINTWIN_EDGAR_USER_AGENT"

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{document}"

_RECENT_FIELDS = (
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "act",
    "form",
    "fileNumber",
    "items",
    "size",
    "isXBRL",
    "isInlineXBRL",
    "primaryDocument",
    "primaryDocDescription",
)


class EdgarOfflineError(ConnectorError):
    """Raised when EDGAR data is needed offline and no local cache exists."""


def normalise_cik(cik: int | str) -> str:
    """Normalise any CIK spelling to the canonical zero-padded 10-digit form.

    Accepts ints, digit strings, and ``CIK``-prefixed strings (case-insensitive).

    Raises:
        ConnectorError: If the value is not a valid CIK.
    """
    text = str(cik).strip()
    if text.upper().startswith("CIK"):
        text = text[3:].strip()
    if not text.isdigit():
        raise ConnectorError(f"invalid CIK {cik!r}: expected digits, optionally 'CIK'-prefixed")
    return f"{int(text):010d}"


def iter_recent_filings(submissions: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one dict per filing from a submissions payload's parallel arrays.

    The SEC encodes ``filings.recent`` as a dict of equal-length lists; this
    re-zips them into per-filing dicts keyed by the standard field names. Short or
    missing arrays yield ``None`` for the affected fields rather than failing.
    """
    recent = (submissions.get("filings") or {}).get("recent") or {}
    accessions = recent.get("accessionNumber") or []
    for index in range(len(accessions)):
        filing: dict[str, Any] = {}
        for field in _RECENT_FIELDS:
            values = recent.get(field) or []
            filing[field] = values[index] if index < len(values) else None
        yield filing


def fact_series(
    facts: dict[str, Any],
    tag: str,
    *,
    taxonomy: str = "us-gaap",
    unit: str | None = None,
) -> list[tuple[str, float]]:
    """Extract a ``(end_date, value)`` series for one concept from companyfacts.

    Args:
        facts: A parsed companyfacts payload (see
            :meth:`EdgarConnector.company_facts`).
        tag: The XBRL concept tag, e.g. ``"Revenues"`` or
            ``"RevenueFromContractWithCustomerExcludingAssessedTax"``.
        taxonomy: The taxonomy namespace (default ``"us-gaap"``).
        unit: The unit to read (e.g. ``"USD"``). ``None`` picks the first unit in
            sorted order.

    Returns:
        ``(end_date, value)`` pairs sorted by end date. Restated periods (the same
        end date reported in multiple filings) are deduplicated keeping the
        last-listed observation, which the SEC orders by filing date.
    """
    concept = ((facts.get("facts") or {}).get(taxonomy) or {}).get(tag) or {}
    units = concept.get("units") or {}
    if not units:
        return []
    if unit is None:
        unit = sorted(units)[0]
    by_end: dict[str, float] = {}
    for item in units.get(unit) or []:
        end = item.get("end")
        value = item.get("val")
        if end is None or value is None:
            continue
        by_end[str(end)] = float(value)
    return sorted(by_end.items())


class EdgarConnector(BaseConnector):
    """Stream SEC EDGAR filing indexes as ``filing.indexed`` envelopes.

    Args:
        ciks: One CIK or a sequence of CIKs, in any accepted spelling
            (``320193``, ``"320193"``, ``"CIK0000320193"``).
        forms: Optional whitelist of form types (e.g. ``["10-K", "10-Q"]``),
            matched case-insensitively. ``None`` streams every filing.
        limit_per_cik: Optional cap on filings emitted per CIK (the submissions
            API lists most-recent first).
        name: Connector name (default ``"edgar"``).
        settings: Settings supplying ``data_dir`` (cache location), ``offline``
            and ``request_timeout``.
        user_agent: Override for the polite User-Agent header; defaults to the
            ``FINTWIN_EDGAR_USER_AGENT`` environment variable, then
            ``"FinTwinOS research contact@example.com"``.
        force_refresh: When true, bypass the cache for reads (responses are still
            written back). Has no effect offline.
    """

    def __init__(
        self,
        ciks: int | str | Sequence[int | str],
        *,
        forms: Sequence[str] | None = None,
        limit_per_cik: int | None = None,
        name: str = "edgar",
        settings: Settings | None = None,
        user_agent: str | None = None,
        force_refresh: bool = False,
    ):
        if isinstance(ciks, int | str):
            ciks = [ciks]
        self.ciks = [normalise_cik(c) for c in ciks]
        self.forms = {f.strip().upper() for f in forms} if forms is not None else None
        self.limit_per_cik = limit_per_cik
        self.settings = settings or get_settings()
        self.user_agent = (
            user_agent
            or os.environ.get(EDGAR_USER_AGENT_ENV)
            or DEFAULT_EDGAR_USER_AGENT
        )
        self.force_refresh = force_refresh
        super().__init__(name=name, source="sec_edgar")

    # -- cache & fetch ---------------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        """Directory holding cached EDGAR JSON responses."""
        return self.settings.data_dir / "edgar"

    def cache_path(self, kind: str, cik: int | str) -> Path:
        """Cache file for one ``(kind, cik)`` pair, e.g. ``("submissions", 320193)``."""
        return self.cache_dir / f"{kind}_CIK{normalise_cik(cik)}.json"

    async def fetch(self, url: str, cache_key: str) -> dict[str, Any]:
        """Fetch one EDGAR JSON document through the local cache.

        Order of precedence: cached copy (unless ``force_refresh``); offline
        error when the platform is offline with no cache; polite network GET,
        cached atomically on success.

        Args:
            url: The fully-formed EDGAR URL.
            cache_key: Cache file stem (the file is ``<cache_key>.json`` under
                :attr:`cache_dir`).

        Raises:
            EdgarOfflineError: Offline with no cached copy.
            ConnectorError: HTTP errors or non-JSON responses.
        """
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists() and not self.force_refresh:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConnectorError(f"corrupt EDGAR cache file {path}: {exc}") from exc

        if self.settings.offline:
            raise EdgarOfflineError(
                f"FinTwinOS is offline (FINTWIN_OFFLINE=1) and no cached copy of {url} "
                f"exists at {path}. Run once online to prime the cache, or place a "
                "fixture JSON file at that path."
            )

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.request_timeout, headers=headers
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise ConnectorError(
                f"EDGAR returned HTTP {exc.response.status_code} for {url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectorError(f"network error fetching {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConnectorError(f"EDGAR returned non-JSON content for {url}") from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
        return data

    async def submissions(self, cik: int | str) -> dict[str, Any]:
        """Fetch (or read from cache) the submissions payload for one CIK."""
        cik10 = normalise_cik(cik)
        return await self.fetch(
            SUBMISSIONS_URL.format(cik10=cik10), f"submissions_CIK{cik10}"
        )

    async def company_facts(self, cik: int | str) -> dict[str, Any]:
        """Fetch (or read from cache) the companyfacts XBRL payload for one CIK."""
        cik10 = normalise_cik(cik)
        return await self.fetch(
            COMPANYFACTS_URL.format(cik10=cik10), f"companyfacts_CIK{cik10}"
        )

    # -- mapping ------------------------------------------------------------------

    def _filing_envelope(
        self, cik10: str, submissions: dict[str, Any], filing: dict[str, Any]
    ) -> EventEnvelope:
        """Map one filing row (from :func:`iter_recent_filings`) to an envelope."""
        company = submissions.get("name") or f"CIK{cik10}"
        accession = filing.get("accessionNumber") or ""
        accession_nodash = accession.replace("-", "")
        form = filing.get("form") or "UNKNOWN"
        filing_date = filing.get("filingDate")
        report_date = filing.get("reportDate")
        primary_document = filing.get("primaryDocument")
        description = filing.get("primaryDocDescription")
        items = filing.get("items")

        url: str | None = None
        if accession_nodash and primary_document:
            url = ARCHIVES_URL.format(
                cik_int=int(cik10), accession_nodash=accession_nodash,
                document=primary_document,
            )

        occurred_at: datetime | None = None
        if filing_date:
            try:
                occurred_at = datetime.strptime(str(filing_date), "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError:
                occurred_at = None
        if occurred_at is None and filing.get("acceptanceDateTime"):
            try:
                occurred_at = datetime.fromisoformat(
                    str(filing["acceptanceDateTime"]).replace("Z", "+00:00")
                )
            except ValueError:
                occurred_at = None

        text_parts = [f"{company} filed form {form}"]
        if filing_date:
            text_parts.append(f"on {filing_date}")
        if description:
            text_parts.append(f"— {description}")
        if report_date:
            text_parts.append(f"(report period ending {report_date})")
        if items:
            text_parts.append(f"Items: {items}.")
        text = " ".join(text_parts).strip()

        entities = [EntityRef(entity_type="legal_entity", entity_id=f"CIK{cik10}")]
        for ticker in submissions.get("tickers") or []:
            entities.append(EntityRef(entity_type="instrument", entity_id=str(ticker)))

        payload: dict[str, Any] = {
            "doc_id": f"edgar:{cik10}:{accession}",
            "title": f"{company} {form} {filing_date or ''}".strip(),
            "text": text,
            "url": url,
            "metadata": {
                "cik": cik10,
                "company": company,
                "form": form,
                "accession_number": accession,
                "filing_date": filing_date,
                "report_date": report_date,
                "primary_document": primary_document,
                "primary_doc_description": description,
                "items": items,
                "size": filing.get("size"),
                "is_xbrl": filing.get("isXBRL"),
                "is_inline_xbrl": filing.get("isInlineXBRL"),
                "file_number": filing.get("fileNumber"),
                "act": filing.get("act"),
                "tickers": list(submissions.get("tickers") or []),
            },
        }

        provenance = Provenance(
            source_system=self.source,
            record_hash=hashlib.sha256(f"{cik10}:{accession}:{form}".encode()).hexdigest(),
            licence="US public domain (SEC EDGAR)",
            notes=url or SUBMISSIONS_URL.format(cik10=cik10),
        )
        kwargs: dict[str, Any] = {
            "kind": "filing.indexed",
            "source": self.source,
            "entities": entities,
            "payload": payload,
            "provenance": provenance,
        }
        if occurred_at is not None:
            kwargs["occurred_at"] = occurred_at
        return EventEnvelope(**kwargs)

    # -- streaming ------------------------------------------------------------------

    async def stream(self) -> AsyncIterator[EventEnvelope]:
        """Yield one ``filing.indexed`` envelope per filing for every configured CIK.

        Filings are emitted in the API's order (most recent first), filtered by
        ``forms`` and capped at ``limit_per_cik`` when configured.
        """
        for cik10 in self.ciks:
            submissions = await self.submissions(cik10)
            emitted = 0
            for filing in iter_recent_filings(submissions):
                form = str(filing.get("form") or "").strip().upper()
                if self.forms is not None and form not in self.forms:
                    continue
                if self.limit_per_cik is not None and emitted >= self.limit_per_cik:
                    break
                yield self._filing_envelope(cik10, submissions, filing)
                emitted += 1
