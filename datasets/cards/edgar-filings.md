# Data card — edgar-filings

## Description

Two related sources behind one registry entry:

1. **Bundled sample corpus** — `fintwinos.datasets.edgar_loader.sample_filing_corpus()`
   returns seven MD&A / risk-factor style excerpts for *fictional* financial issuers
   (a regional bank, a payments processor, a custodian, a ship financier, an asset
   manager, an electronic trading venue and a specialty insurer). Written for
   FinTwinOS so document-store demos and retrieval evals run fully offline.
2. **Cached real filings** — `fintwinos.datasets.edgar_loader.load_cached_filings(data_dir)`
   reads the EDGAR connector's deployment-local JSON cache and normalises each filing
   into the same shape.

## Schema

Both loaders return `PolicyDoc`-like dictionaries:

| Key | Value |
|---|---|
| `policy_id` | `edgar_<normalised accession number>` |
| `title` | `"<company> <form_type> (<filed date>)"` or the cached title |
| `body` | Filing text, or sections joined under `## <section>` headers |
| `tags` | `["edgar", "filing", "<form type>"]` |
| `version` | `"1.0"` |
| `effective_date` | Filing date as an aware UTC datetime (or `null`) |
| `metadata` | `source`, `accession_no`, `cik`, `company`, `form_type`, `url` |

Cache format accepted by `load_cached_filings`: one JSON object per `*.json` file
(lists and an `index.json` with `{"filings": [...]}` also accepted) with tolerated
key aliases (`accession_no`/`accession_number`, `company`/`company_name`,
`form_type`/`form`, `filed_at`/`filing_date`, `text`/`body`/`content` or `sections`).

## Generation / provenance

The sample corpus is original text written for FinTwinOS; every issuer, CIK and
accession number is invented, and any resemblance to a real company is coincidental.
Real cached filings originate from the SEC EDGAR system via the connectors module;
each loaded document records its accession number and source URL in `metadata`.

## Licence and pass-through terms

- Sample corpus: MIT (part of FinTwinOS).
- Real EDGAR filings: US government public-domain material; no copyright restriction
  on the filings themselves, but observe the SEC's fair-access/rate-limit policies
  when fetching, identify your client per SEC guidance, and treat the cache as
  deployment-local working data — never commit it to the repository.

## Intended use

Document-store seeding, retrieval and summarisation demos, compliance-analyst agent
exercises, and offline evals over risk-factor language.

## Limitations

- The sample corpus is small (seven documents) and stylistically uniform; it is not
  a benchmark for retrieval quality at scale.
- Fictional excerpts compress real disclosure style; they are not legal or financial
  disclosure templates.
- `load_cached_filings` depends on the connector having populated the cache; it
  performs no fetching itself.
