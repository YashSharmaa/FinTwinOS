"""Filings & policy document tool pack.

Tools registered here:

- ``observe_filing_search`` — keyword search over the twin's document store with
  optional document-type filtering (10-K, Pillar 3, prospectus, ...).
- ``observe_policy_lookup`` — internal policy documents located by tag and/or query.

Both tools are read-only and reach documents exclusively through
``runtime.documents`` (the ``DocumentStore`` Protocol).
"""

from __future__ import annotations

from typing import Any

from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import RiskTier, SideEffectClass, ToolBand
from fintwinos.tools.domains.common import json_safe, object_schema, provenance_entry
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec

OWNER = "filings"

_SNIPPET_CHARS = 300


def _document_row(runtime: TwinRuntime, doc_id: str, score: float) -> dict[str, Any] | None:
    """Build a JSON-safe search-result row for one document, or None if missing."""
    doc = runtime.documents.get(doc_id)
    if doc is None:
        return None
    text = str(doc.get("text") or doc.get("body") or "")
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {
            key: value
            for key, value in doc.items()
            if key not in {"text", "body", "doc_id"} and not isinstance(value, dict | list)
        }
    snippet = " ".join(text.split())[:_SNIPPET_CHARS]
    return {
        "doc_id": doc_id,
        "score": round(float(score), 4),
        "snippet": snippet,
        "metadata": json_safe(metadata),
    }


def register(registry: ToolRegistry, runtime: TwinRuntime) -> None:
    """Register the filings tool pack on the given registry against the runtime."""

    # -- observe_filing_search -------------------------------------------------------

    def observe_filing_search(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        query = arguments["query"]
        k = int(arguments.get("k", 5))
        doc_type = arguments.get("doc_type")
        # Over-fetch when filtering so the type filter does not starve results.
        fetch_k = k if doc_type is None else max(k * 4, 20)
        rows: list[dict[str, Any]] = []
        for doc_id, score in runtime.documents.search(query, k=fetch_k):
            row = _document_row(runtime, doc_id, score)
            if row is None:
                continue
            if doc_type is not None and row["metadata"].get("doc_type") != doc_type:
                continue
            rows.append(row)
        return {
            "query": query,
            "doc_type": doc_type,
            "count": len(rows[:k]),
            "results": rows[:k],
            "provenance": [provenance_entry("twin_core.documents", f"search:{query}")],
        }

    registry.register(
        ToolSpec(
            name="observe_filing_search",
            description=(
                "Search filings and other documents held in the twin's document store by "
                "keyword query, optionally restricted to one document type (e.g. '10-K'). "
                "Returns ranked snippets with metadata."
            ),
            input_schema=object_schema(
                {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Keyword query, e.g. 'liquidity coverage ratio'.",
                    },
                    "k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "description": "Maximum results to return (default 5).",
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Only documents whose metadata doc_type matches.",
                    },
                },
                ["query"],
            ),
            band=ToolBand.observe,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.read,
            owner=OWNER,
        ),
        observe_filing_search,
    )

    # -- observe_policy_lookup -------------------------------------------------------

    def observe_policy_lookup(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        query = arguments.get("query")
        tag = arguments.get("tag")
        k = int(arguments.get("k", 5))
        search_query = query or tag or ""
        warnings: list[str] = []

        rows: list[dict[str, Any]] = []
        for doc_id, score in runtime.documents.search(search_query, k=max(k * 4, 20)):
            row = _document_row(runtime, doc_id, score)
            if row is not None:
                rows.append(row)

        if tag is not None:
            tagged = [
                row
                for row in rows
                if tag in (row["metadata"].get("tags") or [])
            ]
            if tagged:
                rows = tagged
            else:
                warnings.append(
                    f"no documents carry tag '{tag}'; returning untagged search results"
                )
        policy_rows = [row for row in rows if row["metadata"].get("kind") == "policy"]
        if policy_rows:
            rows = policy_rows
        elif rows:
            warnings.append(
                "no documents are marked kind='policy'; returning unfiltered search results"
            )
        return {
            "query": query,
            "tag": tag,
            "count": len(rows[:k]),
            "results": rows[:k],
            "warnings": warnings,
            "provenance": [
                provenance_entry("twin_core.documents", f"policy_lookup:{search_query}")
            ],
        }

    registry.register(
        ToolSpec(
            name="observe_policy_lookup",
            description=(
                "Locate internal policy documents by tag and/or keyword query. At least one "
                "of 'query' or 'tag' must be supplied; results prefer documents marked "
                "kind='policy' and carrying the requested tag."
            ),
            input_schema=object_schema(
                {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Keyword query over policy text.",
                    },
                    "tag": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Policy tag, e.g. 'aml' or 'liquidity'.",
                    },
                    "k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "description": "Maximum results to return (default 5).",
                    },
                },
                extra={"anyOf": [{"required": ["query"]}, {"required": ["tag"]}]},
            ),
            band=ToolBand.observe,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.read,
            owner=OWNER,
        ),
        observe_policy_lookup,
    )
