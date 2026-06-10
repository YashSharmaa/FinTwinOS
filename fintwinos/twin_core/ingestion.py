"""The twin's single ingestion path: EventEnvelope in, federated state out.

:class:`TwinIngestor` is the only production write path into the twin
(CONTRACTS.md rule 5). Every envelope is:

1. recorded into the replay engine (episodes are the source of truth — an
   empty twin replayed over an episode reconstructs identical state);
2. entity-resolved, so duplicate identifiers and name variants collapse onto
   canonical twin entities;
3. routed by the first segment of its ``kind`` into the graph, time-series
   and/or document stores;
4. appended to the tamper-evident audit trail.

Routing table (``kind`` prefix -> stores touched):

==============  ====================================================================
``customer.*``  graph (customer node; resolver name registration)
``account.*``   graph (account node + ``owns`` edge; ``account.transfer`` ->
                ``transfers_to`` edge with running totals) + timeseries (balances,
                transfer outflows)
``trade.*``     graph (trade node, ``placed``/``on`` edges) + timeseries (notional)
``position.*``  graph (``holds`` edge) + timeseries (market value)
``case.*``      graph (case node, ``involves`` edges) + documents (narrative)
``alert.*``     graph (alert node, ``concerns`` edges) + timeseries (scores)
``policy.*``    documents (full text) + graph (policy node)
``market.*``    timeseries (prices, cash ladders, any keyed series);
                ``market.instrument`` -> graph (instrument node)
``document.*``  documents + graph (document node, ``references`` edges)
==============  ====================================================================

Unknown prefixes are recorded and audited as ``twin.ingest.unrouted`` but do
not raise — forward compatibility with new event kinds must never crash the
ingestion path. Malformed envelopes of a *known* kind (missing required
fields) raise :class:`fintwinos.core.errors.IngestionError` after auditing the
rejection.

Implements the :class:`fintwinos.core.interfaces.EventIngestor` protocol.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fintwinos.core.audit import AuditTrail
from fintwinos.core.errors import IngestionError
from fintwinos.core.interfaces import (
    DocumentStore,
    GraphStore,
    ReplayEngine,
    TimeSeriesStore,
)
from fintwinos.core.types import EntityRef, EventEnvelope
from fintwinos.twin_core.resolution import EntityResolver

AUDIT_ACTOR = "twin_ingestor"

#: kind prefixes the ingestor knows how to route.
ROUTED_PREFIXES = (
    "customer",
    "account",
    "trade",
    "position",
    "case",
    "alert",
    "policy",
    "market",
    "document",
)


class TwinIngestor:
    """Routes :class:`EventEnvelope`s into the federated twin stores.

    Parameters
    ----------
    graph, timeseries, documents:
        The twin's stores (typically the twin_core implementations).
    replay:
        Episode recorder; every envelope is recorded *before* routing so the
        replay log captures even rejected envelopes.
    audit:
        The shared tamper-evident audit trail.
    resolver:
        Entity resolver applied to every envelope entity; a fresh default
        resolver is created when omitted.
    """

    def __init__(
        self,
        graph: GraphStore,
        timeseries: TimeSeriesStore,
        documents: DocumentStore,
        replay: ReplayEngine,
        audit: AuditTrail,
        resolver: EntityResolver | None = None,
    ) -> None:
        self.graph = graph
        self.timeseries = timeseries
        self.documents = documents
        self.replay = replay
        self.audit = audit
        self.resolver = resolver or EntityResolver()
        self.ingested_count = 0
        self._routes = {
            "customer": self._on_customer,
            "account": self._on_account,
            "trade": self._on_trade,
            "position": self._on_position,
            "case": self._on_case,
            "alert": self._on_alert,
            "policy": self._on_policy,
            "market": self._on_market,
            "document": self._on_document,
        }

    # -- public API ---------------------------------------------------------------------

    def ingest(self, envelope: EventEnvelope) -> None:
        """Record, resolve, route and audit one envelope.

        Raises :class:`IngestionError` for malformed envelopes of a known
        kind; unknown kind prefixes are audited and skipped.
        """
        self.replay.record(envelope)
        prefix = envelope.kind.split(".", 1)[0]
        handler = self._routes.get(prefix)
        if handler is None:
            self.audit.append(
                AUDIT_ACTOR,
                "twin.ingest.unrouted",
                {"event_id": envelope.event_id, "kind": envelope.kind, "source": envelope.source},
            )
            return
        resolved = self._resolve_entities(envelope, prefix)
        try:
            routes = handler(resolved)
        except IngestionError as exc:
            self.audit.append(
                AUDIT_ACTOR,
                "twin.ingest.rejected",
                {"event_id": envelope.event_id, "kind": envelope.kind, "error": str(exc)},
            )
            raise
        except KeyError as exc:
            message = (
                f"envelope {envelope.event_id} ({envelope.kind}) is missing "
                f"required payload field {exc}"
            )
            self.audit.append(
                AUDIT_ACTOR,
                "twin.ingest.rejected",
                {"event_id": envelope.event_id, "kind": envelope.kind, "error": message},
            )
            raise IngestionError(message) from exc
        self.ingested_count += 1
        self.audit.append(
            AUDIT_ACTOR,
            "twin.ingested",
            {
                "event_id": envelope.event_id,
                "kind": envelope.kind,
                "source": envelope.source,
                "routes": routes,
                "content_hash": envelope.content_hash(),
            },
        )

    def ingest_many(self, envelopes: Iterable[EventEnvelope]) -> int:
        """Ingest a batch of envelopes in order; returns how many were ingested."""
        count = 0
        for envelope in envelopes:
            self.ingest(envelope)
            count += 1
        return count

    # -- entity resolution ----------------------------------------------------------------

    def _resolve_entities(self, envelope: EventEnvelope, prefix: str) -> EventEnvelope:
        """Canonicalise every envelope entity, auditing any fuzzy/alias merges.

        The payload ``name`` is only offered to the resolver for entities
        whose type matches the event's kind prefix (a ``customer.*`` event
        names its customer, not its accounts).
        """
        if not envelope.entities:
            return envelope
        resolved: list[EntityRef] = []
        changed = False
        for ref in envelope.entities:
            name = envelope.payload.get("name") if ref.entity_type == prefix else None
            result = self.resolver.canonicalise(ref, name=name)
            canonical = result.ref or ref
            if canonical.key() != ref.key():
                changed = True
                self.audit.append(
                    AUDIT_ACTOR,
                    "twin.entity_resolved",
                    {
                        "event_id": envelope.event_id,
                        "original": ref.key(),
                        "canonical": canonical.key(),
                        "confidence": result.confidence,
                        "method": result.method,
                    },
                )
            resolved.append(canonical)
        if not changed:
            return envelope
        return envelope.model_copy(update={"entities": resolved})

    def _entity(
        self,
        envelope: EventEnvelope,
        entity_type: str,
        payload_field: str | None = None,
    ) -> EntityRef | None:
        """Find the (already canonical) entity of a type, or build one from the payload.

        Payload-sourced identifiers are passed through the resolver so stale
        duplicate ids inside payloads still land on canonical entities.
        """
        for ref in envelope.entities:
            if ref.entity_type == entity_type:
                return ref
        if payload_field and payload_field in envelope.payload:
            raw = str(envelope.payload[payload_field])
            result = self.resolver.resolve(entity_type, raw)
            if result.ref is not None:
                return result.ref
            return EntityRef(entity_type=entity_type, entity_id=raw)
        return None

    @staticmethod
    def _require(ref: EntityRef | None, envelope: EventEnvelope, what: str) -> EntityRef:
        if ref is None:
            raise IngestionError(
                f"envelope {envelope.event_id} ({envelope.kind}) carries no {what} "
                "entity and no payload identifier for it"
            )
        return ref

    # -- handlers --------------------------------------------------------------------------

    def _on_customer(self, envelope: EventEnvelope) -> list[str]:
        """``customer.*`` -> customer node upsert (resolver already saw the name)."""
        ref = self._require(
            self._entity(envelope, "customer", "customer_id"), envelope, "customer"
        )
        attributes = {k: v for k, v in envelope.payload.items() if k != "customer_id"}
        self.graph.upsert_entity(ref, attributes)
        return ["graph"]

    def _on_account(self, envelope: EventEnvelope) -> list[str]:
        """``account.*`` -> account node/ownership edge; transfers and balances."""
        payload = envelope.payload
        subkind = envelope.kind.split(".", 1)[1] if "." in envelope.kind else ""
        if subkind == "transfer":
            src = self._account_from_payload(payload, "from_account_id")
            dst = self._account_from_payload(payload, "to_account_id")
            amount = float(payload["amount"])
            existing = self._edge(src, dst, "transfers_to")
            self.graph.add_relationship(
                src,
                dst,
                "transfers_to",
                {
                    "transfer_count": int(existing.get("transfer_count", 0)) + 1,
                    "total_amount": round(float(existing.get("total_amount", 0.0)) + amount, 2),
                    "last_amount": amount,
                    "last_at": envelope.occurred_at.isoformat(),
                    "currency": payload.get("currency", existing.get("currency")),
                },
            )
            self.timeseries.append(
                f"transfers:{src.entity_id}",
                envelope.occurred_at,
                amount,
                tags={"to_account": dst.entity_id, "currency": payload.get("currency")},
            )
            return ["graph", "timeseries"]

        ref = self._require(self._entity(envelope, "account", "account_id"), envelope, "account")
        attributes = {k: v for k, v in payload.items() if k != "account_id"}
        self.graph.upsert_entity(ref, attributes)
        routes = ["graph"]
        owner = self._entity(envelope, "customer", "customer_id")
        if owner is not None:
            self.graph.add_relationship(owner, ref, "owns")
        if "balance" in payload:
            self.timeseries.append(
                f"balance:{ref.entity_id}",
                envelope.occurred_at,
                float(payload["balance"]),
                tags={"currency": payload.get("currency")},
            )
            routes.append("timeseries")
        return routes

    def _on_trade(self, envelope: EventEnvelope) -> list[str]:
        """``trade.*`` -> trade node, placed/on edges, notional time series."""
        payload = envelope.payload
        trade = self._require(self._entity(envelope, "trade", "trade_id"), envelope, "trade")
        account = self._require(
            self._entity(envelope, "account", "account_id"), envelope, "account"
        )
        instrument = self._require(
            self._entity(envelope, "instrument", "instrument_id"), envelope, "instrument"
        )
        attributes = {k: v for k, v in payload.items() if k != "trade_id"}
        attributes.setdefault("executed_at", envelope.occurred_at.isoformat())
        self.graph.upsert_entity(trade, attributes)
        self.graph.add_relationship(account, trade, "placed")
        self.graph.add_relationship(trade, instrument, "on")
        quantity = float(payload["quantity"])
        price = float(payload["price"])
        self.timeseries.append(
            f"trade_notional:{instrument.entity_id}",
            envelope.occurred_at,
            round(abs(quantity) * price, 2),
            tags={"side": payload.get("side"), "account_id": account.entity_id},
        )
        return ["graph", "timeseries"]

    def _on_position(self, envelope: EventEnvelope) -> list[str]:
        """``position.*`` -> holds edge (account -> instrument) + market-value series."""
        payload = envelope.payload
        account = self._require(
            self._entity(envelope, "account", "account_id"), envelope, "account"
        )
        instrument = self._require(
            self._entity(envelope, "instrument", "instrument_id"), envelope, "instrument"
        )
        quantity = float(payload["quantity"])
        market_value = float(payload["market_value"])
        self.graph.add_relationship(
            account,
            instrument,
            "holds",
            {
                "quantity": quantity,
                "market_value": market_value,
                "as_of": envelope.occurred_at.isoformat(),
            },
        )
        self.timeseries.append(
            f"position:{account.entity_id}:{instrument.entity_id}",
            envelope.occurred_at,
            market_value,
            tags={"quantity": quantity},
        )
        return ["graph", "timeseries"]

    def _on_case(self, envelope: EventEnvelope) -> list[str]:
        """``case.*`` -> case node, involves edges, narrative into the document store."""
        payload = envelope.payload
        case = self._require(self._entity(envelope, "case", "case_id"), envelope, "case")
        attributes = {k: v for k, v in payload.items() if k not in {"case_id", "narrative"}}
        self.graph.upsert_entity(case, attributes)
        for ref in envelope.entities:
            if ref.key() != case.key():
                self.graph.add_relationship(case, ref, "involves")
        routes = ["graph"]
        narrative = payload.get("narrative")
        if narrative:
            self.documents.add(
                f"case:{case.entity_id}",
                str(narrative),
                metadata={
                    "type": "case_narrative",
                    "case_id": case.entity_id,
                    "case_kind": payload.get("kind"),
                },
            )
            routes.append("documents")
        return routes

    def _on_alert(self, envelope: EventEnvelope) -> list[str]:
        """``alert.*`` -> alert node, concerns edges, score time series."""
        payload = envelope.payload
        alert = self._require(self._entity(envelope, "alert", "alert_id"), envelope, "alert")
        attributes = {k: v for k, v in payload.items() if k != "alert_id"}
        self.graph.upsert_entity(alert, attributes)
        for ref in envelope.entities:
            if ref.key() != alert.key():
                self.graph.add_relationship(alert, ref, "concerns")
        self.timeseries.append(
            f"alerts:{envelope.kind}",
            envelope.occurred_at,
            float(payload.get("score", 1.0)),
            tags={"alert_id": alert.entity_id, "severity": payload.get("severity")},
        )
        return ["graph", "timeseries"]

    def _on_policy(self, envelope: EventEnvelope) -> list[str]:
        """``policy.*`` -> full text into the document store + a policy node."""
        payload = envelope.payload
        policy = self._require(self._entity(envelope, "policy", "policy_id"), envelope, "policy")
        body = payload.get("body")
        if not body:
            raise IngestionError(
                f"envelope {envelope.event_id} ({envelope.kind}) has no policy body"
            )
        title = payload.get("title", policy.entity_id)
        text = f"{title}\n\n{body}" if title else str(body)
        self.documents.add(
            policy.entity_id,
            text,
            metadata={
                "type": "policy",
                "title": title,
                "tags": list(payload.get("tags", [])),
                "version": payload.get("version", "1.0"),
            },
        )
        self.graph.upsert_entity(
            policy,
            {
                "title": title,
                "tags": list(payload.get("tags", [])),
                "version": payload.get("version", "1.0"),
            },
        )
        return ["documents", "graph"]

    def _on_market(self, envelope: EventEnvelope) -> list[str]:
        """``market.*`` -> time series; ``market.instrument`` also upserts the node."""
        payload = envelope.payload
        subkind = envelope.kind.split(".", 1)[1] if "." in envelope.kind else ""
        if subkind == "instrument":
            instrument = self._require(
                self._entity(envelope, "instrument", "instrument_id"), envelope, "instrument"
            )
            attributes = {k: v for k, v in payload.items() if k != "instrument_id"}
            self.graph.upsert_entity(instrument, attributes)
            return ["graph"]
        series_key = payload.get("series_key")
        if not series_key:
            instrument_id = payload.get("instrument_id") or payload.get("symbol")
            if not instrument_id:
                raise IngestionError(
                    f"envelope {envelope.event_id} ({envelope.kind}) has neither a "
                    "series_key nor an instrument_id/symbol"
                )
            series_key = f"price:{instrument_id}"
        value = next(
            (payload[field] for field in ("value", "mid", "price") if payload.get(field) is not None),
            None,
        )
        if value is None:
            raise IngestionError(
                f"envelope {envelope.event_id} ({envelope.kind}) has no numeric "
                "value/mid/price field"
            )
        self.timeseries.append(
            str(series_key),
            envelope.occurred_at,
            float(value),
            tags=dict(payload.get("tags", {})),
        )
        return ["timeseries"]

    def _on_document(self, envelope: EventEnvelope) -> list[str]:
        """``document.*`` -> document store + document node with references edges."""
        payload = envelope.payload
        doc_ref = self._entity(envelope, "document", "doc_id") or self._entity(
            envelope, "document", "document_id"
        )
        doc_ref = self._require(doc_ref, envelope, "document")
        text = payload.get("text") or payload.get("body")
        if not text:
            raise IngestionError(
                f"envelope {envelope.event_id} ({envelope.kind}) has no text/body"
            )
        metadata = dict(payload.get("metadata", {}))
        if "title" in payload:
            metadata.setdefault("title", payload["title"])
        metadata.setdefault("type", "document")
        self.documents.add(doc_ref.entity_id, str(text), metadata=metadata)
        self.graph.upsert_entity(doc_ref, {"title": payload.get("title")})
        for ref in envelope.entities:
            if ref.key() != doc_ref.key():
                self.graph.add_relationship(doc_ref, ref, "references")
        return ["documents", "graph"]

    # -- helpers -------------------------------------------------------------------------

    def _account_from_payload(self, payload: dict[str, Any], field: str) -> EntityRef:
        raw = str(payload[field])
        result = self.resolver.resolve("account", raw)
        if result.ref is not None:
            return result.ref
        return EntityRef(entity_type="account", entity_id=raw)

    def _edge(self, src: EntityRef, dst: EntityRef, kind: str) -> dict[str, Any]:
        """Existing edge attributes via the store's optional get_relationship hook."""
        getter = getattr(self.graph, "get_relationship", None)
        if getter is None:
            return {}
        return getter(src, dst, kind) or {}
