"""TwinIngestor: routing for every kind prefix, resolution, audit, error paths."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fintwinos.core.errors import IngestionError
from fintwinos.core.types import EntityRef, EventEnvelope
from fintwinos.twin_core.ingestion import ROUTED_PREFIXES

T0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)


def _env(kind: str, payload: dict, entities: list[EntityRef] | None = None) -> EventEnvelope:
    return EventEnvelope(
        kind=kind,
        occurred_at=T0,
        source="test",
        entities=entities or [],
        payload=payload,
        episode_id="ep_test",
    )


def _ref(entity_type: str, entity_id: str) -> EntityRef:
    return EntityRef(entity_type=entity_type, entity_id=entity_id)


def test_routed_prefixes_cover_the_contract():
    assert set(ROUTED_PREFIXES) == {
        "customer", "account", "trade", "position", "case",
        "alert", "policy", "market", "document",
    }


def test_customer_routes_to_graph(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "customer.created",
            {"customer_id": "c1", "name": "Acme Corp Ltd", "segment": "sme"},
            [_ref("customer", "c1")],
        )
    )
    entity = rt.graph.get_entity(_ref("customer", "c1"))
    assert entity["name"] == "Acme Corp Ltd"
    assert rt.ingestor.resolver.known(_ref("customer", "c1"))


def test_account_routes_to_graph_and_timeseries(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "account.created",
            {"account_id": "a1", "customer_id": "c1", "currency": "USD", "balance": 1200.5},
            [_ref("account", "a1"), _ref("customer", "c1")],
        )
    )
    assert rt.graph.get_entity(_ref("account", "a1"))["currency"] == "USD"
    assert rt.graph.get_relationship(_ref("customer", "c1"), _ref("account", "a1"), "owns") == {}
    assert rt.timeseries.latest("balance:a1")[1] == 1200.5


def test_account_transfer_accumulates_edge_totals(empty_runtime):
    rt = empty_runtime
    for amount in (9100.0, 9400.0):
        rt.ingestor.ingest(
            _env(
                "account.transfer",
                {"from_account_id": "a1", "to_account_id": "a2", "amount": amount,
                 "currency": "USD"},
                [_ref("account", "a1"), _ref("account", "a2")],
            )
        )
    edge = rt.graph.get_relationship(_ref("account", "a1"), _ref("account", "a2"), "transfers_to")
    assert edge["transfer_count"] == 2
    assert edge["total_amount"] == pytest.approx(18500.0)
    assert edge["last_amount"] == 9400.0
    outflows = rt.timeseries.window("transfers:a1")
    assert [v for _, v in outflows] == [9100.0, 9400.0]


def test_trade_routes_to_graph_and_timeseries(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "trade.executed",
            {"trade_id": "t1", "account_id": "a1", "instrument_id": "i1",
             "side": "buy", "quantity": 100.0, "price": 25.5},
            [_ref("trade", "t1"), _ref("account", "a1"), _ref("instrument", "i1")],
        )
    )
    assert rt.graph.get_entity(_ref("trade", "t1"))["side"] == "buy"
    assert rt.graph.get_relationship(_ref("account", "a1"), _ref("trade", "t1"), "placed") == {}
    assert rt.graph.get_relationship(_ref("trade", "t1"), _ref("instrument", "i1"), "on") == {}
    assert rt.timeseries.latest("trade_notional:i1")[1] == 2550.0


def test_position_routes_to_holds_edge_and_series(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "position.snapshot",
            {"account_id": "a1", "instrument_id": "i1", "quantity": 50.0,
             "market_value": 1275.0},
            [_ref("account", "a1"), _ref("instrument", "i1")],
        )
    )
    edge = rt.graph.get_relationship(_ref("account", "a1"), _ref("instrument", "i1"), "holds")
    assert edge["quantity"] == 50.0
    assert edge["market_value"] == 1275.0
    assert rt.timeseries.latest("position:a1:i1")[1] == 1275.0


def test_case_routes_to_graph_and_documents(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "case.opened",
            {"case_id": "k1", "kind": "complaint", "priority": "medium",
             "narrative": "Customer complaint about a delayed payment."},
            [_ref("case", "k1"), _ref("customer", "c1")],
        )
    )
    assert rt.graph.get_entity(_ref("case", "k1"))["kind"] == "complaint"
    assert rt.graph.get_relationship(_ref("case", "k1"), _ref("customer", "c1"), "involves") == {}
    doc = rt.documents.get("case:k1")
    assert doc is not None
    assert "delayed payment" in doc["text"]
    # narrative stays out of node attributes — documents own the text
    assert "narrative" not in rt.graph.get_entity(_ref("case", "k1"))


def test_alert_routes_to_graph_and_timeseries(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "alert.structuring",
            {"alert_id": "al1", "severity": "high", "score": 0.91},
            [_ref("alert", "al1"), _ref("account", "a1")],
        )
    )
    assert rt.graph.get_entity(_ref("alert", "al1"))["severity"] == "high"
    assert rt.graph.get_relationship(_ref("alert", "al1"), _ref("account", "a1"), "concerns") == {}
    assert rt.timeseries.latest("alerts:alert.structuring")[1] == 0.91


def test_policy_routes_to_documents_and_graph(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "policy.published",
            {"policy_id": "pol_x", "title": "Test Policy", "body": "Always verify approvals.",
             "tags": ["governance"], "version": "2.0"},
            [_ref("policy", "pol_x")],
        )
    )
    doc = rt.documents.get("pol_x")
    assert "Always verify approvals" in doc["text"]
    assert doc["metadata"]["version"] == "2.0"
    assert rt.graph.get_entity(_ref("policy", "pol_x"))["title"] == "Test Policy"


def test_market_price_routes_to_timeseries(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "market.price",
            {"instrument_id": "i1", "mid": 101.25, "tags": {"symbol": "ALPH"}},
            [_ref("instrument", "i1")],
        )
    )
    assert rt.timeseries.latest("price:i1")[1] == 101.25
    assert rt.timeseries.points("price:i1")[0][2] == {"symbol": "ALPH"}


def test_market_series_key_override(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env("market.cash_ladder", {"series_key": "cash_ladder:USD", "value": 1.4e8}, [])
    )
    assert rt.timeseries.latest("cash_ladder:USD")[1] == 1.4e8


def test_market_instrument_listing_routes_to_graph(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "market.instrument",
            {"instrument_id": "i1", "symbol": "ALPH", "asset_class": "equity"},
            [_ref("instrument", "i1")],
        )
    )
    assert rt.graph.get_entity(_ref("instrument", "i1"))["symbol"] == "ALPH"


def test_document_routes_to_documents_and_graph(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "document.added",
            {"doc_id": "d1", "title": "Annual Filing", "text": "Annual report filing text."},
            [_ref("document", "d1"), _ref("customer", "c1")],
        )
    )
    assert rt.documents.get("d1")["metadata"]["title"] == "Annual Filing"
    assert (
        rt.graph.get_relationship(_ref("document", "d1"), _ref("customer", "c1"), "references")
        == {}
    )


def test_unknown_kind_is_recorded_not_raised(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(_env("telemetry.heartbeat", {"value": 1}, []))
    unrouted = rt.audit.records(action="twin.ingest.unrouted")
    assert len(unrouted) == 1
    assert unrouted[0].payload["kind"] == "telemetry.heartbeat"
    # still captured by the replay engine for forensic completeness
    assert any(e.kind == "telemetry.heartbeat" for e in rt.replay.episode("ep_test"))


def test_malformed_known_kind_raises_ingestion_error(empty_runtime):
    rt = empty_runtime
    with pytest.raises(IngestionError):
        rt.ingestor.ingest(_env("trade.executed", {"trade_id": "t1"}, [_ref("trade", "t1")]))
    with pytest.raises(IngestionError):
        rt.ingestor.ingest(_env("market.price", {}, []))
    with pytest.raises(IngestionError):
        rt.ingestor.ingest(_env("policy.published", {"policy_id": "p"}, [_ref("policy", "p")]))
    assert len(rt.audit.records(action="twin.ingest.rejected")) == 3


def test_entity_resolution_merges_duplicate_customers(empty_runtime):
    rt = empty_runtime
    rt.ingestor.ingest(
        _env(
            "customer.created",
            {"customer_id": "c1", "name": "Acme Corporation Limited"},
            [_ref("customer", "c1")],
        )
    )
    # same counterparty arrives from another feed with a different id and spelling
    rt.ingestor.ingest(
        _env(
            "customer.created",
            {"customer_id": "c_dup", "name": "ACME Corp Ltd"},
            [_ref("customer", "c_dup")],
        )
    )
    assert rt.graph.get_entity(_ref("customer", "c_dup")) is None
    assert rt.graph.get_entity(_ref("customer", "c1")) is not None
    merges = rt.audit.records(action="twin.entity_resolved")
    assert len(merges) == 1
    assert merges[0].payload["canonical"] == "customer:c1"


def test_every_ingest_is_audited_and_replayed(empty_runtime):
    rt = empty_runtime
    envelopes = [
        _env("customer.created", {"customer_id": "c1", "name": "Acme Corp Ltd"},
             [_ref("customer", "c1")]),
        _env("market.price", {"instrument_id": "i1", "mid": 10.0}, []),
    ]
    count = rt.ingestor.ingest_many(envelopes)
    assert count == 2
    assert rt.ingestor.ingested_count == 2
    assert len(rt.audit.records(action="twin.ingested")) == 2
    assert len(rt.replay.episode("ep_test")) == 2
    assert rt.audit.verify()
