"""Tests for the webhook router and the QueueConnector that drains its queue."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fintwinos.connectors.base import pump
from fintwinos.connectors.webhook import QueueConnector, build_webhook_router
from fintwinos.core.types import EventEnvelope


def make_app(queue: asyncio.Queue, **router_kwargs) -> FastAPI:
    app = FastAPI()
    app.include_router(build_webhook_router(queue, **router_kwargs))
    return app


def test_post_ingest_pushes_envelope_to_queue():
    queue: asyncio.Queue = asyncio.Queue()
    client = TestClient(make_app(queue))

    response = client.post(
        "/ingest/payment.received",
        json={
            "payload": {"amount": 250.0, "currency": "USD"},
            "entities": [{"entity_type": "account", "entity_id": "acc_001"}],
            "occurred_at": "2026-01-05T12:00:00+00:00",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] is True
    assert body["kind"] == "payment.received"
    assert body["event_id"].startswith("evt_")
    assert body["queue_depth"] == 1

    envelope = queue.get_nowait()
    assert isinstance(envelope, EventEnvelope)
    assert envelope.event_id == body["event_id"]
    assert envelope.kind == "payment.received"
    assert envelope.source == "webhook"
    assert envelope.payload == {"amount": 250.0, "currency": "USD"}
    assert [ref.key() for ref in envelope.entities] == ["account:acc_001"]
    assert envelope.occurred_at.isoformat() == "2026-01-05T12:00:00+00:00"
    assert envelope.provenance is not None
    assert envelope.provenance.record_hash == envelope.content_hash()


def test_source_override_in_body():
    queue: asyncio.Queue = asyncio.Queue()
    client = TestClient(make_app(queue, source="default_hook"))

    client.post("/ingest/alert.raised", json={"payload": {}, "source": "siem"})
    assert queue.get_nowait().source == "siem"

    client.post("/ingest/alert.raised", json={"payload": {}})
    assert queue.get_nowait().source == "default_hook"


def test_invalid_kind_is_rejected():
    queue: asyncio.Queue = asyncio.Queue()
    client = TestClient(make_app(queue))
    response = client.post("/ingest/Payment Received!", json={"payload": {}})
    assert response.status_code == 422
    assert "invalid event kind" in response.json()["detail"]
    assert queue.qsize() == 0


def test_missing_payload_is_rejected_by_validation():
    queue: asyncio.Queue = asyncio.Queue()
    client = TestClient(make_app(queue))
    response = client.post("/ingest/payment.received", json={"entities": []})
    assert response.status_code == 422
    assert queue.qsize() == 0


def test_non_object_payload_is_rejected():
    queue: asyncio.Queue = asyncio.Queue()
    client = TestClient(make_app(queue))
    response = client.post("/ingest/payment.received", json={"payload": "not-a-dict"})
    assert response.status_code == 422
    assert queue.qsize() == 0


def test_full_queue_returns_503_backpressure():
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    client = TestClient(make_app(queue))
    assert client.post("/ingest/a.b", json={"payload": {}}).status_code == 202
    response = client.post("/ingest/a.b", json={"payload": {}})
    assert response.status_code == 503
    assert queue.qsize() == 1


async def test_queue_connector_streams_until_sentinel():
    queue: asyncio.Queue = asyncio.Queue()
    connector = QueueConnector(queue)
    await queue.put(EventEnvelope(kind="k.one", source="webhook"))
    await queue.put(EventEnvelope(kind="k.two", source="webhook"))
    await connector.close()

    envelopes = [e async for e in connector.stream()]
    assert [e.kind for e in envelopes] == ["k.one", "k.two"]


async def test_queue_connector_idle_timeout_ends_stream():
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(EventEnvelope(kind="k.one", source="webhook"))
    connector = QueueConnector(queue, idle_timeout=0.05)
    envelopes = [e async for e in connector.stream()]
    assert len(envelopes) == 1


async def test_queue_connector_max_events_cap():
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(5):
        await queue.put(EventEnvelope(kind=f"k.{i}", source="webhook"))
    connector = QueueConnector(queue, max_events=3, idle_timeout=0.05)
    envelopes = [e async for e in connector.stream()]
    assert len(envelopes) == 3
    assert queue.qsize() == 2  # the rest stays queued for the next stream


async def test_queue_connector_validates_dicts_and_skips_junk():
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put({"kind": "from.dict", "source": "webhook", "payload": {"x": 1}})
    await queue.put("garbage")
    await queue.put(12345)
    connector = QueueConnector(queue, idle_timeout=0.05)
    envelopes = [e async for e in connector.stream()]
    assert [e.kind for e in envelopes] == ["from.dict"]
    assert connector.skipped == 2


async def test_pump_drains_webhook_queue(recorder_factory):
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(3):
        await queue.put(EventEnvelope(kind="hook.event", source="webhook", payload={"i": i}))
    connector = QueueConnector(queue, idle_timeout=0.05)
    recorder = recorder_factory()
    summary = await pump(connector, recorder)
    assert summary["ingested"] == 3
    assert [e.payload["i"] for e in recorder.envelopes] == [0, 1, 2]


def test_end_to_end_webhook_to_connector():
    """HTTP POST -> queue -> QueueConnector stream, all in one flow."""
    queue: asyncio.Queue = asyncio.Queue()
    client = TestClient(make_app(queue))
    for i in range(2):
        response = client.post("/ingest/case.opened", json={"payload": {"i": i}})
        assert response.status_code == 202

    async def drain() -> list[EventEnvelope]:
        connector = QueueConnector(queue, idle_timeout=0.05)
        return [e async for e in connector.stream()]

    envelopes = asyncio.run(drain())
    assert [e.payload["i"] for e in envelopes] == [0, 1]
    assert all(e.kind == "case.opened" for e in envelopes)
