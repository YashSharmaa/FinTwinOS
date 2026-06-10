"""Tests for BaseConnector and the pump() drain helper."""

from __future__ import annotations

import pytest

from fintwinos.connectors.base import BaseConnector, ConnectorError, pump
from fintwinos.core.interfaces import Connector
from fintwinos.core.types import EventEnvelope


class ListConnector(BaseConnector):
    """Yields a fixed list of envelopes; records whether the stream was closed."""

    def __init__(self, envelopes: list[EventEnvelope], name: str = "list_connector"):
        self.envelopes = envelopes
        self.closed_cleanly = False
        super().__init__(name=name, source="test")

    async def stream(self):
        try:
            for envelope in self.envelopes:
                yield envelope
        finally:
            self.closed_cleanly = True


class ExplodingStreamConnector(BaseConnector):
    """Yields two envelopes then raises mid-stream."""

    def __init__(self):
        super().__init__(name="exploding", source="test")

    async def stream(self):
        yield EventEnvelope(kind="ok.one", source="test")
        yield EventEnvelope(kind="ok.two", source="test")
        raise ConnectorError("source went away mid-stream")


def make_envelopes(n: int, kind: str = "unit.test") -> list[EventEnvelope]:
    return [EventEnvelope(kind=kind, source="test", payload={"i": i}) for i in range(n)]


async def test_pump_ingests_everything(recorder_factory):
    connector = ListConnector(make_envelopes(5))
    recorder = recorder_factory()
    summary = await pump(connector, recorder)

    assert summary["connector"] == "list_connector"
    assert summary["emitted"] == 5
    assert summary["ingested"] == 5
    assert summary["failed"] == 0
    assert summary["errors"] == []
    assert summary["stream_error"] is None
    assert summary["limit_reached"] is False
    assert [e.payload["i"] for e in recorder.envelopes] == [0, 1, 2, 3, 4]
    assert summary["elapsed_ms"] >= 0
    assert connector.closed_cleanly


async def test_pump_isolates_per_envelope_failures(recorder_factory):
    good = make_envelopes(3, kind="good.event")
    bad = make_envelopes(2, kind="bad.event")
    connector = ListConnector([good[0], bad[0], good[1], bad[1], good[2]])
    recorder = recorder_factory(fail_kinds={"bad.event"})

    summary = await pump(connector, recorder)

    assert summary["emitted"] == 5
    assert summary["ingested"] == 3
    assert summary["failed"] == 2
    assert len(summary["errors"]) == 2
    for error in summary["errors"]:
        assert error["kind"] == "bad.event"
        assert "RuntimeError" in error["error"]
        assert error["event_id"].startswith("evt_")
    # the good envelopes all landed despite the poison records
    assert [e.kind for e in recorder.envelopes] == ["good.event"] * 3


async def test_pump_respects_limit_and_closes_stream(recorder_factory):
    connector = ListConnector(make_envelopes(10))
    recorder = recorder_factory()

    summary = await pump(connector, recorder, limit=3)

    assert summary["emitted"] == 3
    assert summary["ingested"] == 3
    assert summary["limit_reached"] is True
    assert len(recorder.envelopes) == 3
    assert connector.closed_cleanly  # generator was aclosed, not abandoned


async def test_pump_zero_limit_emits_nothing(recorder_factory):
    connector = ListConnector(make_envelopes(4))
    summary = await pump(connector, recorder_factory(), limit=0)
    assert summary["emitted"] == 0
    assert summary["ingested"] == 0


async def test_pump_reports_stream_failure_without_raising(recorder_factory):
    connector = ExplodingStreamConnector()
    recorder = recorder_factory()

    summary = await pump(connector, recorder)

    assert summary["ingested"] == 2
    assert summary["stream_error"] is not None
    assert "ConnectorError" in summary["stream_error"]
    assert "source went away" in summary["stream_error"]


async def test_pump_awaits_async_ingestors(async_recorder):
    connector = ListConnector(make_envelopes(4))
    summary = await pump(connector, async_recorder)
    assert summary["ingested"] == 4
    assert len(async_recorder.envelopes) == 4


async def test_pump_caps_recorded_errors(recorder_factory):
    connector = ListConnector(make_envelopes(8, kind="bad.event"))
    recorder = recorder_factory(fail_kinds={"bad.event"})
    summary = await pump(connector, recorder, max_recorded_errors=3)
    assert summary["failed"] == 8  # exact count is always kept
    assert len(summary["errors"]) == 3  # details are capped


def test_base_connector_satisfies_protocol():
    connector = ListConnector([])
    assert isinstance(connector, Connector)
    assert connector.name == "list_connector"
    assert connector.source == "test"


def test_base_connector_is_abstract():
    with pytest.raises(TypeError):
        BaseConnector()  # type: ignore[abstract]
