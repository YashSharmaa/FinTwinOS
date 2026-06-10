"""Shared fixtures for the connectors test suite.

Registers the ``network`` marker and skips network-marked tests unless
``FINTWIN_RUN_NETWORK_TESTS=1`` is set, so the suite is fully offline by default.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fintwinos.core.config import Settings
from fintwinos.core.types import EventEnvelope

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "network: requires real network access; skipped unless FINTWIN_RUN_NETWORK_TESTS=1",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("FINTWIN_RUN_NETWORK_TESTS") == "1":
        return
    skip_network = pytest.mark.skip(
        reason="network tests disabled by default; set FINTWIN_RUN_NETWORK_TESTS=1 to run"
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


class RecordingIngestor:
    """Fake ``EventIngestor`` that records envelopes and can fail on demand."""

    def __init__(self, fail_kinds: set[str] | None = None):
        self.envelopes: list[EventEnvelope] = []
        self.fail_kinds = fail_kinds or set()

    def ingest(self, envelope: EventEnvelope) -> None:
        if envelope.kind in self.fail_kinds:
            raise RuntimeError(f"simulated ingest failure for kind {envelope.kind!r}")
        self.envelopes.append(envelope)


class AsyncRecordingIngestor:
    """Fake ingestor with an async ``ingest`` to prove pump awaits awaitables."""

    def __init__(self):
        self.envelopes: list[EventEnvelope] = []

    async def ingest(self, envelope: EventEnvelope) -> None:
        self.envelopes.append(envelope)


@pytest.fixture
def recorder_factory():
    """Factory fixture: build :class:`RecordingIngestor` instances on demand."""

    def make(fail_kinds: set[str] | None = None) -> RecordingIngestor:
        return RecordingIngestor(fail_kinds=fail_kinds)

    return make


@pytest.fixture
def async_recorder() -> AsyncRecordingIngestor:
    return AsyncRecordingIngestor()


@pytest.fixture
def offline_settings(tmp_path: Path) -> Settings:
    """Settings pinned offline with an isolated, throwaway data directory."""
    return Settings(offline=True, openai_api_key=None, data_dir=tmp_path / "data")


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
