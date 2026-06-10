"""FinTwinOS connectors: adapters from real and synthetic sources to envelope streams.

Every connector here satisfies the :class:`fintwinos.core.interfaces.Connector`
Protocol — a ``name`` plus an async ``stream()`` yielding
:class:`~fintwinos.core.types.EventEnvelope` — and is drained into the twin with
:func:`~fintwinos.connectors.base.pump`. Connectors only *emit*; ingestion,
recording and auditing happen downstream through ``runtime.ingestor``.

Available connectors:

- :class:`~fintwinos.connectors.csv_batch.CsvBatchConnector` — declarative CSV
  row-to-envelope mapping via :class:`~fintwinos.connectors.csv_batch.ColumnMapping`.
- :class:`~fintwinos.connectors.jsonl_cdc.JsonlCdcConnector` — JSONL change-data-
  capture tailing with resumable offsets.
- :func:`~fintwinos.connectors.webhook.build_webhook_router` and
  :class:`~fintwinos.connectors.webhook.QueueConnector` — push ingestion over HTTP.
- :class:`~fintwinos.connectors.edgar.EdgarConnector` — SEC EDGAR filings and XBRL
  company facts, offline-first via a local JSON cache.
- :class:`~fintwinos.connectors.market_synth.SyntheticMarketConnector` — seeded
  GBM+jump market ticks for demos, simulators and tests.
"""

from fintwinos.connectors.base import BaseConnector, ConnectorError, pump
from fintwinos.connectors.csv_batch import (
    ColumnMapping,
    CsvBatchConnector,
    coerce_value,
    row_to_envelope,
)
from fintwinos.connectors.edgar import (
    EdgarConnector,
    EdgarOfflineError,
    fact_series,
    iter_recent_filings,
    normalise_cik,
)
from fintwinos.connectors.jsonl_cdc import JsonlCdcConnector
from fintwinos.connectors.market_synth import SyntheticMarketConnector
from fintwinos.connectors.webhook import QueueConnector, WebhookEventIn, build_webhook_router

__all__ = [
    "BaseConnector",
    "ColumnMapping",
    "ConnectorError",
    "CsvBatchConnector",
    "EdgarConnector",
    "EdgarOfflineError",
    "JsonlCdcConnector",
    "QueueConnector",
    "SyntheticMarketConnector",
    "WebhookEventIn",
    "build_webhook_router",
    "coerce_value",
    "fact_series",
    "iter_recent_filings",
    "normalise_cik",
    "pump",
    "row_to_envelope",
]
