"""Connector foundations: the :class:`BaseConnector` ABC and the :func:`pump` helper.

A connector adapts one source — a CSV export, a CDC log, a webhook queue, a public
API, a synthetic generator — into an async stream of :class:`~fintwinos.core.types.EventEnvelope`.
Connectors never write to the twin themselves; they only *emit*. The :func:`pump`
helper drains a connector into an :class:`~fintwinos.core.interfaces.EventIngestor`
(``runtime.ingestor`` in a full deployment) with per-envelope error isolation, so a
single poison record can never abort a batch load.

Every connector in this package satisfies the structural
:class:`~fintwinos.core.interfaces.Connector` Protocol: a ``name`` attribute and an
async ``stream()`` yielding envelopes.
"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from fintwinos.core.errors import FinTwinError
from fintwinos.core.interfaces import EventIngestor
from fintwinos.core.types import EventEnvelope, utcnow


class ConnectorError(FinTwinError):
    """Raised when a connector cannot read, map or stream records from its source."""


class BaseConnector(ABC):
    """Abstract base for all FinTwinOS connectors.

    Subclasses implement :meth:`stream` as an async generator yielding
    :class:`~fintwinos.core.types.EventEnvelope` instances. The base class only
    carries the two pieces of identity every envelope needs:

    - ``name``: the connector instance's name, used in audit records, pump
      summaries and offset files.
    - ``source``: the value stamped into ``EventEnvelope.source`` and
      ``Provenance.source_system`` for every emitted envelope.
    """

    name: str = "connector"
    source: str = "unknown"

    def __init__(self, *, name: str | None = None, source: str | None = None):
        if name is not None:
            self.name = name
        if source is not None:
            self.source = source

    @abstractmethod
    def stream(self) -> AsyncIterator[EventEnvelope]:
        """Yield envelopes from the underlying source.

        Implementations must be async generators (``async def stream(self)`` with
        ``yield``). They should cooperate with the event loop (``await
        asyncio.sleep(0)`` between records for CPU-bound sources) and must finish
        cleanly when the source is exhausted.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, source={self.source!r})"


async def pump(
    connector: Any,
    ingestor: EventIngestor,
    limit: int | None = None,
    *,
    max_recorded_errors: int = 25,
) -> dict[str, Any]:
    """Drain ``connector.stream()`` into ``ingestor`` with per-envelope error isolation.

    Each envelope is passed to ``ingestor.ingest(envelope)`` individually; an
    exception raised by the ingestor is caught, counted and recorded, and pumping
    continues with the next envelope. A failure of the *stream itself* (the source
    is unreadable mid-iteration, for example) stops the pump and is reported in the
    summary instead of propagating, so callers always receive an accounting of what
    happened.

    Args:
        connector: Anything satisfying the ``Connector`` Protocol (``name`` plus an
            async ``stream()``). All classes in :mod:`fintwinos.connectors` qualify.
        ingestor: The sink, e.g. ``runtime.ingestor``. ``ingest`` may be sync or
            async; awaitables are awaited.
        limit: Stop after this many envelopes have been *emitted* (ingested or
            failed). ``None`` drains the stream to exhaustion.
        max_recorded_errors: Cap on the number of per-envelope error details kept
            in the summary (the ``failed`` counter is always exact).

    Returns:
        A summary dict with keys ``connector``, ``emitted``, ``ingested``,
        ``failed``, ``errors`` (list of ``{event_id, kind, error}``),
        ``stream_error`` (``None`` or a string), ``limit_reached``, ``started_at``,
        ``finished_at`` and ``elapsed_ms``.
    """
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "connector": getattr(connector, "name", type(connector).__name__),
        "emitted": 0,
        "ingested": 0,
        "failed": 0,
        "errors": [],
        "stream_error": None,
        "limit_reached": False,
        "started_at": utcnow().isoformat(),
    }

    stream = connector.stream()
    try:
        if limit is None or limit > 0:
            async for envelope in stream:
                summary["emitted"] += 1
                try:
                    outcome = ingestor.ingest(envelope)
                    if inspect.isawaitable(outcome):
                        await outcome
                    summary["ingested"] += 1
                except Exception as exc:  # noqa: BLE001 - poison records must not stop the batch
                    summary["failed"] += 1
                    if len(summary["errors"]) < max_recorded_errors:
                        summary["errors"].append(
                            {
                                "event_id": envelope.event_id,
                                "kind": envelope.kind,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                if limit is not None and summary["emitted"] >= limit:
                    summary["limit_reached"] = True
                    break
    except Exception as exc:  # noqa: BLE001 - report stream failures, never crash the caller
        summary["stream_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()

    summary["finished_at"] = utcnow().isoformat()
    summary["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return summary
