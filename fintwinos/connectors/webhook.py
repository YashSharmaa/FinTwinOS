"""Push ingestion: a FastAPI webhook router feeding an asyncio queue, plus the
``QueueConnector`` that streams the queue back out as envelopes.

The two halves are deliberately decoupled through a plain ``asyncio.Queue``:

- :func:`build_webhook_router` returns an ``APIRouter`` exposing
  ``POST /ingest/{kind}``. Each validated request becomes an
  :class:`~fintwinos.core.types.EventEnvelope` pushed onto the queue (HTTP 202).
- :class:`QueueConnector` is a normal connector whose :meth:`~QueueConnector.stream`
  drains that queue, so the standard :func:`~fintwinos.connectors.base.pump` /
  ingestor machinery applies unchanged to push sources.

FastAPI itself is imported lazily inside :func:`build_webhook_router`, so the rest
of this module (and the package) works without the ``server`` extra installed.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from fintwinos.connectors.base import BaseConnector
from fintwinos.core.types import EntityRef, EventEnvelope, Provenance, utcnow

KIND_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.\-]{0,127}$")
"""Accepted event kinds: lowercase alphanumerics plus ``_ . -``, max 128 chars."""


class WebhookEventIn(BaseModel):
    """Validated body of ``POST /ingest/{kind}``.

    Attributes:
        payload: The event payload (required, must be a JSON object).
        entities: Optional typed entity references the event concerns.
        occurred_at: Optional event time; defaults to receipt time.
        source: Optional per-event source override for multiplexed endpoints.
    """

    payload: dict[str, Any]
    entities: list[EntityRef] = Field(default_factory=list)
    occurred_at: datetime | None = None
    source: str | None = None


def build_webhook_router(queue: asyncio.Queue, *, source: str = "webhook") -> Any:
    """Build a FastAPI ``APIRouter`` that pushes validated envelopes onto ``queue``.

    The route is ``POST /ingest/{kind}`` and returns HTTP 202 with the assigned
    ``event_id`` and current queue depth. Invalid kinds are rejected with 422 and
    a full queue with 503, so producers get immediate backpressure instead of
    silent loss.

    Args:
        queue: The ``asyncio.Queue`` shared with a :class:`QueueConnector`.
        source: Default ``EventEnvelope.source`` when the request body does not
            override it.

    Returns:
        A ``fastapi.APIRouter`` ready for ``app.include_router(...)``.

    Raises:
        ImportError: If FastAPI is not installed (install the ``server`` extra:
            ``pip install 'fintwinos[server]'``).
    """
    try:
        from fastapi import APIRouter, HTTPException
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "fastapi is required for the webhook connector; "
            "install the server extra: pip install 'fintwinos[server]'"
        ) from exc

    router = APIRouter(tags=["ingest"])

    @router.post("/ingest/{kind}", status_code=202)
    async def ingest_event(kind: str, event: WebhookEventIn) -> dict[str, Any]:
        """Validate one pushed event and enqueue it as an ``EventEnvelope``."""
        if not KIND_PATTERN.match(kind):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"invalid event kind {kind!r}: must match {KIND_PATTERN.pattern} "
                    "(lowercase alphanumerics plus '_', '.', '-')"
                ),
            )
        event_source = event.source or source
        envelope = EventEnvelope(
            kind=kind,
            source=event_source,
            occurred_at=event.occurred_at or utcnow(),
            entities=event.entities,
            payload=event.payload,
            provenance=Provenance(
                source_system=event_source,
                notes=f"webhook POST /ingest/{kind}",
            ),
        )
        if envelope.provenance is not None:
            envelope.provenance.record_hash = envelope.content_hash()
        try:
            queue.put_nowait(envelope)
        except asyncio.QueueFull as exc:
            raise HTTPException(
                status_code=503, detail="ingest queue is full; retry later"
            ) from exc
        return {
            "accepted": True,
            "event_id": envelope.event_id,
            "kind": kind,
            "queue_depth": queue.qsize(),
        }

    return router


class QueueConnector(BaseConnector):
    """Stream envelopes from an ``asyncio.Queue`` (the webhook router's sink).

    The stream ends when any of these happens first:

    - the :data:`QueueConnector.STOP` sentinel is received (see :meth:`close`);
    - ``idle_timeout`` seconds pass with the queue empty;
    - ``max_events`` envelopes have been yielded.

    Queue items may be ``EventEnvelope`` instances or plain dicts (validated into
    envelopes). Anything else is dropped and counted in :attr:`skipped`.

    Args:
        queue: The shared queue.
        name: Connector name (default ``"webhook_queue"``).
        source: Fallback source label (envelopes carry their own).
        idle_timeout: Seconds to wait on an empty queue before ending the stream;
            ``None`` waits forever (until :meth:`close`).
        max_events: Optional hard cap on yielded envelopes.
    """

    STOP: ClassVar[object] = object()
    """Sentinel; put it on the queue (or call :meth:`close`) to end the stream."""

    def __init__(
        self,
        queue: asyncio.Queue,
        *,
        name: str = "webhook_queue",
        source: str = "webhook",
        idle_timeout: float | None = None,
        max_events: int | None = None,
    ):
        self.queue = queue
        self.idle_timeout = idle_timeout
        self.max_events = max_events
        self.skipped = 0
        super().__init__(name=name, source=source)

    async def close(self) -> None:
        """Push the stop sentinel so a blocked :meth:`stream` ends cleanly."""
        await self.queue.put(self.STOP)

    async def stream(self) -> AsyncIterator[EventEnvelope]:
        """Yield envelopes from the queue until stopped, idle, or capped."""
        emitted = 0
        while self.max_events is None or emitted < self.max_events:
            try:
                if self.idle_timeout is None:
                    item = await self.queue.get()
                else:
                    item = await asyncio.wait_for(self.queue.get(), timeout=self.idle_timeout)
            except TimeoutError:
                break
            try:
                if item is self.STOP:
                    break
                if isinstance(item, EventEnvelope):
                    envelope = item
                elif isinstance(item, dict):
                    try:
                        envelope = EventEnvelope.model_validate(item)
                    except Exception:  # noqa: BLE001 - junk on the queue must not kill the stream
                        self.skipped += 1
                        continue
                else:
                    self.skipped += 1
                    continue
            finally:
                self.queue.task_done()
            emitted += 1
            yield envelope
