"""Content-addressed on-disk cache for LLM responses.

Identical prompts cost identical money, so the cache keys an ``LLMResponse`` by
``sha256(model + messages + json_schema)`` over a canonical JSON encoding. Entries
are plain, human-inspectable JSON files in a single directory — consistent with the
FinTwinOS auditability stance — plus a small ``index.json`` recording last-access
order for LRU eviction bounded by ``max_entries``.

The cache is purely local and works fully offline; it never talks to any provider.
Typical wiring::

    cache = ResponseCache(settings.data_dir / "llm_cache", max_entries=512)
    cached = cache.get(model, messages, json_schema)
    if cached is None:
        response = await client.complete(messages, model=model, json_schema=json_schema)
        cache.put(model, messages, response, json_schema)
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from fintwinos.core.config import Settings, get_settings
from fintwinos.core.types import utcnow
from fintwinos.models.llm_routing.client import LLMResponse

INDEX_FILENAME = "index.json"
CACHE_FORMAT_VERSION = 1


class ResponseCache:
    """LRU-bounded, content-addressed JSON file cache for ``LLMResponse`` objects.

    Each entry lives at ``<directory>/<key>.json`` where ``key`` is the SHA-256 of
    the canonicalised request (model + messages + optional JSON schema). Access
    order is persisted in ``<directory>/index.json`` so eviction survives process
    restarts; on load the index is reconciled against the files actually present.

    Thread-safe within a process via an internal lock. Not designed for concurrent
    multi-process writers (last writer wins on the index), which matches its role
    as a per-deployment local cache.
    """

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        max_entries: int = 256,
        settings: Settings | None = None,
    ) -> None:
        """Open (and create if needed) a cache directory.

        Args:
            directory: Cache location. Defaults to ``settings.data_dir / "llm_cache"``.
            max_entries: Maximum number of cached responses kept on disk; the least
                recently used entries beyond this are deleted on ``put``.
            settings: Optional settings override (used only for the default path).
        """
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if directory is not None:
            self.directory = Path(directory)
        else:
            cfg = settings or get_settings()
            self.directory = cfg.data_dir / "llm_cache"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_entries = int(max_entries)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._counter = 0
        self._index: dict[str, int] = {}  # key -> last-access tick (higher = fresher)
        self._index_path = self.directory / INDEX_FILENAME
        self._load_index()
        with self._lock:
            self._evict_locked()
            self._save_index_locked()

    # -- keys ---------------------------------------------------------------------

    @staticmethod
    def compute_key(
        model: str,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Deterministic SHA-256 content key for a request triple."""
        body = json.dumps(
            {"model": model, "messages": messages, "schema": json_schema},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def _entry_path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    # -- core API -----------------------------------------------------------------

    def get(
        self,
        model: str,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse | None:
        """Look up a cached response; refreshes its LRU position on a hit.

        Returns:
            The cached :class:`LLMResponse`, or ``None`` on a miss (including
            corrupted entries, which are deleted on sight).
        """
        key = self.compute_key(model, messages, json_schema)
        with self._lock:
            path = self._entry_path(key)
            if not path.exists():
                self._index.pop(key, None)
                self.misses += 1
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                response = LLMResponse.model_validate(payload["response"])
            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                path.unlink(missing_ok=True)
                self._index.pop(key, None)
                self._save_index_locked()
                self.misses += 1
                return None
            self._touch_locked(key)
            self._save_index_locked()
            self.hits += 1
            return response

    def put(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Store a response, evicting least-recently-used entries beyond capacity.

        Returns:
            The content key the response was stored under.
        """
        key = self.compute_key(model, messages, json_schema)
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "key": key,
            "model": model,
            "created_at": utcnow().isoformat(),
            "response": response.model_dump(mode="json"),
        }
        with self._lock:
            self._entry_path(key).write_text(
                json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
            )
            self._touch_locked(key)
            self._evict_locked()
            self._save_index_locked()
        return key

    def clear(self) -> int:
        """Delete every cached entry (and reset the index). Returns entries removed."""
        with self._lock:
            removed = 0
            for path in self.directory.glob("*.json"):
                if path.name == INDEX_FILENAME:
                    continue
                path.unlink(missing_ok=True)
                removed += 1
            self._index.clear()
            self._counter = 0
            self._save_index_locked()
            return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._index

    def stats(self) -> dict[str, Any]:
        """Hit/miss counters and capacity, for observability surfaces."""
        with self._lock:
            return {
                "entries": len(self._index),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "directory": str(self.directory),
            }

    # -- internals -------------------------------------------------------------------

    def _touch_locked(self, key: str) -> None:
        self._counter += 1
        self._index[key] = self._counter

    def _evict_locked(self) -> None:
        while len(self._index) > self.max_entries:
            oldest = min(self._index, key=self._index.__getitem__)
            self._index.pop(oldest)
            self._entry_path(oldest).unlink(missing_ok=True)

    def _save_index_locked(self) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"counter": self._counter, "entries": self._index}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._index_path)

    def _load_index(self) -> None:
        entries: dict[str, int] = {}
        counter = 0
        if self._index_path.exists():
            try:
                raw = json.loads(self._index_path.read_text(encoding="utf-8"))
                counter = int(raw.get("counter", 0))
                entries = {str(k): int(v) for k, v in raw.get("entries", {}).items()}
            except (json.JSONDecodeError, ValueError, OSError):
                entries = {}
                counter = 0
        # Reconcile against the files actually on disk.
        on_disk = {
            p.stem for p in self.directory.glob("*.json") if p.name != INDEX_FILENAME
        }
        entries = {k: v for k, v in entries.items() if k in on_disk}
        for key in on_disk - entries.keys():
            entries[key] = 0  # unknown history: first in line for eviction
        self._index = entries
        self._counter = max([counter, *entries.values()], default=0)
