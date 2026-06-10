"""Tests for the content-addressed LLM response cache (round-trip, LRU, persistence)."""

from __future__ import annotations

import json

import pytest

from fintwinos.models.llm_routing.cache import INDEX_FILENAME, ResponseCache
from fintwinos.models.llm_routing.client import LLMResponse


def make_response(text: str = "hello", cost: float = 0.001) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="gpt-5-mini",
        usage={"input_tokens": 10, "output_tokens": 5},
        cost_usd=cost,
    )


def messages_for(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


class TestKeying:
    def test_key_is_deterministic(self):
        k1 = ResponseCache.compute_key("gpt-5", messages_for("hi"), {"type": "object"})
        k2 = ResponseCache.compute_key("gpt-5", messages_for("hi"), {"type": "object"})
        assert k1 == k2
        assert len(k1) == 64

    def test_key_varies_with_each_component(self):
        base = ResponseCache.compute_key("gpt-5", messages_for("hi"), None)
        assert ResponseCache.compute_key("gpt-5-mini", messages_for("hi"), None) != base
        assert ResponseCache.compute_key("gpt-5", messages_for("bye"), None) != base
        assert ResponseCache.compute_key("gpt-5", messages_for("hi"), {"a": 1}) != base


class TestRoundTrip:
    def test_put_then_get(self, tmp_path):
        cache = ResponseCache(tmp_path)
        response = make_response()
        cache.put("gpt-5-mini", messages_for("hi"), response)
        cached = cache.get("gpt-5-mini", messages_for("hi"))
        assert cached is not None
        assert cached.model_dump() == response.model_dump()

    def test_miss_returns_none_and_counts(self, tmp_path):
        cache = ResponseCache(tmp_path)
        assert cache.get("gpt-5", messages_for("never seen")) is None
        assert cache.stats()["misses"] == 1
        assert cache.stats()["hits"] == 0

    def test_schema_distinguishes_entries(self, tmp_path):
        cache = ResponseCache(tmp_path)
        cache.put("gpt-5", messages_for("hi"), make_response("with-schema"), {"a": 1})
        assert cache.get("gpt-5", messages_for("hi")) is None  # no-schema variant absent
        cached = cache.get("gpt-5", messages_for("hi"), {"a": 1})
        assert cached is not None and cached.text == "with-schema"

    def test_put_overwrites_same_key(self, tmp_path):
        cache = ResponseCache(tmp_path)
        cache.put("gpt-5", messages_for("hi"), make_response("v1"))
        cache.put("gpt-5", messages_for("hi"), make_response("v2"))
        assert len(cache) == 1
        assert cache.get("gpt-5", messages_for("hi")).text == "v2"

    def test_entries_are_human_readable_json(self, tmp_path):
        cache = ResponseCache(tmp_path)
        key = cache.put("gpt-5", messages_for("hi"), make_response())
        payload = json.loads((tmp_path / f"{key}.json").read_text())
        assert payload["key"] == key
        assert payload["response"]["text"] == "hello"

    def test_corrupted_entry_is_a_miss_and_removed(self, tmp_path):
        cache = ResponseCache(tmp_path)
        key = cache.put("gpt-5", messages_for("hi"), make_response())
        (tmp_path / f"{key}.json").write_text("{not json")
        assert cache.get("gpt-5", messages_for("hi")) is None
        assert len(cache) == 0


class TestLru:
    def test_eviction_beyond_max_entries(self, tmp_path):
        cache = ResponseCache(tmp_path, max_entries=3)
        for i in range(4):
            cache.put("gpt-5", messages_for(f"m{i}"), make_response(f"r{i}"))
        assert len(cache) == 3
        assert cache.get("gpt-5", messages_for("m0")) is None  # oldest evicted
        for i in range(1, 4):
            assert cache.get("gpt-5", messages_for(f"m{i}")) is not None

    def test_get_refreshes_lru_position(self, tmp_path):
        cache = ResponseCache(tmp_path, max_entries=3)
        for i in range(3):
            cache.put("gpt-5", messages_for(f"m{i}"), make_response(f"r{i}"))
        cache.get("gpt-5", messages_for("m0"))  # touch the oldest
        cache.put("gpt-5", messages_for("m3"), make_response("r3"))
        assert cache.get("gpt-5", messages_for("m0")) is not None  # survived
        assert cache.get("gpt-5", messages_for("m1")) is None  # evicted instead

    def test_eviction_removes_files_from_disk(self, tmp_path):
        cache = ResponseCache(tmp_path, max_entries=2)
        for i in range(4):
            cache.put("gpt-5", messages_for(f"m{i}"), make_response())
        entry_files = [p for p in tmp_path.glob("*.json") if p.name != INDEX_FILENAME]
        assert len(entry_files) == 2

    def test_invalid_max_entries_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ResponseCache(tmp_path, max_entries=0)


class TestClearAndPersistence:
    def test_clear_empties_cache(self, tmp_path):
        cache = ResponseCache(tmp_path)
        for i in range(3):
            cache.put("gpt-5", messages_for(f"m{i}"), make_response())
        removed = cache.clear()
        assert removed == 3
        assert len(cache) == 0
        assert cache.get("gpt-5", messages_for("m0")) is None

    def test_cache_survives_reopen(self, tmp_path):
        first = ResponseCache(tmp_path)
        first.put("gpt-5", messages_for("persist me"), make_response("persisted"))
        reopened = ResponseCache(tmp_path)
        cached = reopened.get("gpt-5", messages_for("persist me"))
        assert cached is not None and cached.text == "persisted"

    def test_reopen_with_smaller_capacity_evicts(self, tmp_path):
        first = ResponseCache(tmp_path, max_entries=5)
        for i in range(5):
            first.put("gpt-5", messages_for(f"m{i}"), make_response())
        reopened = ResponseCache(tmp_path, max_entries=2)
        assert len(reopened) == 2

    def test_orphan_files_are_adopted(self, tmp_path):
        first = ResponseCache(tmp_path)
        first.put("gpt-5", messages_for("hi"), make_response())
        (tmp_path / INDEX_FILENAME).unlink()  # lose the index
        reopened = ResponseCache(tmp_path)
        assert len(reopened) == 1
        assert reopened.get("gpt-5", messages_for("hi")) is not None

    def test_stats_shape(self, tmp_path):
        cache = ResponseCache(tmp_path, max_entries=7)
        stats = cache.stats()
        assert stats["max_entries"] == 7
        assert stats["entries"] == 0
        assert stats["directory"] == str(tmp_path)
