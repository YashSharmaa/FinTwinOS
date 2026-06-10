"""NetworkXGraphStore: entities, typed relationships, traversal, subgraphs, stats."""

from __future__ import annotations

import networkx as nx
import pytest

from fintwinos.core.types import EntityRef
from fintwinos.twin_core.graph import NetworkXGraphStore, ref_from_key


def _ref(entity_type: str, entity_id: str) -> EntityRef:
    return EntityRef(entity_type=entity_type, entity_id=entity_id)


@pytest.fixture()
def store() -> NetworkXGraphStore:
    s = NetworkXGraphStore()
    s.upsert_entity(_ref("customer", "c1"), {"name": "Acme Corp Ltd", "segment": "sme"})
    s.upsert_entity(_ref("account", "a1"), {"currency": "USD"})
    s.upsert_entity(_ref("account", "a2"), {"currency": "EUR"})
    s.upsert_entity(_ref("instrument", "i1"), {"symbol": "ALPH"})
    s.add_relationship(_ref("customer", "c1"), _ref("account", "a1"), "owns")
    s.add_relationship(_ref("customer", "c1"), _ref("account", "a2"), "owns")
    s.add_relationship(_ref("account", "a1"), _ref("instrument", "i1"), "holds", {"quantity": 10})
    return s


def test_upsert_and_get_roundtrip(store):
    entity = store.get_entity(_ref("customer", "c1"))
    assert entity is not None
    assert entity["name"] == "Acme Corp Ltd"
    assert entity["entity_type"] == "customer"
    assert entity["entity_id"] == "c1"


def test_upsert_merges_attributes(store):
    store.upsert_entity(_ref("customer", "c1"), {"segment": "corporate", "country": "GB"})
    entity = store.get_entity(_ref("customer", "c1"))
    assert entity["segment"] == "corporate"
    assert entity["country"] == "GB"
    assert entity["name"] == "Acme Corp Ltd"  # untouched attributes survive


def test_upsert_cannot_override_meta_fields(store):
    store.upsert_entity(_ref("customer", "c1"), {"entity_type": "hacked", "entity_id": "x"})
    entity = store.get_entity(_ref("customer", "c1"))
    assert entity["entity_type"] == "customer"
    assert entity["entity_id"] == "c1"


def test_get_entity_missing_returns_none(store):
    assert store.get_entity(_ref("customer", "nope")) is None


def test_relationship_is_typed_and_unique_per_kind(store):
    # re-adding the same (src, dst, kind) merges attributes instead of duplicating
    store.add_relationship(
        _ref("account", "a1"), _ref("instrument", "i1"), "holds", {"quantity": 25}
    )
    edge = store.get_relationship(_ref("account", "a1"), _ref("instrument", "i1"), "holds")
    assert edge == {"quantity": 25}
    assert store.stats()["relationship_kinds"]["holds"] == 1


def test_relationship_auto_creates_endpoints():
    s = NetworkXGraphStore()
    s.add_relationship(_ref("case", "k1"), _ref("customer", "c9"), "involves")
    assert s.get_entity(_ref("case", "k1")) is not None
    assert s.get_entity(_ref("customer", "c9")) is not None


def test_neighbors_kind_filter(store):
    owns = store.neighbors(_ref("customer", "c1"), kind="owns")
    assert {r.key() for r in owns} == {"account:a1", "account:a2"}
    holds = store.neighbors(_ref("customer", "c1"), kind="holds")
    assert holds == []


def test_neighbors_is_bidirectional(store):
    back = store.neighbors(_ref("account", "a1"), kind="owns")
    assert [r.key() for r in back] == ["customer:c1"]


def test_neighbors_depth_two(store):
    two_hops = store.neighbors(_ref("customer", "c1"), depth=2)
    assert {r.key() for r in two_hops} == {"account:a1", "account:a2", "instrument:i1"}
    one_hop = store.neighbors(_ref("customer", "c1"), depth=1)
    assert {r.key() for r in one_hop} == {"account:a1", "account:a2"}


def test_neighbors_unknown_or_zero_depth(store):
    assert store.neighbors(_ref("customer", "missing")) == []
    assert store.neighbors(_ref("customer", "c1"), depth=0) == []


def test_subgraph_returns_networkx_graph(store):
    sub = store.subgraph([_ref("customer", "c1")], depth=1)
    assert isinstance(sub, nx.MultiDiGraph)
    assert set(sub.nodes) == {"customer:c1", "account:a1", "account:a2"}
    assert sub.number_of_edges() == 2
    # the subgraph is a copy: mutating it must not affect the store
    sub.add_node("rogue:x")
    assert store.get_entity(_ref("rogue", "x")) is None


def test_subgraph_depth_two_reaches_instrument(store):
    sub = store.subgraph([_ref("customer", "c1")], depth=2)
    assert "instrument:i1" in sub.nodes


def test_stats(store):
    stats = store.stats()
    assert stats["entities"] == 4
    assert stats["relationships"] == 3
    assert stats["entity_types"] == {"account": 2, "customer": 1, "instrument": 1}
    assert stats["relationship_kinds"] == {"holds": 1, "owns": 2}


def test_entities_and_relationships_enumeration_sorted(store):
    keys = [ref.key() for ref, _ in store.entities()]
    assert keys == sorted(keys)
    rels = store.relationships()
    assert len(rels) == 3
    assert all(len(item) == 4 for item in rels)


def test_ref_from_key_roundtrip():
    ref = _ref("customer", "c:with:colons")
    assert ref_from_key(ref.key()) == ref
    with pytest.raises(ValueError):
        ref_from_key("malformed")
