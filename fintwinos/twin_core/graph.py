"""NetworkX-backed entity graph for the federated twin.

The graph is the relational backbone of the twin: customers, accounts,
instruments, trades, cases, alerts and policy documents all live here as typed
nodes, connected by typed relationships (``owns``, ``transfers_to``, ``holds``,
``involves``...). Nodes are keyed by ``EntityRef.key()`` (``"<type>:<id>"``) on
a :class:`networkx.MultiDiGraph`, and each relationship kind between a given
pair of entities is a distinct keyed edge, so re-ingesting the same
relationship updates its attributes instead of duplicating it.

Implements the :class:`fintwinos.core.interfaces.GraphStore` protocol.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import networkx as nx

from fintwinos.core.types import EntityRef

_NODE_META_FIELDS = ("entity_type", "entity_id")


def ref_from_key(key: str) -> EntityRef:
    """Rebuild an :class:`EntityRef` from a node key of the form ``type:id``.

    Entity types must not contain ``":"``; entity ids may (only the first
    colon separates type from id).
    """
    entity_type, sep, entity_id = key.partition(":")
    if not sep or not entity_type or not entity_id:
        raise ValueError(f"malformed entity key: {key!r} (expected 'type:id')")
    return EntityRef(entity_type=entity_type, entity_id=entity_id)


class NetworkXGraphStore:
    """In-memory typed property graph over a ``networkx.MultiDiGraph``.

    Every node stores its ``entity_type`` and ``entity_id`` plus arbitrary
    attributes; every edge stores its relationship ``kind`` (which is also the
    multigraph edge key) plus arbitrary attributes. All read methods return
    copies, so callers can never mutate twin state behind the store's back.
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    # -- entities ----------------------------------------------------------------

    def upsert_entity(self, ref: EntityRef, attributes: dict[str, Any] | None = None) -> None:
        """Create the entity if missing, otherwise merge ``attributes`` into it.

        The reserved fields ``entity_type``/``entity_id`` are always derived
        from ``ref`` and cannot be overridden through ``attributes``.
        """
        key = ref.key()
        attrs = {k: v for k, v in (attributes or {}).items() if k not in _NODE_META_FIELDS}
        if self._graph.has_node(key):
            self._graph.nodes[key].update(attrs)
        else:
            self._graph.add_node(
                key, entity_type=ref.entity_type, entity_id=ref.entity_id, **attrs
            )

    def get_entity(self, ref: EntityRef) -> dict[str, Any] | None:
        """Return a copy of the entity's attribute dict, or None if absent."""
        key = ref.key()
        if not self._graph.has_node(key):
            return None
        return dict(self._graph.nodes[key])

    def entities(self) -> list[tuple[EntityRef, dict[str, Any]]]:
        """All entities with their (non-meta) attributes, sorted by key."""
        out: list[tuple[EntityRef, dict[str, Any]]] = []
        for key, data in sorted(self._graph.nodes(data=True)):
            attrs = {k: v for k, v in data.items() if k not in _NODE_META_FIELDS}
            out.append((ref_from_key(key), attrs))
        return out

    # -- relationships -------------------------------------------------------------

    def add_relationship(
        self,
        src: EntityRef,
        dst: EntityRef,
        kind: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Create or update the typed edge ``src -[kind]-> dst``.

        The triple ``(src, dst, kind)`` is unique: repeated calls merge the new
        attributes into the existing edge. Missing endpoints are auto-created
        as bare entities so relationship ingestion never has ordering
        dependencies on entity ingestion.
        """
        for ref in (src, dst):
            if not self._graph.has_node(ref.key()):
                self.upsert_entity(ref)
        skey, dkey = src.key(), dst.key()
        attrs = {k: v for k, v in (attributes or {}).items() if k != "kind"}
        if not self._graph.has_edge(skey, dkey, key=kind):
            self._graph.add_edge(skey, dkey, key=kind, kind=kind)
        self._graph[skey][dkey][kind].update(attrs)

    def get_relationship(self, src: EntityRef, dst: EntityRef, kind: str) -> dict[str, Any] | None:
        """Return a copy of the edge attributes for ``src -[kind]-> dst``, or None."""
        skey, dkey = src.key(), dst.key()
        if not self._graph.has_edge(skey, dkey, key=kind):
            return None
        data = dict(self._graph[skey][dkey][kind])
        data.pop("kind", None)
        return data

    def relationships(self) -> list[tuple[EntityRef, EntityRef, str, dict[str, Any]]]:
        """All edges as ``(src, dst, kind, attributes)``, deterministically sorted."""
        out: list[tuple[EntityRef, EntityRef, str, dict[str, Any]]] = []
        for skey, dkey, kind, data in self._graph.edges(keys=True, data=True):
            attrs = {k: v for k, v in data.items() if k != "kind"}
            out.append((ref_from_key(skey), ref_from_key(dkey), kind, attrs))
        out.sort(key=lambda item: (item[0].key(), item[2], item[1].key()))
        return out

    # -- traversal ---------------------------------------------------------------

    def neighbors(self, ref: EntityRef, kind: str | None = None, depth: int = 1) -> list[EntityRef]:
        """Entities reachable within ``depth`` hops, in either edge direction.

        ``kind`` restricts traversal to edges of that relationship kind. The
        start entity itself is never included. Results are sorted by entity
        key for determinism.
        """
        start = ref.key()
        if depth < 1 or not self._graph.has_node(start):
            return []
        seen: set[str] = {start}
        frontier: list[str] = [start]
        found: set[str] = set()
        for _ in range(depth):
            nxt: list[str] = []
            for node in frontier:
                for _, dst, edge_kind in self._graph.out_edges(node, keys=True):
                    if (kind is None or edge_kind == kind) and dst not in seen:
                        seen.add(dst)
                        found.add(dst)
                        nxt.append(dst)
                for src, _, edge_kind in self._graph.in_edges(node, keys=True):
                    if (kind is None or edge_kind == kind) and src not in seen:
                        seen.add(src)
                        found.add(src)
                        nxt.append(src)
            if not nxt:
                break
            frontier = nxt
        return [ref_from_key(k) for k in sorted(found)]

    def subgraph(self, refs: Iterable[EntityRef], depth: int = 1) -> nx.MultiDiGraph:
        """Extract the induced subgraph around ``refs`` expanded ``depth`` hops.

        Returns an independent :class:`networkx.MultiDiGraph` copy containing
        the seed entities (those present in the twin), every entity within
        ``depth`` undirected hops of a seed, and all edges among them.
        """
        nodes = {r.key() for r in refs if self._graph.has_node(r.key())}
        frontier = set(nodes)
        for _ in range(max(depth, 0)):
            nxt: set[str] = set()
            for node in frontier:
                nxt.update(self._graph.successors(node))
                nxt.update(self._graph.predecessors(node))
            nxt -= nodes
            if not nxt:
                break
            nodes |= nxt
            frontier = nxt
        return self._graph.subgraph(nodes).copy()

    # -- introspection -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Counts of entities and relationships, broken down by type and kind."""
        type_counts: Counter[str] = Counter(
            data.get("entity_type", "unknown") for _, data in self._graph.nodes(data=True)
        )
        kind_counts: Counter[str] = Counter(
            kind for _, _, kind in self._graph.edges(keys=True)
        )
        return {
            "entities": self._graph.number_of_nodes(),
            "relationships": self._graph.number_of_edges(),
            "entity_types": dict(sorted(type_counts.items())),
            "relationship_kinds": dict(sorted(kind_counts.items())),
        }
