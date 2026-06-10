"""Egonet and motif features for AML detection on transaction graphs.

The twin's relationship layer stores money movements as a directed graph: nodes are
accounts (or customers / legal entities) and each edge carries an ``amount`` attribute
with the value transferred. Money-laundering typologies leave structural fingerprints
on that graph — fan-in collection points, round-trip 2-cycles, tightly knit triangles,
flow concentrated on a single counterparty, long layering chains — and this module
turns those fingerprints into a fixed-width numeric feature vector per node.

The features are intentionally classical and fully explainable: every value can be
recomputed by hand from the egonet, which matters when an alert lands in front of a
financial-crime investigator or a regulator.

Feature definitions (one row per node, columns in ``FEATURE_NAMES`` order):

- ``in_degree`` / ``out_degree``: number of incoming / outgoing transfer edges
  (parallel edges in a multigraph each count; self-loops are ignored).
- ``fan_in`` / ``fan_out``: number of *unique* sending / receiving counterparties.
- ``two_cycles``: number of counterparties with flow in **both** directions
  (a -> b and b -> a), the classic round-trip motif.
- ``triangles``: number of triangles through the node in the undirected projection
  of its radius-1 egonet (closed rings of three accounts).
- ``unique_counterparties``: distinct accounts transacted with in either direction.
- ``flow_herfindahl``: Herfindahl–Hirschman concentration of total flow per
  counterparty, in ``(0, 1]``; 1.0 means all value moves with one counterparty.
- ``chain_depth``: number of hops to the furthest account reachable downstream via
  out-edges (BFS levels, capped at ``max_chain_depth``) — a proxy for layering depth.

Undirected input graphs are accepted and converted to a directed view with reciprocal
edges; the directional features (2-cycles in particular) are most meaningful on
genuinely directed transaction data.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import networkx as nx
import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "fan_in",
    "fan_out",
    "two_cycles",
    "triangles",
    "unique_counterparties",
    "flow_herfindahl",
    "chain_depth",
)
"""Column order of every feature vector produced by this module."""

DEFAULT_AMOUNT_KEY = "amount"
DEFAULT_MAX_CHAIN_DEPTH = 6


def _as_directed(graph: nx.Graph) -> nx.DiGraph:
    """Return ``graph`` itself if directed, else a directed view with reciprocal edges."""
    if graph.is_directed():
        return graph
    return graph.to_directed(as_view=True)


def _edge_amount(data: Mapping[str, Any], amount_key: str) -> float:
    """Extract a transfer amount from edge attributes.

    Falls back to ``weight`` and finally to ``1.0`` (pure count semantics) so the
    features degrade gracefully on graphs without monetary annotations.
    """
    raw = data.get(amount_key, data.get("weight", 1.0))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(value):
        return 1.0
    return abs(value)


def _chain_depth(g: nx.DiGraph, node: Hashable, max_depth: int) -> int:
    """BFS depth of the furthest node reachable downstream via out-edges, capped."""
    visited: set[Hashable] = {node}
    frontier: list[Hashable] = [node]
    depth = 0
    while frontier and depth < max_depth:
        nxt: list[Hashable] = []
        for u in frontier:
            for v in g.successors(u):
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
        if not nxt:
            break
        depth += 1
        frontier = nxt
    return depth


def egonet(graph: nx.Graph, node: Hashable, radius: int = 1) -> nx.Graph:
    """Return the egonet of ``node``: the subgraph within ``radius`` hops.

    Direction is ignored when discovering the neighbourhood (both senders and
    receivers belong to an account's egonet) but the returned subgraph keeps the
    original directedness and edge attributes.
    """
    return nx.ego_graph(graph, node, radius=radius, undirected=True)


def node_features(
    graph: nx.Graph,
    node: Hashable,
    *,
    amount_key: str = DEFAULT_AMOUNT_KEY,
    max_chain_depth: int = DEFAULT_MAX_CHAIN_DEPTH,
) -> dict[str, float]:
    """Compute the full egonet/motif feature dictionary for one node.

    Args:
        graph: A networkx (di)graph of transfers. Multigraphs are supported; each
            parallel edge counts towards degrees and contributes its own amount.
        node: The node to featurise. Must exist in ``graph``.
        amount_key: Edge attribute holding the monetary value of a transfer.
        max_chain_depth: Cap on the downstream BFS used for ``chain_depth``.

    Returns:
        A dict with exactly the keys in :data:`FEATURE_NAMES`, all ``float``.

    Raises:
        KeyError: If ``node`` is not in the graph.
    """
    g = _as_directed(graph)
    if node not in g:
        raise KeyError(f"node {node!r} is not in the graph")

    in_degree = 0
    out_degree = 0
    fan_in: set[Hashable] = set()
    fan_out: set[Hashable] = set()
    flows: dict[Hashable, float] = {}

    for u, _, data in g.in_edges(node, data=True):
        if u == node:  # self-loops carry no counterparty signal
            continue
        in_degree += 1
        fan_in.add(u)
        flows[u] = flows.get(u, 0.0) + _edge_amount(data, amount_key)

    for _, v, data in g.out_edges(node, data=True):
        if v == node:
            continue
        out_degree += 1
        fan_out.add(v)
        flows[v] = flows.get(v, 0.0) + _edge_amount(data, amount_key)

    two_cycles = float(sum(1 for v in fan_out if v in fan_in))

    neighbours = list(fan_in | fan_out)
    triangles = 0
    for i in range(len(neighbours)):
        for j in range(i + 1, len(neighbours)):
            a, b = neighbours[i], neighbours[j]
            if g.has_edge(a, b) or g.has_edge(b, a):
                triangles += 1

    total_flow = sum(flows.values())
    if total_flow > 0.0:
        herfindahl = float(sum((f / total_flow) ** 2 for f in flows.values()))
    else:
        herfindahl = 0.0

    return {
        "in_degree": float(in_degree),
        "out_degree": float(out_degree),
        "fan_in": float(len(fan_in)),
        "fan_out": float(len(fan_out)),
        "two_cycles": two_cycles,
        "triangles": float(triangles),
        "unique_counterparties": float(len(neighbours)),
        "flow_herfindahl": herfindahl,
        "chain_depth": float(_chain_depth(g, node, max_chain_depth)),
    }


def feature_matrix(
    graph: nx.Graph,
    nodes: Iterable[Hashable] | None = None,
    *,
    amount_key: str = DEFAULT_AMOUNT_KEY,
    max_chain_depth: int = DEFAULT_MAX_CHAIN_DEPTH,
) -> tuple[np.ndarray, list[Hashable]]:
    """Featurise many nodes at once.

    Args:
        graph: The transaction graph.
        nodes: Nodes to featurise, in order. Defaults to ``graph.nodes()`` order.
        amount_key: Edge attribute holding the monetary value of a transfer.
        max_chain_depth: Cap on the downstream BFS used for ``chain_depth``.

    Returns:
        ``(X, node_list)`` where ``X`` has shape ``(len(node_list), len(FEATURE_NAMES))``
        with columns in :data:`FEATURE_NAMES` order, and ``node_list`` preserves the
        requested node order so rows can be mapped back to entities.
    """
    node_list = list(nodes) if nodes is not None else list(graph.nodes())
    X = np.zeros((len(node_list), len(FEATURE_NAMES)), dtype=np.float64)
    for i, node in enumerate(node_list):
        feats = node_features(
            graph, node, amount_key=amount_key, max_chain_depth=max_chain_depth
        )
        X[i, :] = [feats[name] for name in FEATURE_NAMES]
    return X, node_list
