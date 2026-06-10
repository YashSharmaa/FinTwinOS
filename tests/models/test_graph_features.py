"""Tests for egonet/motif features on transaction graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from fintwinos.models.graph.features import (
    FEATURE_NAMES,
    egonet,
    feature_matrix,
    node_features,
)


def _handcrafted_graph() -> nx.DiGraph:
    """a<->b round trip, a->c, c->b closing a triangle, then a c->d->e chain."""
    g = nx.DiGraph()
    g.add_edge("a", "b", amount=100.0)
    g.add_edge("b", "a", amount=50.0)
    g.add_edge("a", "c", amount=300.0)
    g.add_edge("c", "b", amount=25.0)
    g.add_edge("c", "d", amount=10.0)
    g.add_edge("d", "e", amount=10.0)
    g.add_node("z")  # isolated
    return g


class TestNodeFeatures:
    def test_feature_names_complete_and_ordered(self):
        feats = node_features(_handcrafted_graph(), "a")
        assert list(feats.keys()) == list(FEATURE_NAMES)

    def test_degrees_and_fans(self):
        feats = node_features(_handcrafted_graph(), "a")
        assert feats["in_degree"] == 1.0  # b -> a
        assert feats["out_degree"] == 2.0  # a -> b, a -> c
        assert feats["fan_in"] == 1.0
        assert feats["fan_out"] == 2.0

    def test_two_cycles_counts_reciprocated_counterparties(self):
        feats = node_features(_handcrafted_graph(), "a")
        assert feats["two_cycles"] == 1.0  # only b has flow both ways with a

    def test_triangle_count(self):
        feats = node_features(_handcrafted_graph(), "a")
        # neighbours of a are {b, c}; the edge c -> b closes one triangle
        assert feats["triangles"] == 1.0

    def test_unique_counterparties(self):
        feats = node_features(_handcrafted_graph(), "a")
        assert feats["unique_counterparties"] == 2.0

    def test_flow_herfindahl_exact_value(self):
        feats = node_features(_handcrafted_graph(), "a")
        # flows: b = 100 + 50 = 150, c = 300; shares 1/3 and 2/3 -> H = 5/9
        assert feats["flow_herfindahl"] == pytest.approx(5.0 / 9.0)

    def test_chain_depth_follows_out_edges(self):
        feats = node_features(_handcrafted_graph(), "a")
        # a -> c -> d -> e is the deepest downstream path: 3 hops
        assert feats["chain_depth"] == 3.0

    def test_chain_depth_respects_cap(self):
        g = nx.path_graph(12, create_using=nx.DiGraph)
        feats = node_features(g, 0, max_chain_depth=4)
        assert feats["chain_depth"] == 4.0

    def test_isolated_node_is_all_zero(self):
        feats = node_features(_handcrafted_graph(), "z")
        assert all(v == 0.0 for v in feats.values())

    def test_missing_node_raises(self):
        with pytest.raises(KeyError):
            node_features(_handcrafted_graph(), "ghost")

    def test_self_loops_are_ignored(self):
        g = nx.DiGraph()
        g.add_edge("a", "a", amount=999.0)
        g.add_edge("a", "b", amount=10.0)
        feats = node_features(g, "a")
        assert feats["in_degree"] == 0.0
        assert feats["out_degree"] == 1.0
        assert feats["flow_herfindahl"] == 1.0

    def test_missing_amount_falls_back_to_weight_then_one(self):
        g = nx.DiGraph()
        g.add_edge("a", "b", weight=4.0)
        g.add_edge("a", "c")  # no annotation at all -> 1.0
        feats = node_features(g, "a")
        assert feats["flow_herfindahl"] == pytest.approx((4 / 5) ** 2 + (1 / 5) ** 2)

    def test_multigraph_parallel_edges_count_in_degree_not_fan(self):
        g = nx.MultiDiGraph()
        g.add_edge("x", "a", amount=10.0)
        g.add_edge("x", "a", amount=20.0)
        feats = node_features(g, "a")
        assert feats["in_degree"] == 2.0
        assert feats["fan_in"] == 1.0
        assert feats["flow_herfindahl"] == 1.0

    def test_undirected_graph_is_accepted(self):
        g = nx.Graph()
        g.add_edge("a", "b", amount=10.0)
        g.add_edge("a", "c", amount=10.0)
        feats = node_features(g, "a")
        assert feats["unique_counterparties"] == 2.0
        assert feats["fan_in"] == 2.0 and feats["fan_out"] == 2.0


class TestFeatureMatrix:
    def test_shape_and_row_order(self):
        g = _handcrafted_graph()
        nodes = ["a", "b", "z"]
        X, returned = feature_matrix(g, nodes)
        assert X.shape == (3, len(FEATURE_NAMES))
        assert returned == nodes
        np.testing.assert_allclose(
            X[0], [node_features(g, "a")[n] for n in FEATURE_NAMES]
        )
        assert np.all(X[2] == 0.0)  # isolated node row

    def test_defaults_to_all_nodes(self):
        g = _handcrafted_graph()
        X, returned = feature_matrix(g)
        assert X.shape == (g.number_of_nodes(), len(FEATURE_NAMES))
        assert returned == list(g.nodes())


class TestEgonet:
    def test_radius_one_includes_both_directions(self):
        g = _handcrafted_graph()
        ego = egonet(g, "a", radius=1)
        assert set(ego.nodes()) == {"a", "b", "c"}
        assert ego.has_edge("c", "b")  # edge between neighbours retained

    def test_radius_two_expands(self):
        g = _handcrafted_graph()
        ego = egonet(g, "a", radius=2)
        assert "d" in ego.nodes()
