"""Tests for the AML subgraph scorer, AUC and precision-recall utilities."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from fintwinos.models.graph.aml_scorer import (
    AmlSubgraphScorer,
    precision_recall_curve,
    roc_auc,
)
from fintwinos.models.graph.features import FEATURE_NAMES, feature_matrix


def make_ring_vs_normal_graph(
    seed: int = 11, n_normal: int = 90, n_rings: int = 7, ring_size: int = 6
) -> nx.DiGraph:
    """Seeded synthetic transaction graph: ordinary accounts plus laundering rings.

    Normal accounts pay a handful of random counterparties varied amounts. Ring
    members move a near-constant large amount around a cycle (high flow
    concentration, deep chains), with one reciprocated edge per ring (2-cycles)
    and a couple of feeder payments from normal accounts.
    """
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    normal = [f"acct_{i}" for i in range(n_normal)]
    for name in normal:
        g.add_node(name, is_suspicious=False)
    for name in normal:
        k = int(rng.integers(2, 6))
        targets = rng.choice(n_normal, size=k, replace=False)
        for t in targets:
            if normal[int(t)] != name:
                g.add_edge(name, normal[int(t)], amount=float(rng.uniform(50, 500)))
    for r in range(n_rings):
        members = [f"ring_{r}_{i}" for i in range(ring_size)]
        for m in members:
            g.add_node(m, is_suspicious=True)
        amount = float(rng.uniform(9000, 9900))
        for i in range(ring_size):
            g.add_edge(members[i], members[(i + 1) % ring_size], amount=amount)
        g.add_edge(members[1], members[0], amount=amount)  # round-trip motif
        feeders = rng.choice(n_normal, size=2, replace=False)
        for f in feeders:
            g.add_edge(normal[int(f)], members[0], amount=float(rng.uniform(100, 300)))
    return g


def graph_labels(g: nx.DiGraph, nodes: list) -> np.ndarray:
    return np.array([1.0 if g.nodes[n]["is_suspicious"] else 0.0 for n in nodes])


class TestRocAuc:
    def test_perfect_separation_is_one(self):
        assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_inverted_separation_is_zero(self):
        assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0

    def test_all_tied_scores_is_half(self):
        assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_random_scores_near_half(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 4000)
        s = rng.uniform(size=4000)
        assert roc_auc(y, s) == pytest.approx(0.5, abs=0.05)

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            roc_auc([1, 1, 1], [0.1, 0.2, 0.3])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            roc_auc([1, 0], [0.1, 0.2, 0.3])


class TestPrecisionRecallCurve:
    def test_anchor_point_and_monotone_recall(self):
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        s = np.array([0.1, 0.9, 0.3, 0.8, 0.4, 0.2, 0.6, 0.7])
        precision, recall, thresholds = precision_recall_curve(y, s)
        assert precision[0] == 1.0 and recall[0] == 0.0 and thresholds[0] == np.inf
        assert np.all(np.diff(recall) >= 0)
        assert recall[-1] == 1.0
        assert len(precision) == len(recall) == len(thresholds)

    def test_perfect_scores_keep_precision_one_until_full_recall(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        precision, recall, _ = precision_recall_curve(y, s)
        assert precision[recall == 1.0][0] == 1.0

    def test_no_positives_raises(self):
        with pytest.raises(ValueError):
            precision_recall_curve([0, 0], [0.4, 0.6])


class TestAmlSubgraphScorer:
    def test_separates_rings_from_normal_holdout_auc(self):
        g = make_ring_vs_normal_graph(seed=11)
        X, nodes = feature_matrix(g)
        y = graph_labels(g, nodes)
        rng = np.random.default_rng(23)
        perm = rng.permutation(len(nodes))
        split = int(0.7 * len(nodes))
        train, test = perm[:split], perm[split:]

        scorer = AmlSubgraphScorer(seed=7)
        scorer.fit(X[train], y[train])
        auc = roc_auc(y[test], scorer.predict_proba(X[test]))
        assert auc > 0.8

    def test_fit_is_deterministic(self):
        g = make_ring_vs_normal_graph(seed=3, n_normal=40, n_rings=3)
        X, nodes = feature_matrix(g)
        y = graph_labels(g, nodes)
        a = AmlSubgraphScorer(seed=7)
        b = AmlSubgraphScorer(seed=7)
        a.fit(X, y)
        b.fit(X, y)
        np.testing.assert_array_equal(a.weights_, b.weights_)
        assert a.bias_ == b.bias_

    def test_train_on_simulator_reads_node_labels(self):
        g = make_ring_vs_normal_graph(seed=5, n_normal=50, n_rings=4)
        scorer = AmlSubgraphScorer(seed=7)
        summary = scorer.train_on_simulator(g)
        assert summary["auc"] > 0.85
        assert summary["n_pos"] == 4 * 6

    def test_train_on_simulator_accepts_wrapper_objects(self):
        class SimResult:
            def __init__(self, graph):
                self.graph = graph

        g = make_ring_vs_normal_graph(seed=5, n_normal=40, n_rings=3)
        scorer = AmlSubgraphScorer(seed=7)
        summary = scorer.train_on_simulator(SimResult(g))
        assert summary["n"] == g.number_of_nodes()

    def test_train_on_simulator_rejects_unlabelled_graph(self):
        g = nx.DiGraph()
        g.add_edge("a", "b", amount=1.0)
        with pytest.raises(ValueError, match="label"):
            AmlSubgraphScorer().train_on_simulator(g)

    def test_score_graph_and_subgraph(self):
        g = make_ring_vs_normal_graph(seed=9, n_normal=50, n_rings=4)
        scorer = AmlSubgraphScorer(seed=7)
        scorer.train_on_simulator(g)
        scores = scorer.score_graph(g)
        ring_scores = [s for n, s in scores.items() if g.nodes[n]["is_suspicious"]]
        normal_scores = [s for n, s in scores.items() if not g.nodes[n]["is_suspicious"]]
        assert np.mean(ring_scores) > np.mean(normal_scores)
        ring_only = g.subgraph([n for n in g if str(n).startswith("ring_0_")])
        assert scorer.score_subgraph(ring_only) == pytest.approx(
            max(scorer.score_graph(ring_only).values())
        )

    def test_save_load_round_trip_preserves_predictions(self, tmp_path):
        g = make_ring_vs_normal_graph(seed=2, n_normal=40, n_rings=3)
        X, nodes = feature_matrix(g)
        y = graph_labels(g, nodes)
        scorer = AmlSubgraphScorer(seed=7)
        scorer.fit(X, y)
        path = tmp_path / "aml_scorer.json"
        scorer.save(path)

        restored = AmlSubgraphScorer.load(path)
        np.testing.assert_allclose(restored.predict_proba(X), scorer.predict_proba(X))
        assert restored.feature_names == list(FEATURE_NAMES)

    def test_load_rejects_foreign_json(self, tmp_path):
        path = tmp_path / "other.json"
        path.write_text('{"model": "SomethingElse"}')
        with pytest.raises(ValueError):
            AmlSubgraphScorer.load(path)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            AmlSubgraphScorer().predict_proba(np.zeros((1, len(FEATURE_NAMES))))

    def test_single_class_labels_raise(self):
        X = np.zeros((4, len(FEATURE_NAMES)))
        with pytest.raises(ValueError):
            AmlSubgraphScorer().fit(X, [0, 0, 0, 0])

    def test_predict_returns_binary_labels(self):
        g = make_ring_vs_normal_graph(seed=2, n_normal=40, n_rings=3)
        X, nodes = feature_matrix(g)
        y = graph_labels(g, nodes)
        scorer = AmlSubgraphScorer(seed=7)
        scorer.fit(X, y)
        preds = scorer.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})
