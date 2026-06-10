"""Tests for the Elliptic2-layout loader against the bundled synthetic fixture."""

from __future__ import annotations

import shutil

import networkx as nx
import numpy as np
import pytest

from fintwinos.datasets.elliptic2_loader import (
    LABEL_LICIT,
    LABEL_SUSPICIOUS,
    LABEL_UNKNOWN,
    Elliptic2Dataset,
    elliptic2_fixture_dir,
    load_elliptic2,
)


@pytest.fixture(scope="module")
def fixture_ds() -> Elliptic2Dataset:
    return load_elliptic2(elliptic2_fixture_dir())


class TestFixtureLoads:
    def test_fixture_dir_exists(self):
        d = elliptic2_fixture_dir()
        assert d.is_dir()
        for name in ("nodes.csv", "edges.csv", "connected_components.csv"):
            assert (d / name).is_file()

    def test_graph_shape(self, fixture_ds):
        assert isinstance(fixture_ds.graph, nx.DiGraph)
        assert fixture_ds.graph.number_of_nodes() == 14
        assert fixture_ds.graph.number_of_edges() == 14

    def test_labels_aligned_with_node_ids(self, fixture_ds):
        assert len(fixture_ds.node_ids) == 14
        assert fixture_ds.labels.shape == (14,)
        assert fixture_ds.labels.dtype == np.int8
        by_id = dict(zip(fixture_ds.node_ids, fixture_ds.labels, strict=True))
        assert by_id["n001"] == LABEL_SUSPICIOUS
        assert by_id["n006"] == LABEL_SUSPICIOUS
        assert by_id["n007"] == LABEL_LICIT
        assert by_id["n012"] == LABEL_UNKNOWN

    def test_label_counts(self, fixture_ds):
        assert int(np.sum(fixture_ds.labels == LABEL_SUSPICIOUS)) == 6
        assert int(np.sum(fixture_ds.labels == LABEL_LICIT)) == 5
        assert int(np.sum(fixture_ds.labels == LABEL_UNKNOWN)) == 3

    def test_component_labels(self, fixture_ds):
        assert fixture_ds.component_labels == {101: LABEL_SUSPICIOUS, 202: LABEL_LICIT}

    def test_node_attributes_match_arrays(self, fixture_ds):
        for node_id, cc_id, label in zip(
            fixture_ds.node_ids, fixture_ds.cc_ids, fixture_ds.labels, strict=True
        ):
            attrs = fixture_ds.graph.nodes[str(node_id)]
            assert attrs["cc_id"] == int(cc_id)
            assert attrs["label"] == int(label)

    def test_features_extracted(self, fixture_ds):
        assert fixture_ds.features is not None
        assert fixture_ds.features.shape == (14, 2)
        assert fixture_ds.feature_names == ["feat_degree", "feat_volume"]

    def test_suspicious_component_contains_fan_in(self, fixture_ds):
        """The fixture's suspicious component embeds a fan-in onto n006."""
        assert fixture_ds.graph.in_degree("n006") >= 4

    def test_summary(self, fixture_ds):
        summary = fixture_ds.summary()
        assert summary["n_nodes"] == 14
        assert summary["n_suspicious"] == 6
        assert summary["n_components"] == 2
        assert summary["n_features"] == 2


class TestErrorsAndConfigurability:
    def test_missing_dir_mentions_source_and_licence(self, tmp_path):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_elliptic2(tmp_path / "absent")
        message = str(excinfo.value)
        assert "Elliptic2" in message
        assert "github.com/MITIBMxGraph/Elliptic2" in message
        assert "licence" in message

    def test_missing_file_mentions_path(self, tmp_path):
        shutil.copy(elliptic2_fixture_dir() / "nodes.csv", tmp_path / "nodes.csv")
        with pytest.raises(FileNotFoundError, match="edges.csv"):
            load_elliptic2(tmp_path)

    def test_configurable_filenames(self, tmp_path):
        src = elliptic2_fixture_dir()
        shutil.copy(src / "nodes.csv", tmp_path / "my_nodes.csv")
        shutil.copy(src / "edges.csv", tmp_path / "my_edges.csv")
        shutil.copy(src / "connected_components.csv", tmp_path / "my_labels.csv")
        ds = load_elliptic2(
            tmp_path,
            nodes_file="my_nodes.csv",
            edges_file="my_edges.csv",
            labels_file="my_labels.csv",
        )
        assert ds.graph.number_of_nodes() == 14
        assert ds.component_labels == {101: LABEL_SUSPICIOUS, 202: LABEL_LICIT}

    def test_edges_referencing_unknown_nodes_are_added(self, tmp_path):
        src = elliptic2_fixture_dir()
        shutil.copy(src / "nodes.csv", tmp_path / "nodes.csv")
        shutil.copy(src / "connected_components.csv", tmp_path / "connected_components.csv")
        (tmp_path / "edges.csv").write_text(
            "clId1,clId2\nn001,n999\n", encoding="utf-8"
        )
        ds = load_elliptic2(tmp_path)
        assert "n999" in ds.graph
        assert ds.graph.nodes["n999"]["label"] == LABEL_UNKNOWN

    def test_bad_edges_file_raises_value_error(self, tmp_path):
        src = elliptic2_fixture_dir()
        shutil.copy(src / "nodes.csv", tmp_path / "nodes.csv")
        shutil.copy(src / "connected_components.csv", tmp_path / "connected_components.csv")
        (tmp_path / "edges.csv").write_text("only_one_column\nx\n", encoding="utf-8")
        with pytest.raises(ValueError, match="two columns"):
            load_elliptic2(tmp_path)
