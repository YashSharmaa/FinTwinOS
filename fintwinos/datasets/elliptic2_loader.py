"""Loader for the Elliptic2 anti-money-laundering graph dataset.

Elliptic2 is a large graph dataset for money-laundering subgraph detection on the
Bitcoin blockchain, released by Elliptic with the MIT-IBM Watson AI Lab
(https://github.com/MITIBMxGraph/Elliptic2). FinTwinOS does **not** bundle or
redistribute the dataset: download it from the official release, review and accept
the dataset's own licence terms (it is released for research use), and point this
loader at your local copy. A tiny fully synthetic fixture in the same CSV layout
ships at ``datasets/fixtures/elliptic2_sample`` so tests and demos run offline.

Expected CSV layout (filenames configurable):

- ``nodes.csv`` — one row per node. First column is the node id (``clId`` in the
  published files); an optional ``ccId`` column assigns the node to a labelled
  connected component (missing / ``-1`` means background); any remaining numeric
  columns are treated as node features.
- ``edges.csv`` — one row per directed edge; first two columns are source and
  target node ids (``clId1``, ``clId2`` in the published files).
- ``connected_components.csv`` — component labels: ``ccId`` plus ``ccLabel`` with
  values ``suspicious`` or ``licit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

#: Per-node integer labels produced by the loader.
LABEL_SUSPICIOUS = 1
LABEL_LICIT = 0
LABEL_UNKNOWN = -1

LABEL_NAMES: dict[int, str] = {
    LABEL_SUSPICIOUS: "suspicious",
    LABEL_LICIT: "licit",
    LABEL_UNKNOWN: "unknown",
}

_DOWNLOAD_HELP = (
    "The Elliptic2 dataset is not bundled with FinTwinOS. Download it from the "
    "official release (https://github.com/MITIBMxGraph/Elliptic2), review and accept "
    "the dataset's own licence terms (research use; see the release page), and point "
    "data_dir at the directory containing nodes.csv, edges.csv and "
    "connected_components.csv. A small synthetic fixture in the same layout ships at "
    "datasets/fixtures/elliptic2_sample for offline tests and demos "
    "(elliptic2_fixture_dir())."
)

_ID_COLUMN_ALIASES = ["clId", "cl_id", "txId", "node_id", "id"]
_CC_COLUMN_ALIASES = ["ccId", "cc_id", "component_id"]
_CC_LABEL_ALIASES = ["ccLabel", "cc_label", "label"]


@dataclass
class Elliptic2Dataset:
    """The loaded Elliptic2 (or fixture) graph with aligned label arrays.

    Attributes:
        graph: Directed graph; every node carries ``cc_id`` (int, ``-1`` for
            background) and ``label`` (``1`` suspicious / ``0`` licit / ``-1``
            unknown) attributes.
        node_ids: Node identifiers in ``nodes.csv`` row order (numpy object array of
            strings); ``labels`` and ``features`` are aligned to this order.
        labels: ``int8`` array of per-node labels, derived from each node's connected
            component label.
        cc_ids: ``int64`` array of per-node component ids (``-1`` = background).
        component_labels: Mapping ``cc_id -> label int`` for the labelled components.
        features: ``float64`` matrix of any extra numeric node columns, or ``None``
            when the nodes file carries no feature columns.
        feature_names: Column names matching ``features`` columns.
        source_dir: Directory the data was loaded from.
    """

    graph: nx.DiGraph
    node_ids: np.ndarray
    labels: np.ndarray
    cc_ids: np.ndarray
    component_labels: dict[int, int]
    features: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    source_dir: Path | None = None

    def summary(self) -> dict[str, Any]:
        """Counts used by demos and data cards."""
        return {
            "n_nodes": int(self.graph.number_of_nodes()),
            "n_edges": int(self.graph.number_of_edges()),
            "n_suspicious": int(np.sum(self.labels == LABEL_SUSPICIOUS)),
            "n_licit": int(np.sum(self.labels == LABEL_LICIT)),
            "n_unknown": int(np.sum(self.labels == LABEL_UNKNOWN)),
            "n_components": len(self.component_labels),
            "n_features": len(self.feature_names),
        }


def _pick_column(df: pd.DataFrame, aliases: list[str], *, fallback_index: int | None = None) -> str | None:
    """Pick the first matching column by alias (case-insensitive), else by position."""
    lowered = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    if fallback_index is not None and len(df.columns) > fallback_index:
        return str(df.columns[fallback_index])
    return None


def _require_file(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Elliptic2 file not found: {path}. {_DOWNLOAD_HELP}")
    return path


def _label_to_int(raw: Any) -> int:
    text = str(raw).strip().lower()
    if text in {"suspicious", "illicit", "1"}:
        return LABEL_SUSPICIOUS
    if text in {"licit", "legit", "0", "2"}:
        return LABEL_LICIT
    return LABEL_UNKNOWN


def load_elliptic2(
    data_dir: str | Path,
    *,
    nodes_file: str = "nodes.csv",
    edges_file: str = "edges.csv",
    labels_file: str = "connected_components.csv",
) -> Elliptic2Dataset:
    """Load an Elliptic2-layout dataset into a graph plus aligned label arrays.

    Args:
        data_dir: Directory containing the CSV files.
        nodes_file: Nodes CSV filename (default matches the published layout).
        edges_file: Edges CSV filename.
        labels_file: Connected-component labels CSV filename.

    Returns:
        :class:`Elliptic2Dataset` whose ``labels`` array is aligned with ``node_ids``
        (and with the node iteration order implied by ``nodes.csv`` row order).

    Raises:
        FileNotFoundError: When the directory or any required file is missing, with
            instructions pointing to the official Elliptic2 release and its licence.
        ValueError: When a CSV is present but structurally unusable (e.g. an edges
            file with fewer than two columns).
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Elliptic2 data directory not found: {directory}. {_DOWNLOAD_HELP}"
        )
    nodes_path = _require_file(directory, nodes_file)
    edges_path = _require_file(directory, edges_file)
    labels_path = _require_file(directory, labels_file)

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    labels_df = pd.read_csv(labels_path)
    if nodes_df.empty:
        raise ValueError(f"nodes file {nodes_path} contains no rows")
    if len(edges_df.columns) < 2:
        raise ValueError(f"edges file {edges_path} needs at least two columns (src, dst)")

    id_col = _pick_column(nodes_df, _ID_COLUMN_ALIASES, fallback_index=0)
    cc_col = _pick_column(nodes_df, _CC_COLUMN_ALIASES)
    cc_id_col = _pick_column(labels_df, _CC_COLUMN_ALIASES, fallback_index=0)
    cc_label_col = _pick_column(labels_df, _CC_LABEL_ALIASES, fallback_index=1)
    src_col = _pick_column(edges_df, ["clId1", "cl_id1", "src", "source"], fallback_index=0)
    dst_col = _pick_column(edges_df, ["clId2", "cl_id2", "dst", "target"], fallback_index=1)

    component_labels: dict[int, int] = {}
    for _, row in labels_df.iterrows():
        try:
            cc_id = int(row[cc_id_col])
        except (TypeError, ValueError):
            continue
        component_labels[cc_id] = _label_to_int(row[cc_label_col])

    node_ids = nodes_df[id_col].astype(str).to_numpy(dtype=object)
    if cc_col is not None:
        cc_series = pd.to_numeric(nodes_df[cc_col], errors="coerce").fillna(LABEL_UNKNOWN)
        cc_ids = cc_series.astype(np.int64).to_numpy()
    else:
        cc_ids = np.full(len(node_ids), LABEL_UNKNOWN, dtype=np.int64)
    labels = np.array(
        [component_labels.get(int(cc), LABEL_UNKNOWN) for cc in cc_ids], dtype=np.int8
    )

    feature_cols = [
        c for c in nodes_df.columns
        if c not in {id_col, cc_col} and pd.api.types.is_numeric_dtype(nodes_df[c])
    ]
    features = nodes_df[feature_cols].to_numpy(dtype=np.float64) if feature_cols else None

    graph: nx.DiGraph = nx.DiGraph()
    for node_id, cc_id, label in zip(node_ids, cc_ids, labels, strict=True):
        graph.add_node(str(node_id), cc_id=int(cc_id), label=int(label))
    for _, row in edges_df.iterrows():
        src, dst = str(row[src_col]), str(row[dst_col])
        # Published edge lists may reference background nodes absent from nodes.csv;
        # add them as unknown so the graph stays internally consistent.
        for endpoint in (src, dst):
            if endpoint not in graph:
                graph.add_node(endpoint, cc_id=LABEL_UNKNOWN, label=LABEL_UNKNOWN)
        graph.add_edge(src, dst)

    return Elliptic2Dataset(
        graph=graph,
        node_ids=node_ids,
        labels=labels,
        cc_ids=cc_ids,
        component_labels=component_labels,
        features=features,
        feature_names=[str(c) for c in feature_cols],
        source_dir=directory,
    )


def elliptic2_fixture_dir() -> Path:
    """Path to the bundled synthetic fixture mirroring the Elliptic2 CSV layout.

    The fixture is tiny (14 nodes, 2 labelled components, a handful of background
    nodes) and entirely synthetic — it contains no Elliptic2 data and exists purely
    so loaders, demos and tests can run offline without the real download.
    """
    return Path(__file__).resolve().parents[2] / "datasets" / "fixtures" / "elliptic2_sample"
