"""AML subgraph scorer: explainable logistic regression over egonet/motif features.

A deliberately classical model in pure numpy — no torch, no sklearn — because an AML
score that feeds a human investigation queue must be reproducible from first
principles. The pipeline is:

1. :mod:`fintwinos.models.graph.features` turns each account node into a fixed-width
   vector of motif counts (fan-in/out, 2-cycles, triangles, flow concentration,
   layering chain depth, ...).
2. ``AmlSubgraphScorer`` standardises the features and fits an L2-regularised
   logistic regression by full-batch gradient descent (the problem is convex, so
   gradient descent with a deterministic seed converges to the same weights on the
   same data, every time).
3. ``save``/``load`` round-trip the whole fitted state through a plain JSON file so a
   model can be versioned in git and audited line by line.

Evaluation utilities (``roc_auc``, ``precision_recall_curve``) live here too so the
evals module and tests share one numerically-vetted implementation.
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from fintwinos.models.graph.features import FEATURE_NAMES, feature_matrix

SCORER_FORMAT_VERSION = 1

#: Node-attribute names probed (in order) when extracting labels from a simulated graph.
LABEL_KEY_CANDIDATES: tuple[str, ...] = (
    "is_suspicious",
    "suspicious",
    "is_ring",
    "ring",
    "is_mule",
    "label",
    "y",
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def roc_auc(y_true: Sequence[float] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Area under the ROC curve via the rank (Mann–Whitney) statistic.

    Handles tied scores exactly by assigning average ranks, so the result matches
    the trapezoidal AUC over the full ROC curve.

    Args:
        y_true: Binary labels (anything truthy/1 is positive).
        scores: Real-valued scores; higher means more positive.

    Returns:
        AUC in ``[0, 1]``; 0.5 is chance level.

    Raises:
        ValueError: If inputs differ in length or one class is absent.
    """
    y = (np.asarray(y_true, dtype=np.float64).ravel() > 0).astype(np.float64)
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same length")
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc requires at least one positive and one negative label")
    # Average ranks with exact tie handling.
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts).astype(np.float64)
    ranks = cumulative[inverse] - (counts[inverse] - 1) / 2.0
    rank_sum_pos = float(ranks[y == 1.0].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def precision_recall_curve(
    y_true: Sequence[float] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision/recall pairs at every distinct score threshold.

    Points are ordered by increasing recall and prefixed with the conventional
    anchor ``(precision=1.0, recall=0.0, threshold=+inf)``, so the arrays can be
    plotted or integrated directly.

    Args:
        y_true: Binary labels.
        scores: Real-valued scores; higher means more positive.

    Returns:
        ``(precision, recall, thresholds)`` — equal-length 1-D arrays. The point at
        index ``i`` describes classifying ``score >= thresholds[i]`` as positive.

    Raises:
        ValueError: If inputs differ in length or there are no positive labels.
    """
    y = (np.asarray(y_true, dtype=np.float64).ravel() > 0).astype(np.float64)
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same length")
    n_pos = float(y.sum())
    if n_pos == 0:
        raise ValueError("precision_recall_curve requires at least one positive label")

    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    # Evaluate only at the last index of each run of equal scores (true thresholds).
    boundaries = np.r_[np.nonzero(np.diff(ss))[0], ss.size - 1]
    precision = np.r_[1.0, tp[boundaries] / (tp[boundaries] + fp[boundaries])]
    recall = np.r_[0.0, tp[boundaries] / n_pos]
    thresholds = np.r_[np.inf, ss[boundaries]]
    return precision, recall, thresholds


class AmlSubgraphScorer:
    """L2-regularised numpy logistic regression over AML graph features.

    The scorer is fitted on per-node feature vectors (see
    :func:`fintwinos.models.graph.features.feature_matrix`) with binary labels —
    typically "belongs to a simulated laundering ring" vs "ordinary account".
    Features are standardised internally (mean/scale are stored with the model), so
    callers always pass raw feature values.

    Attributes:
        feature_names: Column names the model expects, in order.
        weights_: Fitted coefficients on the standardised features (after ``fit``).
        bias_: Fitted intercept (after ``fit``).
        mean_ / scale_: Standardisation parameters captured at fit time.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.5,
        epochs: int = 1500,
        l2: float = 1e-3,
        seed: int = 7,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ) -> None:
        """Configure the trainer.

        Args:
            learning_rate: Gradient-descent step size (the loss is convex; 0.1–1.0
                is stable on standardised features).
            epochs: Number of full-batch gradient steps.
            l2: L2 penalty strength on the weights (the bias is not penalised).
            seed: Seed for the ``numpy.random.default_rng`` used to initialise the
                weights; with fixed data this makes training fully reproducible.
            feature_names: Expected feature columns, in order.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.feature_names: list[str] = list(feature_names)
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    # -- training -----------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Sequence[float] | np.ndarray) -> dict[str, float]:
        """Fit the logistic regression by full-batch gradient descent.

        Args:
            X: Feature matrix of shape ``(n, len(feature_names))`` with raw
                (unstandardised) values.
            y: Binary labels of length ``n``.

        Returns:
            Training summary: ``final_loss``, in-sample ``auc``, ``n``, ``n_pos``.

        Raises:
            ValueError: On shape mismatches or single-class labels.
        """
        X = np.asarray(X, dtype=np.float64)
        yv = (np.asarray(y, dtype=np.float64).ravel() > 0).astype(np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if X.shape[0] != yv.size:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but the scorer expects "
                f"{len(self.feature_names)} ({self.feature_names})"
            )
        if yv.sum() == 0 or yv.sum() == yv.size:
            raise ValueError("fit requires both positive and negative examples")

        n = X.shape[0]
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0  # constant columns carry no signal; avoid 0-division
        self.scale_ = scale
        Z = (X - self.mean_) / self.scale_

        rng = np.random.default_rng(self.seed)
        w = rng.normal(0.0, 0.01, size=Z.shape[1])
        b = 0.0
        self.loss_history_ = []
        eps = 1e-12
        for _ in range(self.epochs):
            p = _sigmoid(Z @ w + b)
            grad_w = Z.T @ (p - yv) / n + self.l2 * w
            grad_b = float(np.mean(p - yv))
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
            loss = float(
                -np.mean(yv * np.log(p + eps) + (1.0 - yv) * np.log(1.0 - p + eps))
                + 0.5 * self.l2 * float(w @ w)
            )
            self.loss_history_.append(loss)

        self.weights_ = w
        self.bias_ = float(b)
        return {
            "final_loss": self.loss_history_[-1],
            "auc": roc_auc(yv, self.predict_proba(X)),
            "n": float(n),
            "n_pos": float(yv.sum()),
        }

    def train_on_simulator(
        self,
        sim_result_graph: Any,
        *,
        label_key: str | None = None,
        amount_key: str = "amount",
    ) -> dict[str, float]:
        """Train directly from a (simulated) labelled transaction graph.

        Designed for the output of the AML ring simulator: a networkx graph whose
        nodes carry a truthy attribute marking planted launderers. Also accepts any
        object exposing a ``.graph`` attribute holding such a graph (e.g. a
        simulation-result wrapper).

        Args:
            sim_result_graph: A ``networkx.Graph``/``DiGraph`` or an object with a
                ``.graph`` attribute containing one.
            label_key: Node attribute holding the binary label. When ``None``, the
                first attribute found among :data:`LABEL_KEY_CANDIDATES` is used;
                nodes missing the attribute are treated as legitimate (label 0).
            amount_key: Edge attribute holding transfer amounts.

        Returns:
            The training summary from :meth:`fit`.

        Raises:
            ValueError: If no graph or no positive labels can be found.
        """
        graph = sim_result_graph
        if not isinstance(graph, nx.Graph):
            graph = getattr(sim_result_graph, "graph", None)
        if not isinstance(graph, nx.Graph):
            raise ValueError(
                "train_on_simulator expects a networkx graph or an object with a "
                "'.graph' attribute containing one"
            )

        keys = (label_key,) if label_key is not None else LABEL_KEY_CANDIDATES
        chosen_key: str | None = None
        for key in keys:
            if key is not None and any(key in d for _, d in graph.nodes(data=True)):
                chosen_key = key
                break
        if chosen_key is None:
            raise ValueError(
                f"no label attribute found on graph nodes (looked for {list(keys)})"
            )

        X, nodes = feature_matrix(graph, amount_key=amount_key)
        y = np.array(
            [1.0 if graph.nodes[node].get(chosen_key) else 0.0 for node in nodes],
            dtype=np.float64,
        )
        return self.fit(X, y)

    # -- inference ----------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.weights_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("AmlSubgraphScorer is not fitted; call fit() first")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive (suspicious) class for each row of ``X``."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but the scorer expects {len(self.feature_names)}"
            )
        Z = (X - self.mean_) / self.scale_
        return _sigmoid(Z @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at the given probability threshold."""
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def score_graph(
        self,
        graph: nx.Graph,
        nodes: Sequence[Hashable] | None = None,
        *,
        amount_key: str = "amount",
    ) -> dict[Hashable, float]:
        """Score every node of a transaction graph.

        Returns:
            Mapping node -> probability of being part of a laundering structure.
        """
        X, node_list = feature_matrix(graph, nodes, amount_key=amount_key)
        probs = self.predict_proba(X)
        return {node: float(p) for node, p in zip(node_list, probs, strict=True)}

    def score_subgraph(self, graph: nx.Graph, *, amount_key: str = "amount") -> float:
        """Aggregate suspicion score for a whole subgraph.

        A subgraph is as suspicious as its most suspicious member, so this returns
        the maximum node probability — the natural alert-level statistic for a
        candidate ring extracted by community detection or case expansion.
        """
        scores = self.score_graph(graph, amount_key=amount_key)
        if not scores:
            return 0.0
        return max(scores.values())

    # -- persistence ----------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Serialise the fitted model to a human-auditable JSON file."""
        self._check_fitted()
        assert self.weights_ is not None and self.mean_ is not None and self.scale_ is not None
        payload = {
            "format_version": SCORER_FORMAT_VERSION,
            "model": "AmlSubgraphScorer",
            "feature_names": self.feature_names,
            "weights": self.weights_.tolist(),
            "bias": self.bias_,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "l2": self.l2,
                "seed": self.seed,
            },
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path | str) -> AmlSubgraphScorer:
        """Load a model previously written by :meth:`save`.

        Raises:
            ValueError: If the file is not a serialised ``AmlSubgraphScorer`` or
                uses an unsupported format version.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model") != "AmlSubgraphScorer":
            raise ValueError(f"{path} does not contain an AmlSubgraphScorer")
        if payload.get("format_version") != SCORER_FORMAT_VERSION:
            raise ValueError(
                f"unsupported AmlSubgraphScorer format version {payload.get('format_version')!r}"
            )
        hp = payload.get("hyperparameters", {})
        scorer = cls(
            learning_rate=hp.get("learning_rate", 0.5),
            epochs=hp.get("epochs", 1500),
            l2=hp.get("l2", 1e-3),
            seed=hp.get("seed", 7),
            feature_names=payload["feature_names"],
        )
        scorer.weights_ = np.asarray(payload["weights"], dtype=np.float64)
        scorer.bias_ = float(payload["bias"])
        scorer.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        scorer.scale_ = np.asarray(payload["scale"], dtype=np.float64)
        return scorer
