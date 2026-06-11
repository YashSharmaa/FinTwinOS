# Data card, elliptic2

## Description

Loader for **Elliptic2**, the large-scale anti-money-laundering graph dataset for
money-laundering *subgraph* detection on the Bitcoin blockchain, released by Elliptic
together with the MIT-IBM Watson AI Lab. Nodes are clusters of Bitcoin addresses,
edges are aggregated transactions between clusters, and labelled connected components
are marked `suspicious` or `licit`.

FinTwinOS does **not** bundle or redistribute Elliptic2. The loader reads a local
copy in the published CSV layout:
`fintwinos.datasets.elliptic2_loader.load_elliptic2(data_dir, nodes_file=...,
edges_file=..., labels_file=...)`. A tiny fully synthetic fixture in the same layout
ships at `datasets/fixtures/elliptic2_sample/` for offline tests and demos.

## Schema

Input CSVs (filenames configurable; published column names with tolerated aliases):

| File | Columns |
|---|---|
| `nodes.csv` | `clId` (node id), optional `ccId` (component membership, `-1`/missing = background), optional numeric feature columns |
| `edges.csv` | `clId1`, `clId2` (directed source → target) |
| `connected_components.csv` | `ccId`, `ccLabel` (`suspicious` / `licit`) |

Output `Elliptic2Dataset`:

- `graph`: `networkx.DiGraph`, node attributes `cc_id` and `label`;
- `node_ids`: node ids in `nodes.csv` row order;
- `labels`: aligned `int8` array, `1` suspicious, `0` licit, `-1` unknown;
- `cc_ids`, `component_labels`, optional `features` matrix + `feature_names`;
- `summary()` with node/edge/label counts.

## Generation / provenance

The real dataset derives from Bitcoin blockchain data with labels produced by
Elliptic's forensic analysts; see the release paper *"The Shape of Money Laundering:
Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset"*
(Bellei et al., 2024) and https://github.com/MITIBMxGraph/Elliptic2. The bundled
fixture is invented data written for FinTwinOS and carries no provenance from the
real dataset.

## Licence and pass-through terms

- Real dataset: released by Elliptic / MIT-IBM Watson AI Lab under its **own terms
  for research use**, review the licence at the official release before downloading,
  and do not redistribute the data through this repository or any FinTwinOS
  deployment artefact. Cite the release paper in published work.
- Bundled fixture: MIT (part of FinTwinOS), fully synthetic.

When files are absent the loader raises a `FileNotFoundError` repeating these
pointers.

## Intended use

Benchmarking AML graph detection against a real labelled dataset, validating graph
ingestion at realistic scale, and academic experimentation consistent with the
dataset's research-use terms.

## Limitations

- Blockchain cluster graphs differ structurally from retail banking transaction
  graphs; transfer of results to fiat AML settings is not automatic.
- Labels cover a minority of components; most of the graph is unlabelled background.
- The full dataset is large (tens of millions of edges); the loader materialises a
  `networkx` graph in memory, so subsample for constrained environments.
- The fixture is illustrative only, 14 nodes prove layout compatibility, nothing
  more.
