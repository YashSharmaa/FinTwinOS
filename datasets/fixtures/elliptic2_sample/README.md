# Elliptic2 sample fixture (synthetic)

This directory mirrors the **CSV layout** of the Elliptic2 AML dataset
(`nodes.csv`, `edges.csv`, `connected_components.csv`) so that
`fintwinos.datasets.elliptic2_loader.load_elliptic2` can be exercised offline in
tests and demos.

It contains **no Elliptic2 data**. All 14 node ids, both component labels and every
edge are invented for FinTwinOS and are MIT-licensed like the rest of the
repository. The real dataset is released by Elliptic and the MIT-IBM Watson AI Lab
under its own research-use terms and must be downloaded from the official source:
https://github.com/MITIBMxGraph/Elliptic2, see `datasets/cards/elliptic2.md`.
