"""Tests for the dataset registry: completeness, cards on disk, loader wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from fintwinos.datasets.registry import (
    CARDS_DIR,
    DATASETS,
    DatasetEntry,
    get_dataset,
    list_datasets,
    load_dataset,
)

EXPECTED_NAMES = {
    "synthetic-transactions",
    "fraud-rings",
    "journeys",
    "market-paths",
    "treasury-ladders",
    "edgar-filings",
    "elliptic2",
}

REQUIRED_CARD_SECTIONS = [
    "## Description",
    "## Schema",
    "## Generation / provenance",
    "## Licence and pass-through terms",
    "## Intended use",
    "## Limitations",
]


class TestRegistryContents:
    def test_all_expected_datasets_registered(self):
        assert set(DATASETS) == EXPECTED_NAMES

    def test_entries_well_formed(self):
        for name, entry in DATASETS.items():
            assert isinstance(entry, DatasetEntry)
            assert entry.name == name
            assert callable(entry.loader)
            assert entry.licence.strip()
            assert entry.description.strip()
            assert entry.kind in {"synthetic", "bundled", "external"}

    def test_every_card_file_exists(self):
        for entry in DATASETS.values():
            assert entry.data_card.is_file(), f"missing data card: {entry.data_card}"

    def test_cards_have_required_sections(self):
        for entry in DATASETS.values():
            text = entry.data_card.read_text(encoding="utf-8")
            for section in REQUIRED_CARD_SECTIONS:
                assert section in text, (
                    f"{entry.data_card.name} missing section {section!r}"
                )

    def test_external_datasets_flag_pass_through_terms(self):
        elliptic = get_dataset("elliptic2")
        assert "Not redistributed" in elliptic.licence
        edgar = get_dataset("edgar-filings")
        assert "public-domain" in edgar.licence


class TestListDatasets:
    def test_list_shape_and_sorting(self):
        listed = list_datasets()
        assert [d["name"] for d in listed] == sorted(EXPECTED_NAMES)
        for d in listed:
            assert set(d) == {
                "name", "description", "kind", "licence", "data_card",
                "card_exists", "offline_available", "loader",
            }
            assert d["card_exists"] is True
            assert d["loader"].startswith("fintwinos.datasets.")

    def test_loaders_resolve_to_real_callables(self):
        for d in list_datasets():
            entry = get_dataset(d["name"])
            module, _, qualname = d["loader"].rpartition(".")
            assert entry.loader.__module__ == module
            assert entry.loader.__qualname__ == qualname


class TestGetDataset:
    def test_known(self):
        entry = get_dataset("market-paths")
        assert entry.kind == "synthetic"

    def test_unknown_lists_options(self):
        with pytest.raises(KeyError, match="known datasets"):
            get_dataset("does-not-exist")


class TestLoadDataset:
    def test_load_synthetic_with_defaults(self):
        data = load_dataset("synthetic-transactions")
        assert data is not None

    def test_load_external_uses_bundled_sample_offline(self):
        # elliptic2 needs a data_dir; offline_kwargs points it at the bundled fixture.
        data = load_dataset("elliptic2")
        assert data is not None

    def test_load_bundled_filing_corpus(self):
        corpus = load_dataset("edgar-filings")
        assert isinstance(corpus, list) and len(corpus) > 0

    def test_load_unknown_raises(self):
        with pytest.raises(KeyError, match="known datasets"):
            load_dataset("does-not-exist")

    def test_every_dataset_advertises_offline_availability(self):
        for entry in list_datasets():
            assert entry["offline_available"] is True


class TestSyntheticLoadersRunFromRegistry:
    """Smoke-run each synthetic loader through the registry surface."""

    @pytest.mark.parametrize(
        "name,kwargs",
        [
            ("synthetic-transactions", {"n": 30, "ring_fraction": 0.1, "seed": 3}),
            ("fraud-rings", {"size": 6, "layers": 3, "seed": 3}),
            ("journeys", {"n": 5, "seed": 3}),
            ("market-paths", {"n_instruments": 2, "n_steps": 10, "seed": 3}),
            ("treasury-ladders", {"currencies": ("USD",), "days": 5, "seed": 3}),
        ],
    )
    def test_loader_runs(self, name, kwargs):
        ds = get_dataset(name).loader(**kwargs)
        assert ds.envelopes
        assert ds.labels

    def test_edgar_default_loader_is_offline_corpus(self):
        docs = get_dataset("edgar-filings").loader()
        assert 6 <= len(docs) <= 8


class TestRepoDocumentation:
    def test_readme_indexes_every_card(self):
        readme = CARDS_DIR.parent / "README.md"
        assert readme.is_file()
        text = readme.read_text(encoding="utf-8")
        for entry in DATASETS.values():
            assert f"cards/{entry.data_card.name}" in text, (
                f"datasets/README.md does not index {entry.data_card.name}"
            )

    def test_readme_states_privileged_data_rule(self):
        text = (CARDS_DIR.parent / "README.md").read_text(encoding="utf-8")
        assert "never" in text.lower() and "committed" in text.lower()
        assert "deployment-local" in text.lower()

    def test_cards_dir_has_no_orphan_cards(self):
        registered = {entry.data_card.name for entry in DATASETS.values()}
        on_disk = {p.name for p in Path(CARDS_DIR).glob("*.md")}
        assert on_disk == registered
