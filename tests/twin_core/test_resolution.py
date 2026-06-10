"""EntityResolver: id matches, normalised/fuzzy name matches, canonicalisation."""

from __future__ import annotations

import pytest

from fintwinos.core.types import EntityRef
from fintwinos.twin_core.resolution import (
    EntityResolver,
    levenshtein,
    levenshtein_ratio,
    name_similarity,
    normalise_name,
    token_set_jaccard,
)


def _ref(entity_type: str, entity_id: str) -> EntityRef:
    return EntityRef(entity_type=entity_type, entity_id=entity_id)


# -- primitives ---------------------------------------------------------------------


def test_levenshtein_basics():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("flaw", "lawn") == 2


def test_levenshtein_ratio_bounds():
    assert levenshtein_ratio("", "") == 1.0
    assert levenshtein_ratio("abc", "abc") == 1.0
    assert levenshtein_ratio("abc", "xyz") == 0.0


def test_token_set_jaccard():
    assert token_set_jaccard("a b c", "a b c") == 1.0
    assert token_set_jaccard("a b", "b c") == pytest.approx(1 / 3)
    assert token_set_jaccard("", "a") == 0.0


def test_normalise_name_expands_abbreviations():
    assert normalise_name("ACME Corp Ltd") == "acme corporation limited"
    assert normalise_name("Smith & Sons Co.") == "smith and sons company"
    assert normalise_name("Acme Corporation Limited") == "acme corporation limited"


# -- resolver ------------------------------------------------------------------------


@pytest.fixture()
def resolver() -> EntityResolver:
    r = EntityResolver()
    r.register(_ref("customer", "cus_1"), name="Acme Corporation Limited")
    r.register(_ref("customer", "cus_2"), name="Zenith Marine Holdings PLC")
    r.register(_ref("account", "acc_1"))
    return r


def test_exact_id_match_first(resolver):
    result = resolver.resolve("customer", "cus_1")
    assert result.method == "exact_id"
    assert result.confidence == 1.0
    assert result.ref.key() == "customer:cus_1"


def test_acme_corp_ltd_matches_acme_corporation_limited(resolver):
    """The contracted fuzzy-match example: spelling/abbreviation variants merge."""
    result = resolver.resolve("customer", "ACME Corp Ltd")
    assert result.ref is not None
    assert result.ref.key() == "customer:cus_1"
    assert result.confidence >= 0.9
    assert result.method in {"exact_name", "fuzzy"}


def test_fuzzy_partial_name_match(resolver):
    result = resolver.resolve_name("customer", "Acme Corp")
    assert result.method == "fuzzy"
    assert result.ref.key() == "customer:cus_1"
    assert 0.6 <= result.confidence < 1.0


def test_distinct_names_do_not_match(resolver):
    result = resolver.resolve_name("customer", "Borealis Textiles GmbH")
    assert result.method == "none"
    assert result.ref is None
    assert result.confidence == 0.0


def test_shared_suffix_alone_is_not_a_match(resolver):
    """Two different companies sharing 'Holdings PLC' must not merge."""
    result = resolver.resolve_name("customer", "Kestrel Marine Holdings PLC")
    if result.ref is not None:  # if it matched, it must not be high confidence
        assert result.confidence < 0.9
    # and registering it as a new entity must keep it distinct
    outcome = EntityResolver().canonicalise(
        _ref("customer", "cus_99"), name="Kestrel Marine Holdings PLC"
    )
    assert outcome.method == "new"


def test_resolution_is_scoped_by_entity_type(resolver):
    result = resolver.resolve_name("account", "Acme Corporation Limited")
    assert result.method == "none"


def test_threshold_is_configurable():
    strict = EntityResolver(threshold=0.99)
    strict.register(_ref("customer", "cus_1"), name="Acme Corporation Limited")
    assert strict.resolve_name("customer", "Acme Corp").method == "none"
    with pytest.raises(ValueError):
        EntityResolver(threshold=0.0)


def test_canonicalise_merges_duplicates_and_records_alias(resolver):
    duplicate = _ref("customer", "cus_77")
    result = resolver.canonicalise(duplicate, name="ACME Corp Ltd")
    assert result.ref.key() == "customer:cus_1"
    assert result.confidence >= 0.9
    # the duplicate id now resolves deterministically through the alias table
    again = resolver.resolve("customer", "cus_77")
    assert again.method == "alias"
    assert again.ref.key() == "customer:cus_1"
    assert again.confidence == 1.0


def test_canonicalise_registers_new_entities(resolver):
    fresh = _ref("customer", "cus_50")
    result = resolver.canonicalise(fresh, name="Umberline Foods Group")
    assert result.method == "new"
    assert result.confidence == 1.0
    assert resolver.known(fresh)
    assert resolver.display_name(fresh) == "Umberline Foods Group"


def test_name_similarity_symmetric():
    a = normalise_name("Acme Corp Ltd")
    b = normalise_name("Acme Corporation Limited")
    assert name_similarity(a, b) == pytest.approx(name_similarity(b, a))
    assert name_similarity(a, b) == pytest.approx(1.0)


def test_demo_book_resolver_finds_acme(demo_runtime):
    """The demo book seeds 'Acme Corp Ltd'; variants resolve onto it."""
    result = demo_runtime.ingestor.resolver.resolve("customer", "Acme Corporation Limited")
    assert result.ref is not None
    assert result.ref.entity_id == "cus_0001"
    assert result.confidence >= 0.9
