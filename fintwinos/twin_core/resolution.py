"""Deterministic entity resolution for the twin.

Connectors and upstream systems rarely agree on identifiers: the same
counterparty arrives as ``"ACME Corp Ltd"`` from one feed and
``"Acme Corporation Limited"`` from another. The :class:`EntityResolver` maps
incoming references onto canonical twin entities with a transparent,
fully deterministic cascade:

1. **Exact id match** — the reference key is already canonical.
2. **Alias match** — the key was previously merged into a canonical entity.
3. **Exact normalised-name match** — names are lowercased, tokenised to
   alphanumerics and legal-form abbreviations are expanded
   (``ltd -> limited``, ``corp -> corporation``...), so common corporate
   spelling variants collapse to the same normal form.
4. **Fuzzy name match** — a weighted blend of token-set Jaccard similarity and
   Levenshtein ratio (both implemented in pure Python) over normalised names,
   accepted only at or above a configurable threshold.

Every resolution returns the canonical :class:`EntityRef` *plus* a confidence
score and the method used, so downstream consumers (and the audit trail) can
always explain why two records were linked.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from fintwinos.core.types import EntityRef

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Legal-form and trade abbreviations expanded during name normalisation.
ABBREVIATIONS: dict[str, str] = {
    "ltd": "limited",
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "intl": "international",
    "mfg": "manufacturing",
    "svcs": "services",
    "grp": "group",
    "hldgs": "holdings",
    "bros": "brothers",
    "assoc": "associates",
    "mgmt": "management",
    "tech": "technologies",
}

_JACCARD_WEIGHT = 0.7
_LEVENSHTEIN_WEIGHT = 0.3


def normalise_name(name: str) -> str:
    """Canonical lowercase form of a name with legal abbreviations expanded."""
    text = name.lower().replace("&", " and ")
    tokens = [ABBREVIATIONS.get(token, token) for token in _TOKEN_RE.findall(text)]
    return " ".join(tokens)


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (pure-Python two-row dynamic program)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    """Similarity in [0, 1]: ``1 - distance / max(len)`` (1.0 for two empty strings)."""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def token_set_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of the token sets of two (normalised) strings."""
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def name_similarity(a: str, b: str) -> float:
    """Blended similarity of two normalised names in [0, 1].

    Weighted ``0.7 * token-set Jaccard + 0.3 * Levenshtein ratio``: token
    overlap dominates (word order and fragments matter less for entity names),
    while the character-level ratio separates near-identical spellings from
    coincidental token overlap.
    """
    return _JACCARD_WEIGHT * token_set_jaccard(a, b) + _LEVENSHTEIN_WEIGHT * levenshtein_ratio(a, b)


class ResolutionResult(BaseModel):
    """Outcome of one resolution attempt: canonical ref, confidence and method."""

    ref: EntityRef | None = None
    confidence: float = 0.0
    method: str = "none"  # exact_id | alias | exact_name | fuzzy | new | none
    matched_name: str | None = None
    candidates_considered: int = Field(default=0, ge=0)


class EntityResolver:
    """Registry of canonical entities with deterministic id/name resolution.

    Parameters
    ----------
    threshold:
        Minimum :func:`name_similarity` score (over normalised names) for a
        fuzzy match to be accepted. The default of ``0.65`` accepts true
        spelling/abbreviation variants while rejecting distinct companies that
        merely share legal-form suffixes.
    """

    def __init__(self, threshold: float = 0.65) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self._by_key: dict[str, EntityRef] = {}
        self._names: dict[str, dict[str, str]] = {}  # entity_type -> normalised name -> key
        self._aliases: dict[str, str] = {}  # duplicate key -> canonical key
        self._display: dict[str, str] = {}  # canonical key -> primary display name

    # -- registration -----------------------------------------------------------------

    def register(
        self,
        ref: EntityRef,
        name: str | None = None,
        aliases: Iterable[str] = (),
    ) -> None:
        """Register a canonical entity, optionally with a display name and aliases.

        The first name registered for an entity becomes its display name;
        additional names (or alias strings) become extra lookup keys for the
        same canonical entity. Within an entity type the first registrant of a
        normalised name keeps it — later entities never silently steal a name.
        """
        key = ref.key()
        self._by_key[key] = ref
        index = self._names.setdefault(ref.entity_type, {})
        for candidate in ([name] if name else []) + list(aliases):
            self._display.setdefault(key, candidate)
            index.setdefault(normalise_name(candidate), key)

    def add_alias(self, duplicate: EntityRef, canonical: EntityRef) -> None:
        """Record that ``duplicate``'s key should resolve to ``canonical``."""
        if canonical.key() not in self._by_key:
            raise KeyError(f"canonical entity {canonical.key()!r} is not registered")
        self._aliases[duplicate.key()] = canonical.key()

    def known(self, ref: EntityRef) -> bool:
        """Whether the reference is already canonical (not merely aliased)."""
        return ref.key() in self._by_key

    def display_name(self, ref: EntityRef) -> str | None:
        """The primary display name registered for a canonical entity, if any."""
        return self._display.get(ref.key())

    # -- resolution ---------------------------------------------------------------------

    def resolve(self, entity_type: str, value: str) -> ResolutionResult:
        """Resolve an id *or* a name within one entity type.

        Tries the deterministic id match (and alias table) first; if ``value``
        is not a known identifier it is treated as a name and resolved via
        :meth:`resolve_name`.
        """
        key = f"{entity_type}:{value}"
        if key in self._by_key:
            return ResolutionResult(
                ref=self._by_key[key],
                confidence=1.0,
                method="exact_id",
                matched_name=self._display.get(key),
            )
        if key in self._aliases:
            canonical = self._aliases[key]
            return ResolutionResult(
                ref=self._by_key[canonical],
                confidence=1.0,
                method="alias",
                matched_name=self._display.get(canonical),
            )
        return self.resolve_name(entity_type, value)

    def resolve_name(self, entity_type: str, name: str) -> ResolutionResult:
        """Resolve a display name to a canonical entity of ``entity_type``.

        Exact normalised-name hits return confidence 1.0; otherwise the best
        fuzzy candidate at or above the threshold wins, with the similarity
        score reported as the confidence. Ties break on the candidate's
        normalised name for determinism.
        """
        normalised = normalise_name(name)
        index = self._names.get(entity_type, {})
        if not normalised or not index:
            return ResolutionResult(method="none", candidates_considered=0)
        exact_key = index.get(normalised)
        if exact_key is not None:
            return ResolutionResult(
                ref=self._by_key[exact_key],
                confidence=1.0,
                method="exact_name",
                matched_name=self._display.get(exact_key),
                candidates_considered=len(index),
            )
        best_score, best_key = 0.0, None
        for candidate_norm, key in sorted(index.items()):
            score = name_similarity(normalised, candidate_norm)
            if score > best_score:
                best_score, best_key = score, key
        if best_key is not None and best_score >= self.threshold:
            return ResolutionResult(
                ref=self._by_key[best_key],
                confidence=round(best_score, 4),
                method="fuzzy",
                matched_name=self._display.get(best_key),
                candidates_considered=len(index),
            )
        return ResolutionResult(method="none", candidates_considered=len(index))

    def canonicalise(self, ref: EntityRef, name: str | None = None) -> ResolutionResult:
        """Map an incoming reference onto the canonical twin entity.

        Used by the ingestor on every envelope entity:

        - a known canonical key resolves to itself (registering any new name
          as an alias of it);
        - a key previously merged resolves through the alias table;
        - an unknown key with a name that matches an existing entity (exactly
          or fuzzily) is merged into that entity, and the mapping is recorded
          so future events with the duplicate id resolve deterministically;
        - otherwise the reference is registered as a new canonical entity.
        """
        key = ref.key()
        if key in self._by_key:
            if name:
                self.register(ref, name=name)
            return ResolutionResult(
                ref=self._by_key[key],
                confidence=1.0,
                method="exact_id",
                matched_name=self._display.get(key),
            )
        if key in self._aliases:
            canonical = self._aliases[key]
            return ResolutionResult(
                ref=self._by_key[canonical],
                confidence=1.0,
                method="alias",
                matched_name=self._display.get(canonical),
            )
        if name:
            result = self.resolve_name(ref.entity_type, name)
            if result.ref is not None:
                self._aliases[key] = result.ref.key()
                return result
        self.register(ref, name=name)
        return ResolutionResult(ref=ref, confidence=1.0, method="new", matched_name=name)
