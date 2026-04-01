"""Unit tests for src/kg/entity_resolver.py"""

from __future__ import annotations

import pytest

from src.kg.entity_resolver import EntityResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(name: str, etype: str, confidence: float, doc_id: str) -> dict:
    return {"name": name, "type": etype, "confidence": confidence, "doc_id": doc_id}


def _group_by_canonical(groups: list[dict]) -> dict[str, dict]:
    return {g["canonical_name"]: g for g in groups}


# ---------------------------------------------------------------------------
# Basic instantiation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_threshold(self):
        resolver = EntityResolver()
        assert resolver.fuzzy_threshold == 75

    def test_custom_threshold(self):
        resolver = EntityResolver(fuzzy_threshold=90)
        assert resolver.fuzzy_threshold == 90

    def test_resolve_empty_list(self):
        resolver = EntityResolver()
        assert resolver.resolve([]) == []


# ---------------------------------------------------------------------------
# Single-entity pass-through
# ---------------------------------------------------------------------------

class TestSingleEntity:
    def test_single_entity_creates_one_group(self):
        resolver = EntityResolver()
        result = resolver.resolve([_make("John Smith", "PERSON", 0.9, "doc1")])
        assert len(result) == 1

    def test_single_entity_fields(self):
        resolver = EntityResolver()
        groups = resolver.resolve([_make("John Smith", "PERSON", 0.9, "doc1")])
        g = groups[0]
        assert g["canonical_name"] == "John Smith"
        assert g["type"] == "PERSON"
        assert g["aliases"] == []
        assert g["doc_ids"] == ["doc1"]
        assert g["confidence"] == pytest.approx(0.9)
        assert g["mention_count"] == 1


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

class TestAliasResolution:
    def test_exact_match_merges(self):
        resolver = EntityResolver()
        entities = [
            _make("Acme Corp", "ORGANIZATION", 0.8, "doc1"),
            _make("Acme Corp", "ORGANIZATION", 0.7, "doc2"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1
        assert result[0]["mention_count"] == 2

    def test_fuzzy_match_creates_alias(self):
        """'Acme Corporation' should fuzzy-match 'Acme Corp' and become an alias."""
        resolver = EntityResolver(fuzzy_threshold=75)
        entities = [
            _make("Acme Corp", "ORGANIZATION", 0.8, "doc1"),
            _make("Acme Corporation", "ORGANIZATION", 0.6, "doc2"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1
        g = result[0]
        # Lower-confidence name is the alias; higher-confidence is canonical.
        assert g["canonical_name"] == "Acme Corp"
        assert "Acme Corporation" in g["aliases"]

    def test_fuzzy_match_promotes_to_canonical(self):
        """Higher-confidence fuzzy match should become the canonical name."""
        resolver = EntityResolver(fuzzy_threshold=75)
        entities = [
            _make("Acme Corp", "ORGANIZATION", 0.6, "doc1"),
            _make("Acme Corporation", "ORGANIZATION", 0.9, "doc2"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1
        g = result[0]
        assert g["canonical_name"] == "Acme Corporation"
        assert "Acme Corp" in g["aliases"]

    def test_aliases_are_sorted(self):
        resolver = EntityResolver(fuzzy_threshold=75)
        entities = [
            _make("J. Smith", "PERSON", 0.5, "doc1"),
            _make("John Smith", "PERSON", 0.7, "doc2"),
            _make("Jonathan Smith", "PERSON", 0.6, "doc3"),
        ]
        result = resolver.resolve(entities)
        # All three share partial overlap; exactly which merge depends on order,
        # but aliases list must be sorted.
        for g in result:
            assert g["aliases"] == sorted(g["aliases"])

    def test_no_duplicate_aliases(self):
        """The same name should not appear in aliases more than once."""
        resolver = EntityResolver()
        entities = [
            _make("ACME", "ORGANIZATION", 0.5, "doc1"),
            _make("ACME", "ORGANIZATION", 0.5, "doc2"),
            _make("ACME", "ORGANIZATION", 0.5, "doc3"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1
        assert result[0]["aliases"] == []

    def test_below_threshold_no_merge(self):
        """Names that score below the threshold remain separate groups.

        'John Smith' vs 'Jane Doe' scores ~25 with partial_ratio, well below
        any reasonable threshold, so they must never be merged.
        """
        resolver = EntityResolver(fuzzy_threshold=75)
        entities = [
            _make("John Smith", "PERSON", 0.8, "doc1"),
            _make("Jane Doe", "PERSON", 0.8, "doc2"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Different types never merge
# ---------------------------------------------------------------------------

class TestTypeSeparation:
    def test_same_name_different_types_not_merged(self):
        """'Smith' as PERSON and 'Smith' as ORGANIZATION must remain separate groups."""
        resolver = EntityResolver()
        entities = [
            _make("Smith", "PERSON", 0.9, "doc1"),
            _make("Smith", "ORGANIZATION", 0.9, "doc1"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 2
        types = {g["type"] for g in result}
        assert types == {"PERSON", "ORGANIZATION"}

    def test_fuzzy_match_different_types_not_merged(self):
        resolver = EntityResolver(fuzzy_threshold=75)
        entities = [
            _make("Apple Inc", "ORGANIZATION", 0.9, "doc1"),
            _make("Apple Inc", "PRODUCT", 0.9, "doc2"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 2

    def test_multiple_types_stay_independent(self):
        resolver = EntityResolver()
        entities = [
            _make("California", "LOCATION", 0.8, "doc1"),
            _make("California", "LOCATION", 0.7, "doc2"),
            _make("California", "ORGANIZATION", 0.9, "doc3"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 2
        by_type = {g["type"]: g for g in result}
        assert by_type["LOCATION"]["mention_count"] == 2
        assert by_type["ORGANIZATION"]["mention_count"] == 1


# ---------------------------------------------------------------------------
# Cross-document linking
# ---------------------------------------------------------------------------

class TestCrossDocumentLinking:
    def test_same_entity_multiple_docs_merged(self):
        resolver = EntityResolver()
        entities = [
            _make("John Smith", "PERSON", 0.9, "doc1"),
            _make("John Smith", "PERSON", 0.85, "doc2"),
            _make("John Smith", "PERSON", 0.80, "doc3"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1
        g = result[0]
        assert g["doc_ids"] == ["doc1", "doc2", "doc3"]
        assert g["mention_count"] == 3

    def test_doc_ids_are_sorted(self):
        resolver = EntityResolver()
        entities = [
            _make("Acme", "ORGANIZATION", 0.8, "doc_z"),
            _make("Acme", "ORGANIZATION", 0.8, "doc_a"),
            _make("Acme", "ORGANIZATION", 0.8, "doc_m"),
        ]
        result = resolver.resolve(entities)
        assert result[0]["doc_ids"] == ["doc_a", "doc_m", "doc_z"]

    def test_no_duplicate_doc_ids(self):
        resolver = EntityResolver()
        entities = [
            _make("Acme", "ORGANIZATION", 0.8, "doc1"),
            _make("Acme", "ORGANIZATION", 0.8, "doc1"),
        ]
        result = resolver.resolve(entities)
        assert result[0]["doc_ids"] == ["doc1"]

    def test_fuzzy_match_cross_document(self):
        """Fuzzy variants from different documents should share the same group."""
        resolver = EntityResolver(fuzzy_threshold=75)
        entities = [
            _make("John Smith", "PERSON", 0.9, "doc1"),
            _make("J. Smith", "PERSON", 0.7, "doc2"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1
        g = result[0]
        assert sorted(g["doc_ids"]) == ["doc1", "doc2"]


# ---------------------------------------------------------------------------
# Confidence propagation
# ---------------------------------------------------------------------------

class TestConfidencePropagation:
    def test_single_mention_confidence(self):
        resolver = EntityResolver()
        result = resolver.resolve([_make("X", "TYPE", 0.6, "d1")])
        assert result[0]["confidence"] == pytest.approx(0.6)

    def test_weighted_average_two_mentions(self):
        resolver = EntityResolver()
        entities = [
            _make("X", "TYPE", 0.8, "d1"),
            _make("X", "TYPE", 0.4, "d2"),
        ]
        result = resolver.resolve(entities)
        # (0.8 + 0.4) / 2 == 0.6
        assert result[0]["confidence"] == pytest.approx(0.6)

    def test_weighted_average_three_mentions(self):
        resolver = EntityResolver()
        entities = [
            _make("X", "TYPE", 0.9, "d1"),
            _make("X", "TYPE", 0.6, "d2"),
            _make("X", "TYPE", 0.3, "d3"),
        ]
        result = resolver.resolve(entities)
        # (0.9 + 0.6 + 0.3) / 3 == 0.6
        assert result[0]["confidence"] == pytest.approx(0.6)

    def test_canonical_promotion_updates_confidence(self):
        """When a higher-confidence mention promotes to canonical the weighted
        average must still reflect all mentions."""
        resolver = EntityResolver(fuzzy_threshold=75)
        entities = [
            _make("Acme Corp", "ORGANIZATION", 0.5, "d1"),
            _make("Acme Corporation", "ORGANIZATION", 0.9, "d2"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1
        assert result[0]["confidence"] == pytest.approx((0.5 + 0.9) / 2)

    def test_confidence_independent_per_type(self):
        """Groups for different types maintain independent confidence averages."""
        resolver = EntityResolver()
        entities = [
            _make("Delta", "ORGANIZATION", 1.0, "d1"),
            _make("Delta", "LOCATION", 0.5, "d1"),
        ]
        result = resolver.resolve(entities)
        by_type = {g["type"]: g for g in result}
        assert by_type["ORGANIZATION"]["confidence"] == pytest.approx(1.0)
        assert by_type["LOCATION"]["confidence"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Output structure contract
# ---------------------------------------------------------------------------

class TestOutputStructure:
    REQUIRED_KEYS = {"canonical_name", "type", "aliases", "doc_ids",
                     "confidence", "mention_count"}

    def test_output_has_required_keys(self):
        resolver = EntityResolver()
        result = resolver.resolve([_make("Test", "TYPE", 0.5, "d1")])
        assert set(result[0].keys()) == self.REQUIRED_KEYS

    def test_no_internal_keys_exposed(self):
        """Internal accumulator key '_total_confidence' must not leak out."""
        resolver = EntityResolver()
        result = resolver.resolve([_make("Test", "TYPE", 0.5, "d1")])
        assert "_total_confidence" not in result[0]

    def test_multiple_distinct_entities(self):
        resolver = EntityResolver()
        entities = [
            _make("Alice", "PERSON", 0.9, "d1"),
            _make("Bob", "PERSON", 0.8, "d2"),
            _make("Acme", "ORGANIZATION", 0.7, "d1"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 3
        for g in result:
            assert set(g.keys()) == self.REQUIRED_KEYS
