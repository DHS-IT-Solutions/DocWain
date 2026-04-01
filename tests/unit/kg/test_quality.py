"""Unit tests for src/kg/quality.py"""

from __future__ import annotations

import pytest

from src.kg.quality import (
    detect_gaps,
    score_entity_completeness,
    score_relationship_evidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(**kwargs) -> dict:
    """Build a minimal entity dict; type defaults to PERSON."""
    base = {"type": "PERSON"}
    base.update(kwargs)
    return base


# ===========================================================================
# score_entity_completeness
# ===========================================================================

class TestScoreEntityCompleteness:

    # --- name-only entity ---------------------------------------------------

    def test_name_only_scores_approx_0_2(self):
        entity = {"type": "PERSON", "name": "Alice"}
        score = score_entity_completeness(entity)
        assert abs(score - 0.20) < 1e-9

    def test_empty_entity_scores_zero(self):
        entity = {"type": "PERSON"}
        assert score_entity_completeness(entity) == 0.0

    # --- individual field contributions ------------------------------------

    def test_aliases_adds_0_15(self):
        entity = {"type": "PERSON", "name": "Alice", "aliases": ["Al"]}
        score = score_entity_completeness(entity)
        assert abs(score - 0.35) < 1e-9  # name(0.20) + aliases(0.15)

    def test_relationships_adds_0_25(self):
        entity = {"type": "PERSON", "name": "Alice", "relationships": ["employed_by"]}
        score = score_entity_completeness(entity)
        assert abs(score - 0.45) < 1e-9  # name + relationships

    def test_doc_ids_adds_0_15(self):
        entity = {"type": "PERSON", "name": "Alice", "doc_ids": ["doc-1"]}
        score = score_entity_completeness(entity)
        assert abs(score - 0.35) < 1e-9

    def test_temporal_bounds_adds_0_15(self):
        entity = {"type": "PERSON", "name": "Alice", "temporal_bounds": {"start": "2020"}}
        score = score_entity_completeness(entity)
        assert abs(score - 0.35) < 1e-9

    def test_confidence_adds_0_10(self):
        entity = {"type": "PERSON", "name": "Alice", "confidence": 0.9}
        score = score_entity_completeness(entity)
        assert abs(score - 0.30) < 1e-9

    # --- fully populated entity --------------------------------------------

    def test_fully_populated_entity_scores_1_0(self):
        entity = {
            "type": "PERSON",
            "name": "Alice",
            "aliases": ["Al"],
            "relationships": ["employed_by"],
            "doc_ids": ["doc-1"],
            "temporal_bounds": {"start": "2020"},
            "confidence": 0.95,
        }
        score = score_entity_completeness(entity)
        assert score == 1.0

    # --- empty / falsy values are not counted ------------------------------

    def test_empty_string_name_not_counted(self):
        entity = {"type": "PERSON", "name": ""}
        assert score_entity_completeness(entity) == 0.0

    def test_empty_list_aliases_not_counted(self):
        entity = {"type": "PERSON", "name": "Alice", "aliases": []}
        assert abs(score_entity_completeness(entity) - 0.20) < 1e-9

    def test_none_field_not_counted(self):
        entity = {"type": "PERSON", "name": "Alice", "confidence": None}
        assert abs(score_entity_completeness(entity) - 0.20) < 1e-9

    # --- cap at 1.0 ---------------------------------------------------------

    def test_score_never_exceeds_1(self):
        entity = {
            "type": "PERSON",
            "name": "Alice",
            "aliases": ["Al", "A"],
            "relationships": ["employed_by", "reports_to"],
            "doc_ids": ["doc-1", "doc-2"],
            "temporal_bounds": {"start": "2010", "end": "2025"},
            "confidence": 1.0,
            "extra_field": "ignored",
        }
        assert score_entity_completeness(entity) <= 1.0

    # --- type field carries no weight ---------------------------------------

    def test_type_field_carries_no_weight(self):
        entity = {"type": "ORGANIZATION"}
        assert score_entity_completeness(entity) == 0.0


# ===========================================================================
# score_relationship_evidence
# ===========================================================================

class TestScoreRelationshipEvidence:

    def test_zero_docs(self):
        assert score_relationship_evidence(0) == 0.0

    def test_negative_docs(self):
        assert score_relationship_evidence(-1) == 0.0

    def test_one_doc(self):
        assert abs(score_relationship_evidence(1) - 0.3) < 1e-9

    def test_two_docs(self):
        assert abs(score_relationship_evidence(2) - 0.6) < 1e-9

    def test_three_docs(self):
        assert abs(score_relationship_evidence(3) - 0.7) < 1e-9

    def test_four_docs(self):
        assert abs(score_relationship_evidence(4) - 0.8) < 1e-9

    def test_five_docs(self):
        assert abs(score_relationship_evidence(5) - 0.9) < 1e-9

    def test_six_docs(self):
        assert abs(score_relationship_evidence(6) - 0.92) < 1e-9

    def test_ten_docs(self):
        assert abs(score_relationship_evidence(10) - 1.0) < 1e-9

    def test_very_large_count_capped_at_1(self):
        assert score_relationship_evidence(1000) == 1.0

    def test_score_is_monotonically_non_decreasing(self):
        scores = [score_relationship_evidence(n) for n in range(0, 20)]
        for a, b in zip(scores, scores[1:]):
            assert b >= a, f"Score decreased: {a} -> {b}"


# ===========================================================================
# detect_gaps
# ===========================================================================

class TestDetectGaps:

    # --- empty input --------------------------------------------------------

    def test_empty_list_returns_empty(self):
        assert detect_gaps([]) == []

    # --- PERSON: no expected relationships ----------------------------------

    def test_person_with_no_relationships_has_gap(self):
        entity = {"type": "PERSON", "name": "Alice", "relationships": []}
        gaps = detect_gaps([entity])
        assert len(gaps) == 1
        assert gaps[0]["entity"] is entity
        assert gaps[0]["type"] == "PERSON"

    def test_person_missing_all_expected_rels_no_relationships_key(self):
        entity = {"type": "PERSON", "name": "Alice"}
        gaps = detect_gaps([entity])
        assert len(gaps) == 1

    def test_person_with_unrelated_relationships_has_gap(self):
        entity = {"type": "PERSON", "name": "Alice", "relationships": ["knows"]}
        gaps = detect_gaps([entity])
        assert len(gaps) == 1

    # --- PERSON: role_of without employed_by --------------------------------

    def test_person_role_of_without_employed_by_has_gap(self):
        entity = {"type": "PERSON", "name": "Alice", "relationships": ["role_of"]}
        gaps = detect_gaps([entity])
        assert len(gaps) == 1
        assert "employed_by" in gaps[0]["gap"]

    def test_person_role_of_with_employed_by_no_gap(self):
        entity = {
            "type": "PERSON",
            "name": "Alice",
            "relationships": ["role_of", "employed_by"],
        }
        gaps = detect_gaps([entity])
        assert gaps == []

    def test_person_employed_by_without_role_of_no_gap(self):
        entity = {"type": "PERSON", "name": "Alice", "relationships": ["employed_by"]}
        gaps = detect_gaps([entity])
        assert gaps == []

    # --- ORGANIZATION -------------------------------------------------------

    def test_organization_with_no_relationships_has_gap(self):
        entity = {"type": "ORGANIZATION", "name": "Acme", "relationships": []}
        gaps = detect_gaps([entity])
        assert len(gaps) == 1
        assert gaps[0]["type"] == "ORGANIZATION"

    def test_organization_with_employs_no_gap(self):
        entity = {
            "type": "ORGANIZATION",
            "name": "Acme",
            "relationships": ["employs"],
        }
        gaps = detect_gaps([entity])
        assert gaps == []

    # --- unknown entity type ------------------------------------------------

    def test_unknown_type_produces_no_gap(self):
        entity = {"type": "DOCUMENT", "name": "Contract"}
        gaps = detect_gaps([entity])
        assert gaps == []

    # --- mixed list ---------------------------------------------------------

    def test_mixed_entities_correct_gap_count(self):
        entities = [
            {"type": "PERSON", "name": "Alice", "relationships": []},          # gap: no rels
            {"type": "PERSON", "name": "Bob", "relationships": ["employed_by"]},  # ok
            {"type": "ORGANIZATION", "name": "Acme", "relationships": []},     # gap: no rels
            {"type": "DOCUMENT", "name": "SLA"},                               # unknown, no gap
        ]
        gaps = detect_gaps(entities)
        assert len(gaps) == 2

    # --- relationships as list of dicts (alternative representation) --------

    def test_relationships_as_list_of_dicts(self):
        entity = {
            "type": "PERSON",
            "name": "Alice",
            "relationships": [{"type": "employed_by", "target": "Acme"}],
        }
        gaps = detect_gaps([entity])
        assert gaps == []

    def test_relationships_as_list_of_dicts_role_of_without_employed_by(self):
        entity = {
            "type": "PERSON",
            "name": "Alice",
            "relationships": [{"type": "role_of", "target": "CTO"}],
        }
        gaps = detect_gaps([entity])
        assert len(gaps) == 1
        assert "employed_by" in gaps[0]["gap"]

    # --- gap dict structure -------------------------------------------------

    def test_gap_dict_has_required_keys(self):
        entity = {"type": "PERSON", "name": "Alice"}
        gaps = detect_gaps([entity])
        assert len(gaps) == 1
        gap = gaps[0]
        assert "entity" in gap
        assert "type" in gap
        assert "gap" in gap
