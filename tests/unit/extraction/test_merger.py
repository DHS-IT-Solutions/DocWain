"""Unit tests for ExtractionMerger (src/extraction/merger.py)."""

import pytest

from src.extraction.merger import ExtractionMerger
from src.extraction.models import Entity, ExtractionResult, TriageResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_triage(weights: dict, confidence: float = 0.9) -> TriageResult:
    return TriageResult(
        document_type="clean_digital",
        engine_weights=weights,
        preprocessing_directives=[],
        page_types=["digital"],
        confidence=confidence,
    )


def entity_dict(text: str, etype: str = "ORG", confidence: float = 0.6) -> dict:
    return {"text": text, "type": etype, "confidence": confidence, "locations": []}


@pytest.fixture
def merger():
    return ExtractionMerger()


@pytest.fixture
def empty_engine():
    """An engine payload with no useful content."""
    return {}


@pytest.fixture
def base_structural():
    return {
        "sections": [{"id": "s1", "title": "Intro"}],
        "entities": [entity_dict("Acme Corp", "ORG", 0.8)],
        "tables": [],
    }


@pytest.fixture
def base_semantic():
    return {
        "context": "contract",
        "entities": [entity_dict("Acme Corp", "ORG", 0.6)],
        "relationships": [],
    }


@pytest.fixture
def base_vision():
    return {
        "ocr_text": "Acme Corp Agreement",
        "entities": [],
        "table_images": [],
    }


# ---------------------------------------------------------------------------
# Backward compatibility — no v2, no triage
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_returns_extraction_result(self, merger, base_structural, base_semantic, base_vision):
        result = merger.merge(
            document_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
            structural=base_structural,
            semantic=base_semantic,
            vision=base_vision,
        )
        assert isinstance(result, ExtractionResult)

    def test_no_v2_no_triage_deduplicates_with_flat_boost(
        self, merger, base_structural, base_semantic, base_vision
    ):
        """Without triage, an entity found by two engines gets +0.1 boost."""
        result = merger.merge(
            document_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
            structural=base_structural,  # Acme Corp @ 0.8
            semantic=base_semantic,      # Acme Corp @ 0.6 (duplicate)
            vision=base_vision,
        )
        # Should be deduplicated to one entity
        acme = [e for e in result.entities if e.text.lower() == "acme corp"]
        assert len(acme) == 1
        # Boost: starts at 0.8, one duplicate → +0.1 = 0.9
        assert acme[0].confidence == pytest.approx(0.9, abs=1e-6)

    def test_no_v2_no_triage_unique_entities_unchanged(self, merger):
        structural = {"entities": [entity_dict("Alpha Inc", "ORG", 0.7)]}
        semantic = {"entities": [entity_dict("Beta LLC", "ORG", 0.5)]}
        vision = {}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision=vision,
        )
        assert len(result.entities) == 2

    def test_metadata_has_quality_scorecard(self, merger, base_structural, base_semantic, base_vision):
        result = merger.merge(
            document_id="doc-1", subscription_id="sub-1", profile_id="prof-1",
            structural=base_structural, semantic=base_semantic, vision=base_vision,
        )
        assert "quality_scorecard" in result.metadata
        qs = result.metadata["quality_scorecard"]
        assert "engine_contributions" in qs
        assert "conflict_count" in qs

    def test_extraction_confidence_without_triage(self, merger):
        structural = {"sections": ["s1"]}
        semantic = {"context": "invoice"}
        vision = {"ocr_text": "some text"}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision=vision,
        )
        # Average of (0.8, 0.7, 0.6) = 0.7
        assert result.metadata["extraction_confidence"] == pytest.approx(0.7, abs=1e-6)


# ---------------------------------------------------------------------------
# Triage-weighted merging
# ---------------------------------------------------------------------------

class TestTriageWeightedMerge:

    def test_weighted_confidence_single_engine(self, merger):
        """When only one engine finds an entity, its confidence is used directly."""
        triage = make_triage({"structural": 0.8, "semantic": 0.1, "vision": 0.1, "v2": 0.0})
        structural = {"entities": [entity_dict("DocWain Ltd", "ORG", 0.7)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic={}, vision={},
            triage=triage,
        )
        entity = result.entities[0]
        assert entity.confidence == pytest.approx(0.7, abs=1e-6)

    def test_weighted_confidence_two_engines_agree(self, merger):
        """Two engines agree → weighted average + 0.05 boost."""
        weights = {"structural": 0.6, "semantic": 0.4, "vision": 0.0, "v2": 0.0}
        triage = make_triage(weights)
        structural = {"entities": [entity_dict("Gamma SA", "ORG", 0.8)]}
        semantic = {"entities": [entity_dict("Gamma SA", "ORG", 0.6)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision={},
            triage=triage,
        )
        gamma = [e for e in result.entities if e.text.lower() == "gamma sa"]
        assert len(gamma) == 1

        # weighted_conf = (0.6*0.8 + 0.4*0.6) / (0.6 + 0.4) = (0.48 + 0.24) / 1.0 = 0.72
        # boost = 0.72 + 0.05 = 0.77
        assert gamma[0].confidence == pytest.approx(0.77, abs=1e-6)

    def test_weighted_confidence_three_engines_agree(self, merger):
        """Three engines agree → weighted average + 0.10 boost (2 extra engines)."""
        weights = {"structural": 0.5, "semantic": 0.3, "vision": 0.2, "v2": 0.0}
        triage = make_triage(weights)
        structural = {"entities": [entity_dict("Delta PLC", "ORG", 0.9)]}
        semantic = {"entities": [entity_dict("Delta PLC", "ORG", 0.7)]}
        vision = {"entities": [entity_dict("Delta PLC", "ORG", 0.5)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision=vision,
            triage=triage,
        )
        delta = [e for e in result.entities if e.text.lower() == "delta plc"]
        assert len(delta) == 1

        # weighted_conf = (0.5*0.9 + 0.3*0.7 + 0.2*0.5) / (0.5+0.3+0.2)
        #               = (0.45 + 0.21 + 0.10) / 1.0 = 0.76
        # boost = 0.76 + 0.05*2 = 0.86
        assert delta[0].confidence == pytest.approx(0.86, abs=1e-6)

    def test_confidence_capped_at_1(self, merger):
        """Confidence never exceeds 1.0 even with large boosts."""
        weights = {"structural": 0.5, "semantic": 0.5, "vision": 0.0, "v2": 0.0}
        triage = make_triage(weights)
        structural = {"entities": [entity_dict("Epsilon GmbH", "ORG", 1.0)]}
        semantic = {"entities": [entity_dict("Epsilon GmbH", "ORG", 1.0)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision={},
            triage=triage,
        )
        eps = result.entities[0]
        assert eps.confidence <= 1.0

    def test_triage_confidence_used_as_extraction_confidence(self, merger):
        """When triage is provided, its confidence populates extraction_confidence."""
        triage = make_triage({"structural": 1.0, "semantic": 0.0, "vision": 0.0, "v2": 0.0},
                             confidence=0.88)
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural={}, semantic={}, vision={},
            triage=triage,
        )
        assert result.metadata["extraction_confidence"] == pytest.approx(0.88, abs=1e-6)

    def test_engines_with_zero_weight_contribute_nothing_to_score(self, merger):
        """An engine with weight 0 still contributes its entity but not to weighted sum."""
        weights = {"structural": 1.0, "semantic": 0.0, "vision": 0.0, "v2": 0.0}
        triage = make_triage(weights)
        structural = {"entities": [entity_dict("Zeta SA", "ORG", 0.8)]}
        semantic = {"entities": [entity_dict("Zeta SA", "ORG", 0.2)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision={},
            triage=triage,
        )
        zeta = [e for e in result.entities if e.text.lower() == "zeta sa"]
        assert len(zeta) == 1
        # Only structural has weight; weighted_conf = 0.8, boost +0.05 = 0.85
        assert zeta[0].confidence == pytest.approx(0.85, abs=1e-6)


# ---------------------------------------------------------------------------
# Conflict logging in QualityScorecard
# ---------------------------------------------------------------------------

class TestQualityScorecard:

    def test_conflict_logged_when_engines_agree(self, merger):
        """conflict_count increments for each entity resolved across multiple engines."""
        triage = make_triage({"structural": 0.5, "semantic": 0.5, "vision": 0.0, "v2": 0.0})
        structural = {"entities": [
            entity_dict("Eta Inc", "ORG", 0.7),
            entity_dict("Theta Corp", "ORG", 0.6),
        ]}
        semantic = {"entities": [
            entity_dict("Eta Inc", "ORG", 0.5),   # conflict
        ]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision={},
            triage=triage,
        )
        qs = result.metadata["quality_scorecard"]
        # One entity found by two engines → 1 conflict
        assert qs["conflict_count"] == 1

    def test_no_conflict_when_all_entities_unique(self, merger):
        triage = make_triage({"structural": 0.5, "semantic": 0.5, "vision": 0.0, "v2": 0.0})
        structural = {"entities": [entity_dict("Alpha", "ORG", 0.7)]}
        semantic = {"entities": [entity_dict("Beta", "ORG", 0.6)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision={},
            triage=triage,
        )
        qs = result.metadata["quality_scorecard"]
        assert qs["conflict_count"] == 0

    def test_engine_contributions_tracked(self, merger):
        triage = make_triage({"structural": 0.6, "semantic": 0.4, "vision": 0.0, "v2": 0.0})
        structural = {"entities": [entity_dict("Iota", "ORG", 0.7)]}
        semantic = {"entities": [entity_dict("Kappa", "ORG", 0.5)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision={},
            triage=triage,
        )
        contrib = result.metadata["quality_scorecard"]["engine_contributions"]
        assert contrib["structural"] == 1
        assert contrib["semantic"] == 1
        assert contrib["vision"] == 0

    def test_scorecard_present_without_triage(self, merger):
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural={}, semantic={}, vision={},
        )
        assert "quality_scorecard" in result.metadata


# ---------------------------------------------------------------------------
# V2 engine integration
# ---------------------------------------------------------------------------

class TestV2Integration:

    def test_v2_entities_included_without_triage(self, merger):
        v2 = {"entities": [entity_dict("Lambda LLC", "ORG", 0.75)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural={}, semantic={}, vision={}, v2=v2,
        )
        names = [e.text for e in result.entities]
        assert "Lambda LLC" in names

    def test_v2_entities_weighted_with_triage(self, merger):
        weights = {"structural": 0.0, "semantic": 0.0, "vision": 0.0, "v2": 1.0}
        triage = make_triage(weights)
        v2 = {"entities": [entity_dict("Mu Corp", "ORG", 0.82)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural={}, semantic={}, vision={},
            v2=v2, triage=triage,
        )
        mu = [e for e in result.entities if e.text == "Mu Corp"]
        assert len(mu) == 1
        assert mu[0].confidence == pytest.approx(0.82, abs=1e-6)

    def test_v2_deduplicates_with_other_engines(self, merger):
        weights = {"structural": 0.5, "semantic": 0.0, "vision": 0.0, "v2": 0.5}
        triage = make_triage(weights)
        structural = {"entities": [entity_dict("Nu SA", "ORG", 0.8)]}
        v2 = {"entities": [entity_dict("Nu SA", "ORG", 0.6)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic={}, vision={},
            v2=v2, triage=triage,
        )
        nu = [e for e in result.entities if e.text.lower() == "nu sa"]
        assert len(nu) == 1
        # weighted_conf = (0.5*0.8 + 0.5*0.6) / 1.0 = 0.7, boost +0.05 = 0.75
        assert nu[0].confidence == pytest.approx(0.75, abs=1e-6)

    def test_merge_without_v2_still_works(self, merger):
        """Omitting v2 entirely (default None) is fully backward compatible."""
        structural = {"entities": [entity_dict("Xi Corp", "ORG", 0.7)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic={}, vision={},
        )
        assert len(result.entities) == 1

    def test_v2_adds_docwain_v2_to_models_used(self, merger):
        v2 = {"entities": [entity_dict("Omicron", "ORG", 0.8)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural={}, semantic={}, vision={}, v2=v2,
        )
        assert "docwain-v2" in result.metadata["models_used"]

    def test_v2_clean_text_preferred_over_vision_ocr(self, merger):
        v2 = {"clean_text": "V2 clean text"}
        vision = {"ocr_text": "Vision OCR text"}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural={}, semantic={}, vision=vision, v2=v2,
        )
        assert result.clean_text == "V2 clean text"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_empty_engines(self, merger):
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural={}, semantic={}, vision={},
        )
        assert result.entities == []
        assert result.relationships == []
        assert result.tables == []
        assert result.clean_text == ""

    def test_entity_type_case_insensitive_dedup(self, merger):
        """Entities with same text and type varying only in case are deduplicated."""
        structural = {"entities": [{"text": "Pi Inc", "type": "org", "confidence": 0.7}]}
        semantic = {"entities": [{"text": "Pi Inc", "type": "ORG", "confidence": 0.5}]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision={},
        )
        assert len([e for e in result.entities if e.text == "Pi Inc"]) == 1

    def test_entity_text_whitespace_insensitive_dedup(self, merger):
        """Leading/trailing whitespace on entity text is stripped for dedup."""
        structural = {"entities": [{"text": "  Rho Ltd  ", "type": "ORG", "confidence": 0.7}]}
        semantic = {"entities": [{"text": "Rho Ltd", "type": "ORG", "confidence": 0.6}]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic=semantic, vision={},
        )
        assert len(result.entities) == 1

    def test_invalid_entity_entries_skipped(self, merger):
        """Non-dict, non-Entity entries in entity lists are silently skipped."""
        structural = {"entities": [None, 42, "bad", entity_dict("Valid", "ORG", 0.7)]}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic={}, vision={},
        )
        assert len(result.entities) == 1
        assert result.entities[0].text == "Valid"

    def test_result_to_dict_serializable(self, merger):
        structural = {"entities": [entity_dict("Sigma", "ORG", 0.9)], "sections": []}
        result = merger.merge(
            document_id="d", subscription_id="s", profile_id="p",
            structural=structural, semantic={}, vision={},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["document_id"] == "d"
