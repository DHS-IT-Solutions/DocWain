"""Unit tests for ExtractionValidator (Task 25)."""

import pytest

from src.extraction.models import Entity, ExtractionResult, ValidationResult
from src.extraction.validator import ExtractionValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    extraction_confidence: float = 0.9,
    start_date=None,
    end_date=None,
    entities=None,
) -> ExtractionResult:
    metadata = {"extraction_confidence": extraction_confidence}
    if start_date is not None:
        metadata["start_date"] = start_date
    if end_date is not None:
        metadata["end_date"] = end_date
    return ExtractionResult(
        document_id="doc-1",
        subscription_id="sub-1",
        profile_id="profile-1",
        clean_text="sample text",
        structure={},
        entities=entities or [],
        relationships=[],
        tables=[],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestExtractionValidatorInit:
    def test_defaults(self):
        v = ExtractionValidator()
        assert v.confidence_threshold == 0.7
        assert v.max_retries == 2

    def test_custom_values(self):
        v = ExtractionValidator(confidence_threshold=0.5, max_retries=5)
        assert v.confidence_threshold == 0.5
        assert v.max_retries == 5


# ---------------------------------------------------------------------------
# validate() — happy path
# ---------------------------------------------------------------------------

class TestValidatePass:
    def test_returns_validation_result(self):
        v = ExtractionValidator()
        result = v.validate(_make_result())
        assert isinstance(result, ValidationResult)

    def test_passes_when_all_ok(self):
        v = ExtractionValidator()
        vr = v.validate(_make_result(extraction_confidence=0.95))
        assert vr.passed is True
        assert vr.failed_checks == []
        assert vr.retry_recommended is False

    def test_field_confidences_populated(self):
        v = ExtractionValidator()
        vr = v.validate(_make_result(extraction_confidence=0.85))
        assert "extraction_confidence" in vr.field_confidences
        assert vr.field_confidences["extraction_confidence"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Check 1: low_confidence
# ---------------------------------------------------------------------------

class TestLowConfidenceCheck:
    def test_below_threshold_fails(self):
        v = ExtractionValidator(confidence_threshold=0.7)
        vr = v.validate(_make_result(extraction_confidence=0.5))
        assert "low_confidence" in vr.failed_checks
        assert vr.passed is False

    def test_exactly_at_threshold_passes(self):
        v = ExtractionValidator(confidence_threshold=0.7)
        vr = v.validate(_make_result(extraction_confidence=0.7))
        assert "low_confidence" not in vr.failed_checks

    def test_above_threshold_passes(self):
        v = ExtractionValidator(confidence_threshold=0.7)
        vr = v.validate(_make_result(extraction_confidence=0.8))
        assert "low_confidence" not in vr.failed_checks

    def test_custom_threshold_respected(self):
        v = ExtractionValidator(confidence_threshold=0.9)
        vr = v.validate(_make_result(extraction_confidence=0.85))
        assert "low_confidence" in vr.failed_checks

    def test_missing_confidence_defaults_to_pass(self):
        """When extraction_confidence is absent, default 1.0 should pass."""
        result = _make_result()
        del result.metadata["extraction_confidence"]
        v = ExtractionValidator()
        vr = v.validate(result)
        assert "low_confidence" not in vr.failed_checks


# ---------------------------------------------------------------------------
# Check 2: date_chronology
# ---------------------------------------------------------------------------

class TestDateChronologyCheck:
    def test_start_after_end_fails(self):
        v = ExtractionValidator()
        vr = v.validate(_make_result(start_date="2024-12-01", end_date="2024-01-01"))
        assert "date_chronology" in vr.failed_checks

    def test_start_before_end_passes(self):
        v = ExtractionValidator()
        vr = v.validate(_make_result(start_date="2024-01-01", end_date="2024-12-01"))
        assert "date_chronology" not in vr.failed_checks

    def test_start_equals_end_passes(self):
        v = ExtractionValidator()
        vr = v.validate(_make_result(start_date="2024-06-01", end_date="2024-06-01"))
        assert "date_chronology" not in vr.failed_checks

    def test_missing_start_date_skips_check(self):
        v = ExtractionValidator()
        vr = v.validate(_make_result(end_date="2024-12-01"))
        assert "date_chronology" not in vr.failed_checks

    def test_missing_end_date_skips_check(self):
        v = ExtractionValidator()
        vr = v.validate(_make_result(start_date="2024-01-01"))
        assert "date_chronology" not in vr.failed_checks

    def test_integer_dates_start_after_end_fails(self):
        v = ExtractionValidator()
        vr = v.validate(_make_result(start_date=20241201, end_date=20240101))
        assert "date_chronology" in vr.failed_checks


# ---------------------------------------------------------------------------
# Check 3: entity_type_conflict
# ---------------------------------------------------------------------------

class TestEntityTypeConflictCheck:
    def _entity(self, text, etype):
        return Entity(text=text, type=etype, confidence=0.9, source="structural")

    def test_same_name_different_types_fails(self):
        entities = [
            self._entity("Acme", "ORG"),
            self._entity("Acme", "PERSON"),
        ]
        v = ExtractionValidator()
        vr = v.validate(_make_result(entities=entities))
        assert "entity_type_conflict:acme" in vr.failed_checks

    def test_same_name_same_type_passes(self):
        entities = [
            self._entity("Acme", "ORG"),
            self._entity("Acme", "ORG"),
        ]
        v = ExtractionValidator()
        vr = v.validate(_make_result(entities=entities))
        assert not any(c.startswith("entity_type_conflict") for c in vr.failed_checks)

    def test_case_insensitive_matching(self):
        entities = [
            self._entity("acme", "ORG"),
            self._entity("ACME", "PERSON"),
        ]
        v = ExtractionValidator()
        vr = v.validate(_make_result(entities=entities))
        assert "entity_type_conflict:acme" in vr.failed_checks

    def test_different_names_no_conflict(self):
        entities = [
            self._entity("Acme", "ORG"),
            self._entity("John", "PERSON"),
        ]
        v = ExtractionValidator()
        vr = v.validate(_make_result(entities=entities))
        assert not any(c.startswith("entity_type_conflict") for c in vr.failed_checks)

    def test_multiple_conflicts_reported_separately(self):
        entities = [
            self._entity("Alpha", "ORG"),
            self._entity("Alpha", "PERSON"),
            self._entity("Beta", "DATE"),
            self._entity("Beta", "AMOUNT"),
        ]
        v = ExtractionValidator()
        vr = v.validate(_make_result(entities=entities))
        assert "entity_type_conflict:alpha" in vr.failed_checks
        assert "entity_type_conflict:beta" in vr.failed_checks

    def test_dict_entities_supported(self):
        """Entities supplied as plain dicts (e.g., from to_dict serialization)."""
        entities = [
            {"text": "Acme", "type": "ORG", "confidence": 0.9, "source": "structural"},
            {"text": "acme", "type": "PERSON", "confidence": 0.8, "source": "vision"},
        ]
        v = ExtractionValidator()
        vr = v.validate(_make_result(entities=entities))
        assert "entity_type_conflict:acme" in vr.failed_checks

    def test_conflict_name_is_deduplicated(self):
        """Three entities with same name + two different types → only one check entry."""
        entities = [
            self._entity("Acme", "ORG"),
            self._entity("Acme", "PERSON"),
            self._entity("Acme", "DATE"),
        ]
        v = ExtractionValidator()
        vr = v.validate(_make_result(entities=entities))
        conflict_entries = [c for c in vr.failed_checks if c == "entity_type_conflict:acme"]
        assert len(conflict_entries) == 1


# ---------------------------------------------------------------------------
# retry_recommended behaviour
# ---------------------------------------------------------------------------

class TestRetryRecommended:
    def test_retry_recommended_when_failed_and_retries_allowed(self):
        v = ExtractionValidator(confidence_threshold=0.9, max_retries=2)
        vr = v.validate(_make_result(extraction_confidence=0.5))
        assert vr.retry_recommended is True

    def test_no_retry_when_max_retries_zero(self):
        v = ExtractionValidator(confidence_threshold=0.9, max_retries=0)
        vr = v.validate(_make_result(extraction_confidence=0.5))
        assert vr.retry_recommended is False

    def test_no_retry_when_passed(self):
        v = ExtractionValidator()
        vr = v.validate(_make_result(extraction_confidence=0.95))
        assert vr.retry_recommended is False


# ---------------------------------------------------------------------------
# Multiple simultaneous failures
# ---------------------------------------------------------------------------

class TestMultipleFailures:
    def test_all_three_checks_can_fail_simultaneously(self):
        entities = [
            Entity(text="Acme", type="ORG", confidence=0.9, source="structural"),
            Entity(text="Acme", type="PERSON", confidence=0.8, source="vision"),
        ]
        v = ExtractionValidator(confidence_threshold=0.8)
        vr = v.validate(
            _make_result(
                extraction_confidence=0.5,
                start_date="2025-12-01",
                end_date="2025-01-01",
                entities=entities,
            )
        )
        assert "low_confidence" in vr.failed_checks
        assert "date_chronology" in vr.failed_checks
        assert "entity_type_conflict:acme" in vr.failed_checks
        assert vr.passed is False
        assert vr.retry_recommended is True
