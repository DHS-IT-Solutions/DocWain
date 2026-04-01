"""Unit tests for extraction data models."""

import pytest
from dataclasses import fields, asdict
from src.extraction.models import (
    TriageResult,
    ValidationResult,
    QualityScorecard,
    Entity,
    Relationship,
    TableData,
    Section,
    ExtractionResult,
)


class TestTriageResult:
    def test_instantiation_with_all_fields(self):
        result = TriageResult(
            document_type="scanned",
            engine_weights={"structural": 0.4, "semantic": 0.4, "vision": 0.2},
            preprocessing_directives=["upscale", "denoise"],
            page_types={"page_1": "scanned", "page_2": "clean_digital"},
            confidence=0.85,
        )
        assert result.document_type == "scanned"
        assert result.engine_weights == {"structural": 0.4, "semantic": 0.4, "vision": 0.2}
        assert result.preprocessing_directives == ["upscale", "denoise"]
        assert result.page_types == {"page_1": "scanned", "page_2": "clean_digital"}
        assert result.confidence == 0.85

    def test_confidence_default_is_zero(self):
        result = TriageResult(
            document_type="clean_digital",
            engine_weights={"structural": 1.0},
            preprocessing_directives=[],
            page_types={},
        )
        assert result.confidence == 0.0

    def test_page_types_is_dict(self):
        result = TriageResult(
            document_type="mixed",
            engine_weights={"structural": 0.5, "vision": 0.5},
            preprocessing_directives=["deskew"],
            page_types={"p1": "table_heavy", "p2": "handwritten"},
            confidence=0.7,
        )
        assert isinstance(result.page_types, dict)

    def test_engine_weights_empty_dict(self):
        result = TriageResult(
            document_type="handwritten",
            engine_weights={},
            preprocessing_directives=[],
            page_types={},
            confidence=0.0,
        )
        assert result.engine_weights == {}

    def test_preprocessing_directives_multiple_values(self):
        directives = ["upscale", "denoise", "deskew", "contrast"]
        result = TriageResult(
            document_type="scanned",
            engine_weights={"vision": 1.0},
            preprocessing_directives=directives,
            page_types={},
            confidence=0.6,
        )
        assert result.preprocessing_directives == directives

    def test_field_types(self):
        field_map = {f.name: f for f in fields(TriageResult)}
        assert "document_type" in field_map
        assert "engine_weights" in field_map
        assert "preprocessing_directives" in field_map
        assert "page_types" in field_map
        assert "confidence" in field_map


class TestValidationResult:
    def test_instantiation_passed(self):
        result = ValidationResult(
            passed=True,
            failed_checks=[],
            field_confidences={"invoice_number": 0.95, "total_amount": 0.88},
        )
        assert result.passed is True
        assert result.failed_checks == []
        assert result.field_confidences == {"invoice_number": 0.95, "total_amount": 0.88}
        assert result.retry_recommended is False

    def test_instantiation_failed(self):
        result = ValidationResult(
            passed=False,
            failed_checks=["missing_date", "low_confidence_total"],
            field_confidences={"date": 0.3, "total": 0.45},
            retry_recommended=True,
        )
        assert result.passed is False
        assert "missing_date" in result.failed_checks
        assert "low_confidence_total" in result.failed_checks
        assert result.retry_recommended is True

    def test_retry_recommended_default_is_false(self):
        result = ValidationResult(
            passed=True,
            failed_checks=[],
            field_confidences={},
        )
        assert result.retry_recommended is False

    def test_failed_checks_is_list(self):
        result = ValidationResult(
            passed=False,
            failed_checks=["check_a", "check_b"],
            field_confidences={},
        )
        assert isinstance(result.failed_checks, list)

    def test_field_confidences_is_dict(self):
        result = ValidationResult(
            passed=True,
            failed_checks=[],
            field_confidences={"field_x": 0.99},
        )
        assert isinstance(result.field_confidences, dict)

    def test_field_confidences_empty(self):
        result = ValidationResult(passed=True, failed_checks=[], field_confidences={})
        assert result.field_confidences == {}

    def test_all_fields_present(self):
        field_names = {f.name for f in fields(ValidationResult)}
        assert field_names == {"passed", "failed_checks", "field_confidences", "retry_recommended"}


class TestQualityScorecard:
    def test_instantiation_with_all_fields(self):
        scorecard = QualityScorecard(
            overall_confidence=0.92,
            engine_contributions={"structural": 0.4, "semantic": 0.35, "vision": 0.25},
            conflict_count=2,
            conflict_log=["field mismatch on page 1", "date ambiguity"],
        )
        assert scorecard.overall_confidence == 0.92
        assert scorecard.engine_contributions == {
            "structural": 0.4,
            "semantic": 0.35,
            "vision": 0.25,
        }
        assert scorecard.conflict_count == 2
        assert scorecard.conflict_log == ["field mismatch on page 1", "date ambiguity"]

    def test_conflict_count_default_is_zero(self):
        scorecard = QualityScorecard(
            overall_confidence=0.75,
            engine_contributions={"structural": 1.0},
        )
        assert scorecard.conflict_count == 0

    def test_conflict_log_default_is_empty_list(self):
        scorecard = QualityScorecard(
            overall_confidence=0.75,
            engine_contributions={"structural": 1.0},
        )
        assert scorecard.conflict_log == []

    def test_conflict_log_default_factory_independence(self):
        """Each instance must get its own list, not a shared reference."""
        s1 = QualityScorecard(overall_confidence=0.5, engine_contributions={})
        s2 = QualityScorecard(overall_confidence=0.6, engine_contributions={})
        s1.conflict_log.append("conflict A")
        assert s2.conflict_log == []

    def test_engine_contributions_empty(self):
        scorecard = QualityScorecard(overall_confidence=0.0, engine_contributions={})
        assert scorecard.engine_contributions == {}

    def test_overall_confidence_boundary_values(self):
        low = QualityScorecard(overall_confidence=0.0, engine_contributions={})
        high = QualityScorecard(overall_confidence=1.0, engine_contributions={})
        assert low.overall_confidence == 0.0
        assert high.overall_confidence == 1.0

    def test_all_fields_present(self):
        field_names = {f.name for f in fields(QualityScorecard)}
        assert field_names == {
            "overall_confidence",
            "engine_contributions",
            "conflict_count",
            "conflict_log",
        }
