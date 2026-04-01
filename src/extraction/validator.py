"""Post-extraction validation — sanity-checks an ExtractionResult before downstream steps."""

from __future__ import annotations

from typing import Dict, List

from src.extraction.models import ExtractionResult, ValidationResult


class ExtractionValidator:
    """Validates an ExtractionResult against a set of quality checks."""

    def __init__(self, confidence_threshold: float = 0.7, max_retries: int = 2) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

    def validate(self, result: ExtractionResult) -> ValidationResult:
        """Run all validation checks and return a ValidationResult.

        Checks performed:
        - low_confidence: extraction_confidence below threshold
        - date_chronology: start_date > end_date in metadata
        - entity_type_conflict:<name>: same entity name (case-insensitive) has multiple types

        Args:
            result: The ExtractionResult to validate.

        Returns:
            ValidationResult with pass/fail status, failed checks, per-field confidences,
            and whether a retry is recommended.
        """
        failed_checks: List[str] = []
        field_confidences: Dict[str, float] = {}

        # Check 1: Low confidence
        extraction_confidence = result.metadata.get("extraction_confidence", 1.0)
        field_confidences["extraction_confidence"] = float(extraction_confidence)
        if extraction_confidence < self.confidence_threshold:
            failed_checks.append("low_confidence")

        # Check 2: Date chronology
        start_date = result.metadata.get("start_date")
        end_date = result.metadata.get("end_date")
        if start_date is not None and end_date is not None:
            try:
                if start_date > end_date:
                    failed_checks.append("date_chronology")
            except TypeError:
                # If comparison is not supported (mixed types), skip
                pass

        # Check 3: Entity type conflict — same name (lowercase) with different types
        entity_type_map: Dict[str, str] = {}
        for entity in result.entities:
            if hasattr(entity, "text") and hasattr(entity, "type"):
                name = entity.text.lower()
                etype = entity.type
            elif isinstance(entity, dict):
                name = entity.get("text", "").lower()
                etype = entity.get("type", "")
            else:
                continue

            if name in entity_type_map:
                if entity_type_map[name] != etype:
                    check_key = f"entity_type_conflict:{name}"
                    if check_key not in failed_checks:
                        failed_checks.append(check_key)
            else:
                entity_type_map[name] = etype

        passed = len(failed_checks) == 0
        retry_recommended = not passed and self.max_retries > 0

        return ValidationResult(
            passed=passed,
            failed_checks=failed_checks,
            field_confidences=field_confidences,
            retry_recommended=retry_recommended,
        )
