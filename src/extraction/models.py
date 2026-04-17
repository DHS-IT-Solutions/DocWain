"""Extraction pipeline data models."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class TriageResult:
    """Output of DocumentTriager.triage() — routing decision for a document."""
    document_type: str                          # clean_digital | scanned | handwritten | mixed | table_heavy
    engine_weights: Dict[str, float]            # structural, semantic, vision, v2
    preprocessing_directives: List[str]         # upscale | denoise | deskew | contrast
    page_types: Dict[str, str]                  # per-page classification keyed by page identifier
    confidence: float = 0.0                    # 0.0 – 1.0


@dataclass
class ValidationResult:
    """Result of post-extraction validation checks."""
    passed: bool
    failed_checks: List[str]
    field_confidences: Dict[str, float]
    retry_recommended: bool = False


@dataclass
class QualityScorecard:
    """Aggregated quality metrics across extraction engines."""
    overall_confidence: float
    engine_contributions: Dict[str, float]
    conflict_count: int = 0
    conflict_log: List[str] = field(default_factory=list)


@dataclass
class Entity:
    text: str
    type: str  # ORG, PERSON, DATE, AMOUNT, REGULATION, etc.
    confidence: float
    source: str  # "structural", "semantic", "vision"
    locations: list = field(default_factory=list)


@dataclass
class Relationship:
    subject: str
    predicate: str
    object: str
    confidence: float
    evidence: str


@dataclass
class TableData:
    id: str
    page: int
    rows: int
    cols: int
    headers: list
    data: list
    source: str  # "structural", "vision"
    cross_validated: bool = False


@dataclass
class Section:
    id: str
    title: str
    level: int
    start_page: int
    end_page: int
    content: str


@dataclass
class ExtractionResult:
    """Unified output from the three-model extraction pipeline."""
    document_id: str
    subscription_id: str
    profile_id: str
    clean_text: str
    structure: dict
    entities: list
    relationships: list
    tables: list
    metadata: dict
    # Layer 1 deterministic extraction (see src/extraction/deterministic.py).
    # Optional / default None for backward compat with callers that construct
    # ExtractionResult directly without wiring the deterministic stage.
    raw_extraction: Optional[dict] = None

    def to_dict(self) -> dict:
        """Serialize to dict for Azure Blob storage."""
        return {
            "document_id": self.document_id,
            "subscription_id": self.subscription_id,
            "profile_id": self.profile_id,
            "clean_text": self.clean_text,
            "structure": self.structure,
            "entities": [vars(e) if hasattr(e, '__dict__') else e for e in self.entities],
            "relationships": [vars(r) if hasattr(r, '__dict__') else r for r in self.relationships],
            "tables": [vars(t) if hasattr(t, '__dict__') else t for t in self.tables],
            "metadata": self.metadata,
            "raw_extraction": self.raw_extraction,
        }

    def to_summary(self) -> dict:
        """Generate MongoDB summary (no document content)."""
        return {
            "page_count": self.metadata.get("page_count", 0),
            "entity_count": len(self.entities),
            "section_count": len(self.structure.get("sections", [])),
            "table_count": len(self.tables),
            "doc_type_detected": self.metadata.get("doc_type_detected", "unknown"),
            "extraction_confidence": self.metadata.get("extraction_confidence", 0.0),
            "models_used": self.metadata.get("models_used", []),
        }
