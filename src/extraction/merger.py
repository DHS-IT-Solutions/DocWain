"""Merger — reconciles outputs from structural, semantic, vision, and V2 pipelines."""

from typing import Optional
from src.extraction.models import ExtractionResult, Entity, Relationship, TableData, TriageResult
import logging

logger = logging.getLogger(__name__)

# Canonical engine names used in triage weights and source tracking
_ENGINE_NAMES = ("structural", "semantic", "vision", "v2")


class QualityScorecard:
    """Tracks per-engine contributions and conflict count for a merge run."""

    def __init__(self):
        self.engine_contributions: dict[str, int] = {e: 0 for e in _ENGINE_NAMES}
        self.conflict_count: int = 0

    def to_dict(self) -> dict:
        return {
            "engine_contributions": dict(self.engine_contributions),
            "conflict_count": self.conflict_count,
        }


class ExtractionMerger:
    """Merges and reconciles outputs from up to four extraction pipelines.

    Responsibilities:
    - Deduplicate entities by text+type match across all engines
    - Weight confidence by triage engine_weights when available
    - Cross-validate tables (structural vs vision)
    - Reconcile section boundaries
    - Assign confidence scores based on cross-model agreement
    - Build unified reading order
    - Emit a QualityScorecard in result metadata
    """

    def merge(self, document_id: str, subscription_id: str, profile_id: str,
              structural: dict, semantic: dict, vision: dict,
              page_count: int = 0,
              v2: Optional[dict] = None,
              triage: Optional[TriageResult] = None) -> ExtractionResult:
        """Merge up to four pipeline outputs into a unified ExtractionResult.

        Parameters
        ----------
        document_id, subscription_id, profile_id:
            Standard document identifiers.
        structural, semantic, vision:
            Extraction outputs from the three baseline engines.
        page_count:
            Total page count of the source document.
        v2:
            Optional output from the V2 (vision-grafted Qwen3) engine.
        triage:
            Optional TriageResult whose engine_weights govern confidence scoring.
            When None, falls back to the original flat +0.1 boost behaviour.
        """

        scorecard = QualityScorecard()

        entities = self._merge_entities(
            structural.get("entities", []),
            semantic.get("entities", []),
            vision.get("entities", []),
            v2_entities=(v2.get("entities", []) if v2 else []),
            triage=triage,
            scorecard=scorecard,
        )

        relationships = []
        for r in semantic.get("relationships", []):
            if isinstance(r, Relationship):
                relationships.append(r)
            elif isinstance(r, dict):
                relationships.append(Relationship(**r))

        # V2 may also contribute relationships
        if v2:
            for r in v2.get("relationships", []):
                if isinstance(r, Relationship):
                    relationships.append(r)
                elif isinstance(r, dict):
                    relationships.append(Relationship(**r))

        tables = self._merge_tables(
            structural.get("tables", []),
            vision.get("table_images", [])
        )

        # Build clean text: prefer structural reading order, fill with vision OCR
        clean_text = self._build_clean_text(structural, semantic, vision, v2)

        models_used = []
        if structural.get("layout") or structural.get("sections"):
            models_used.append("layoutlm-v3")
        if semantic.get("entities") or semantic.get("context"):
            models_used.append("DHS/DocWain:latest")
        if vision.get("ocr_text") or vision.get("scanned_text"):
            models_used.append("glm-ocr")
        if v2 and (v2.get("entities") or v2.get("ocr_text")):
            models_used.append("docwain-v2")

        extraction_confidence = self._calculate_confidence(
            structural, semantic, vision, v2, triage
        )

        return ExtractionResult(
            document_id=document_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            clean_text=clean_text,
            structure={
                "sections": structural.get("sections", []),
                "headers": structural.get("headers", []),
                "footers": structural.get("footers", []),
                "reading_order": structural.get("reading_order", [])
            },
            entities=entities,
            relationships=relationships,
            tables=tables,
            metadata={
                "page_count": page_count,
                "language": "en",
                "doc_type_detected": semantic.get("context", "unknown"),
                "models_used": models_used,
                "extraction_confidence": extraction_confidence,
                "quality_scorecard": scorecard.to_dict(),
            }
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _merge_entities(
        self,
        structural_entities,
        semantic_entities,
        vision_entities,
        v2_entities=None,
        triage: Optional[TriageResult] = None,
        scorecard: Optional[QualityScorecard] = None,
    ) -> list:
        """Deduplicate entities across engines with optional triage-weighted confidence.

        When *triage* is supplied:
            weighted_confidence = Σ(weight_i * conf_i) / Σ(weight_i for agreeing engines)
            Then boost by 0.05 for each additional engine beyond the first.

        When *triage* is None:
            Falls back to the original flat +0.1 boost per additional engine.
        """
        if v2_entities is None:
            v2_entities = []

        weights: dict[str, float] = {}
        if triage is not None:
            weights = triage.engine_weights  # e.g. {"structural": 0.4, "semantic": 0.3, ...}

        # key → list of (engine_name, Entity)
        groups: dict[tuple, list[tuple[str, Entity]]] = {}
        order: list[tuple] = []  # preserve first-seen order

        source_lists = [
            ("structural", structural_entities),
            ("semantic", semantic_entities),
            ("vision", vision_entities),
            ("v2", v2_entities),
        ]

        for engine_name, entity_list in source_lists:
            for e in entity_list:
                if isinstance(e, dict):
                    e = Entity(
                        text=e.get("text", ""),
                        type=e.get("type", "UNKNOWN"),
                        confidence=e.get("confidence", 0.5),
                        source=engine_name,
                        locations=e.get("locations", [])
                    )
                elif not isinstance(e, Entity):
                    continue

                key = (e.text.lower().strip(), e.type.upper())
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append((engine_name, e))

        merged = []
        for key in order:
            hits = groups[key]
            representative = hits[0][1]  # first entity found; we update its confidence

            if triage is not None:
                # Weighted confidence across engines that found this entity
                total_weight = 0.0
                weighted_sum = 0.0
                for engine_name, entity in hits:
                    w = weights.get(engine_name, 0.0)
                    weighted_sum += w * entity.confidence
                    total_weight += w

                if total_weight > 0.0:
                    base_confidence = weighted_sum / total_weight
                else:
                    # All contributing engines have zero weight; simple average
                    base_confidence = sum(e.confidence for _, e in hits) / len(hits)

                # Boost for multi-engine agreement (0.05 per extra engine)
                extra_engines = len(hits) - 1
                final_confidence = min(1.0, base_confidence + 0.05 * extra_engines)

                if extra_engines > 0:
                    # Log conflict / agreement
                    if scorecard:
                        scorecard.conflict_count += 1
                        logger.debug(
                            "Entity conflict resolved: %s | engines: %s",
                            key, [eng for eng, _ in hits]
                        )
            else:
                # Legacy behaviour: flat +0.1 boost for each duplicate
                final_confidence = representative.confidence
                for i in range(1, len(hits)):
                    final_confidence = min(1.0, final_confidence + 0.1)

            representative.confidence = final_confidence

            # Track engine contributions in scorecard
            if scorecard:
                for engine_name, _ in hits:
                    if engine_name in scorecard.engine_contributions:
                        scorecard.engine_contributions[engine_name] += 1

            merged.append(representative)

        return merged

    def _merge_tables(self, structural_tables, vision_tables) -> list:
        """Cross-validate tables from structural and vision pipelines."""
        tables = []
        for t in structural_tables:
            if isinstance(t, dict):
                t = TableData(
                    id=t.get("id", ""),
                    page=t.get("page", 0),
                    rows=t.get("rows", 0),
                    cols=t.get("cols", 0),
                    headers=t.get("headers", []),
                    data=t.get("data", []),
                    source="structural"
                )
            tables.append(t)
        # TODO: Cross-validate with vision tables
        return tables

    def _build_clean_text(self, structural, semantic, vision, v2=None) -> str:
        """Build clean text from best available sources."""
        # Priority: structural reading order > V2 OCR > vision OCR > semantic context
        if structural.get("reading_order"):
            # TODO: reconstruct text from reading order
            pass

        if v2 and (v2.get("ocr_text") or v2.get("clean_text")):
            return v2.get("ocr_text") or v2.get("clean_text", "")

        ocr_text = vision.get("ocr_text", "") or vision.get("scanned_text", "")
        if ocr_text:
            return ocr_text

        return semantic.get("summary", "") or semantic.get("context", "")

    def _calculate_confidence(self, structural, semantic, vision,
                              v2=None,
                              triage: Optional[TriageResult] = None) -> float:
        """Calculate overall extraction confidence based on model agreement."""
        if triage is not None:
            # Use triage's own confidence as the primary signal
            return triage.confidence

        scores = []
        if structural.get("layout") or structural.get("sections"):
            scores.append(0.8)
        if semantic.get("entities") or semantic.get("context"):
            scores.append(0.7)
        if vision.get("ocr_text") or vision.get("scanned_text"):
            scores.append(0.6)
        if v2 and (v2.get("entities") or v2.get("ocr_text")):
            scores.append(0.85)

        if not scores:
            return 0.0
        return sum(scores) / len(scores)
