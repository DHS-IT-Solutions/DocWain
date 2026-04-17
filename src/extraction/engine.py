"""Extraction engine — deterministic Layer 1 + AI Layer 2, then merge.

Flow per document:

    Step 1  deterministic.extract(bytes, filename)  -> RawExtraction
            (PDF/DOCX/XLSX/image/CSV/TXT via native parsers + Tesseract;
             no AI; faithful content capture)

    Step 2  deterministic.validate(raw)             -> quality gates
            (non-blocking; failures logged + attached to metadata)

    Step 3  Four-parallel engines run with Layer 1's text as their input:
              structural, semantic, vision  (currently TODO stubs)
              v2_extractor (Ollama LLM — produces semantic entities/fields)

    Step 4  merger.merge()                          -> ExtractionResult
            Layer 1 owns clean_text + tables (higher fidelity than LLM).
            Layer 2 contributes entities + fields + relationships.

The deterministic stage always runs. V2/AI runs as a semantic-understanding
layer on top of clean Layer 1 text, never against raw bytes.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.extraction import deterministic
from src.extraction.structural import StructuralExtractor
from src.extraction.semantic import SemanticExtractor
from src.extraction.vision import VisionExtractor
from src.extraction.merger import ExtractionMerger
from src.extraction.models import ExtractionResult, TableData
from src.extraction.v2_extractor import V2Extractor
from src.extraction.triage import DocumentTriager

logger = logging.getLogger(__name__)


def _file_type_to_filename(file_type: str) -> str:
    """Coerce a file_type string (``pdf``, ``.pdf``, or full filename) into a filename.

    ``deterministic.extract()`` routes by extension, so we only need something
    with a usable suffix.
    """
    ft = (file_type or "").strip()
    if not ft:
        return "document"
    if "." in ft:
        return ft if not ft.startswith(".") else f"document{ft}"
    return f"document.{ft.lower()}"


def _deterministic_tables_to_model(raw_tables) -> list:
    """Convert deterministic.Table objects into extraction.models.TableData."""
    out = []
    for i, t in enumerate(raw_tables):
        out.append(TableData(
            id=f"det-{i}",
            page=t.page or 0,
            rows=t.n_rows,
            cols=t.n_cols,
            headers=list(t.headers or []),
            data=[list(r) for r in t.rows],
            source="deterministic",
            cross_validated=t.cross_page,  # stitched across pages
        ))
    return out


class ExtractionEngine:
    """Deterministic Layer 1 + AI Layer 2 extraction."""

    def __init__(self, triton_url: str = None, ollama_host: str = None):
        self.structural = StructuralExtractor(triton_url=triton_url)
        self.semantic = SemanticExtractor(ollama_host=ollama_host)
        self.vision = VisionExtractor(ollama_host=ollama_host)
        self.merger = ExtractionMerger()
        self.v2_extractor = V2Extractor(ollama_host=ollama_host)
        self.triager = DocumentTriager()

    def extract(self, document_id: str, subscription_id: str, profile_id: str,
                document_bytes: bytes, file_type: str,
                text_content: Optional[str] = None) -> ExtractionResult:
        """Run deterministic Layer 1, validate, then four-engine Layer 2, merge."""
        # ---- Step 1: deterministic extraction ------------------------------
        t0 = time.monotonic()
        filename = _file_type_to_filename(file_type)
        raw = deterministic.extract(document_bytes, filename)
        t_deterministic = time.monotonic() - t0
        logger.info(
            "Deterministic extraction: doc=%s fmt=%s text_chars=%d blocks=%d tables=%d elapsed=%.3fs",
            document_id, raw.file_format, len(raw.text_full),
            len(raw.blocks), len(raw.tables), t_deterministic,
        )

        # ---- Step 2: validate --------------------------------------------------
        validation = deterministic.validate(raw)
        if not validation["passed"]:
            logger.warning(
                "Deterministic validation FAILED for %s: %s",
                document_id, validation["failed_checks"],
            )
        for advisory in validation["advisories"]:
            logger.info("Deterministic advisory for %s: %s", document_id, advisory)

        # Layer 1 text becomes the authoritative input for Layer 2 engines.
        # If the caller passed text_content we respect it (e.g. pre-extracted
        # text from an upstream stage), else we use Layer 1.
        effective_text = text_content if text_content else raw.text_full

        triage_result = self.triager.triage(
            file_type=file_type,
            has_text_layer=bool(effective_text),
        )

        # ---- Step 3: four-parallel engines ------------------------------------
        structural_result = {}
        semantic_result = {}
        vision_result = {}
        v2_result = {}

        t1 = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self.structural.extract, document_bytes, file_type
                ): "structural",
                executor.submit(
                    self.semantic.extract, effective_text, file_type
                ): "semantic",
                executor.submit(
                    self.vision.extract, document_bytes, file_type
                ): "vision",
                executor.submit(
                    self.v2_extractor.extract, document_bytes, file_type,
                    text_content=effective_text,
                ): "v2",
            }

            for future in as_completed(futures):
                pipeline_name = futures[future]
                try:
                    result = future.result(timeout=600)
                    if pipeline_name == "structural":
                        structural_result = result
                    elif pipeline_name == "semantic":
                        semantic_result = result
                    elif pipeline_name == "vision":
                        vision_result = result
                    elif pipeline_name == "v2":
                        v2_result = result
                    logger.info(f"{pipeline_name} extraction completed for {document_id}")
                except Exception as e:
                    logger.error(f"{pipeline_name} extraction failed for {document_id}: {e}")
        t_parallel = time.monotonic() - t1

        # ---- Step 4: merge ----------------------------------------------------
        merged = self.merger.merge(
            document_id=document_id,
            subscription_id=subscription_id,
            profile_id=profile_id,
            structural=structural_result,
            semantic=semantic_result,
            vision=vision_result,
            v2=v2_result,
            triage=triage_result,
        )

        # Layer 1 owns clean_text + tables. Prefer them when available —
        # deterministic output is higher-fidelity than any LLM-produced text.
        if raw.text_full:
            merged.clean_text = raw.text_full
        det_tables = _deterministic_tables_to_model(raw.tables)
        if det_tables:
            merged.tables = det_tables

        # Attach Layer 1 output + validation result so downstream consumers
        # can see exactly what came from deterministic vs AI.
        merged.raw_extraction = raw.to_dict()
        merged.metadata.setdefault("deterministic_validation", validation)
        merged.metadata.setdefault("deterministic_elapsed_s", round(t_deterministic, 4))
        merged.metadata.setdefault("parallel_engines_elapsed_s", round(t_parallel, 4))
        # Models used: always include deterministic; merger already handled the rest
        models_used = list(merged.metadata.get("models_used", []))
        if "deterministic" not in models_used:
            models_used.insert(0, "deterministic")
        merged.metadata["models_used"] = models_used

        logger.info(
            "Extraction complete doc=%s entities=%d tables=%d confidence=%.2f "
            "deterministic=%.3fs parallel=%.3fs validation=%s",
            document_id, len(merged.entities), len(merged.tables),
            merged.metadata.get("extraction_confidence", 0.0),
            t_deterministic, t_parallel,
            "PASS" if validation["passed"] else "FAIL",
        )

        return merged
