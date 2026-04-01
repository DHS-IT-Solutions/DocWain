"""Extraction engine — orchestrates four-model parallel extraction with triage."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extraction.structural import StructuralExtractor
from src.extraction.semantic import SemanticExtractor
from src.extraction.vision import VisionExtractor
from src.extraction.merger import ExtractionMerger
from src.extraction.models import ExtractionResult
from src.extraction.v2_extractor import V2Extractor
from src.extraction.triage import DocumentTriager

logger = logging.getLogger(__name__)


class ExtractionEngine:
    """Orchestrates parallel extraction using four model pipelines with adaptive triage."""

    def __init__(self, triton_url: str = None, ollama_host: str = None):
        self.structural = StructuralExtractor(triton_url=triton_url)
        self.semantic = SemanticExtractor(ollama_host=ollama_host)
        self.vision = VisionExtractor(ollama_host=ollama_host)
        self.merger = ExtractionMerger()
        self.v2_extractor = V2Extractor(ollama_host=ollama_host)
        self.triager = DocumentTriager()

    def extract(self, document_id: str, subscription_id: str, profile_id: str,
                document_bytes: bytes, file_type: str,
                text_content: str = None) -> ExtractionResult:
        """Run triage then four-model extraction in parallel, then merge results."""
        triage_result = self.triager.triage(
            file_type=file_type,
            has_text_layer=bool(text_content),
        )

        structural_result = {}
        semantic_result = {}
        vision_result = {}
        v2_result = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self.structural.extract, document_bytes, file_type
                ): "structural",
                executor.submit(
                    self.semantic.extract, text_content or "", file_type
                ): "semantic",
                executor.submit(
                    self.vision.extract, document_bytes, file_type
                ): "vision",
                executor.submit(
                    self.v2_extractor.extract, document_bytes, file_type,
                    text_content=text_content or "",
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

        logger.info(
            f"Extraction complete for {document_id}: "
            f"{len(merged.entities)} entities, "
            f"{len(merged.tables)} tables, "
            f"confidence={merged.metadata.get('extraction_confidence', 0):.2f}"
        )

        return merged
