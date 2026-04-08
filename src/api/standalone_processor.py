"""Core orchestrator for the DocWain Standalone API.

Ties together extraction, intelligence, chunking/embedding, retrieval and
generation into a single coherent pipeline.
"""
from __future__ import annotations

import io
import json
import re
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# File-type detection
# ---------------------------------------------------------------------------

_EXT_MAP: Dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".pptx": "pptx",
    ".ppt": "pptx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".txt": "txt",
    ".md": "txt",
    ".rtf": "txt",
}


def detect_file_type(filename: str, content_bytes: bytes) -> str:
    """Return a normalised file-type string for *filename*.

    Extension is checked first.  If the extension is absent or unrecognised,
    magic bytes are used as a fallback.
    """
    if filename:
        lower = filename.lower()
        for ext, ftype in _EXT_MAP.items():
            if lower.endswith(ext):
                return ftype

    # Magic bytes fallback
    if content_bytes:
        if content_bytes[:4] == b"%PDF":
            return "pdf"
        if content_bytes[:2] == b"PK":
            return "docx"
        if content_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "image"
        if content_bytes[:3] == b"\xff\xd8\xff":
            return "image"

    return "txt"


# ---------------------------------------------------------------------------
# Document extractor singleton
# ---------------------------------------------------------------------------

_extractor_instance = None


def _get_document_extractor():
    """Return a lazy singleton DocumentExtractor."""
    global _extractor_instance
    if _extractor_instance is None:
        from src.api.dw_document_extractor import DocumentExtractor
        _extractor_instance = DocumentExtractor()
    return _extractor_instance


# ---------------------------------------------------------------------------
# Image OCR helper
# ---------------------------------------------------------------------------


def _ocr_image_bytes(content: bytes) -> str:
    """Run pytesseract OCR on raw image bytes and return the extracted text."""
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(content))
        return pytesseract.image_to_string(img)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image OCR failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_from_bytes(content: bytes, filename: str):
    """Extract document content from raw bytes.

    Routes to the correct DocumentExtractor method based on file type.
    Returns an ExtractedDocument.
    """
    import pandas as pd

    ftype = detect_file_type(filename, content)
    extractor = _get_document_extractor()

    if ftype == "pdf":
        return extractor.extract_text_from_pdf(content, filename=filename)

    if ftype == "docx":
        return extractor.extract_text_from_docx(content, filename=filename)

    if ftype == "pptx":
        return extractor.extract_text_from_pptx(content, filename=filename)

    if ftype == "csv":
        df = pd.read_csv(io.BytesIO(content))
        return extractor.extract_dataframe(df, filename=filename or "data.csv")

    if ftype == "excel":
        df = pd.read_excel(io.BytesIO(content))
        return extractor.extract_dataframe(df, filename=filename or "data.xlsx")

    if ftype == "image":
        ocr_text = _ocr_image_bytes(content)
        return extractor.extract_text_from_txt(ocr_text or b"", filename=filename)

    # txt / default
    return extractor.extract_text_from_txt(content, filename=filename)


# ---------------------------------------------------------------------------
# Intelligence
# ---------------------------------------------------------------------------


def run_intelligence(extracted, document_id: str) -> Dict[str, Any]:
    """Run document intelligence on *extracted* if the module is available.

    Returns a result dict.  Gracefully handles ImportError and runtime errors.
    """
    try:
        from src.intelligence.integration import process_document_intelligence  # type: ignore
        return process_document_intelligence(extracted, document_id) or {}
    except ImportError:
        logger.debug("Intelligence module not available — skipping")
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Intelligence processing failed for %s: %s", document_id, exc)
        return {}


# ---------------------------------------------------------------------------
# Chunking + Embedding
# ---------------------------------------------------------------------------

from src.embedding.chunking.section_chunker import SectionChunker  # noqa: E402
from src.embedding.model_loader import get_embedding_model  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, VectorParams, PointStruct  # noqa: E402


def chunk_and_embed(extracted, document_id: str, collection_name: str) -> int:
    """Chunk, embed, and index *extracted* into a Qdrant collection.

    Returns the number of chunks indexed.
    """
    from src.api.config import Config

    chunker = SectionChunker()
    chunks = chunker.chunk(extracted)
    if not chunks:
        return 0

    model, dim = get_embedding_model()
    texts = [c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "") for c in chunks]
    texts = [t for t in texts if t.strip()]
    if not texts:
        return 0

    embeddings = model.encode(texts)

    qdrant = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)

    try:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    except Exception:  # noqa: BLE001
        # Collection may already exist
        pass

    points = []
    for idx, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        if isinstance(chunk, dict):
            meta = chunk.get("metadata", {})
            text = chunk.get("text", "")
        else:
            meta = getattr(chunk, "metadata", {}) or {}
            text = getattr(chunk, "text", "")

        payload = {
            "text": text,
            "document_id": document_id,
            "chunk_index": idx,
            **meta,
        }
        points.append(
            PointStruct(
                id=idx,
                vector=vec.tolist() if hasattr(vec, "tolist") else list(vec),
                payload=payload,
            )
        )

    if points:
        qdrant.upsert(collection_name=collection_name, points=points)

    return len(points)


def cleanup_collection(collection_name: str) -> None:
    """Delete a temporary Qdrant collection, tolerating errors."""
    from src.api.config import Config

    try:
        qdrant = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
        qdrant.delete_collection(collection_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("cleanup_collection(%s) failed: %s", collection_name, exc)


# ---------------------------------------------------------------------------
# Retrieval + Generation
# ---------------------------------------------------------------------------

from src.execution.router import execute_request  # noqa: E402


def retrieve_and_generate(
    query: str,
    collection_name: str,
    subscription_id: str,
    profile_id: str = "standalone",
    document_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run retrieval + generation for *query* against *collection_name*.

    Builds lightweight SimpleNamespace request/ctx objects accepted by
    ``execute_request`` and returns a normalised answer dict.
    """
    request = SimpleNamespace(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        document_id=document_id,
        system_prompt=system_prompt,
        collection_name=collection_name,
    )
    ctx = SimpleNamespace(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        session_id=None,
        collection_name=collection_name,
    )

    try:
        result = execute_request(request, session_state=None, ctx=ctx, debug=debug)
        answer_data = result.answer if hasattr(result, "answer") else {}
        if isinstance(answer_data, dict):
            return answer_data
        return {"answer": str(answer_data), "sources": [], "confidence": 0.0, "grounded": False}
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieve_and_generate failed: %s", exc)
        return {
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "grounded": False,
            "context_found": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Structured prompts
# ---------------------------------------------------------------------------

_MODE_SYSTEM_PROMPTS: Dict[str, str] = {
    "qa": (
        "You are an expert document analyst. Answer the user's question accurately and concisely "
        "based solely on the content of the document provided. If the answer is not present in the "
        "document, say so clearly."
    ),
    "table": (
        "You are a data extraction specialist. Your task is to extract tabular data from the "
        "document and return it as structured JSON. Identify all tables and return them with "
        "clearly labelled columns and rows. Output must be valid JSON."
    ),
    "entities": (
        "You are an entity extraction specialist. Extract all named entities (people, "
        "organisations, dates, locations, monetary amounts, and other key facts) from the "
        "document and return them as structured JSON with entity type labels. "
        "Output must be valid JSON."
    ),
    "summary": (
        "You are a professional document summariser. Produce a concise, structured summary of "
        "the document covering the key points, main findings, and any action items. "
        "Return the summary as structured JSON with sections for key_points, findings, and "
        "action_items."
    ),
}

from src.api.standalone_templates import apply_template, get_template  # noqa: E402


def build_structured_prompt(
    mode: str,
    user_prompt: str,
    document_text: str,
    template=None,
) -> Dict[str, str]:
    """Build a system + user prompt dict for the given *mode*.

    If *template* is provided, ``apply_template`` is used; otherwise the
    built-in ``_MODE_SYSTEM_PROMPTS`` are used.
    """
    if template is not None:
        return apply_template(template, user_prompt)

    system = _MODE_SYSTEM_PROMPTS.get(mode, _MODE_SYSTEM_PROMPTS["qa"])
    return {"system_prompt": system, "user_prompt": user_prompt}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_structured_response(response_text: str, mode: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from *response_text*.

    Attempts in order:
    1. ```json ... ``` fenced blocks
    2. Direct JSON parse of the full text
    3. Extraction of the first { ... } span
    """
    if not response_text:
        return None

    # 1. Fenced JSON block
    fence_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Direct parse
    try:
        parsed = json.loads(response_text.strip())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3. Extract first { ... } span
    brace_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _build_low_confidence_reasons(answer: str, confidence: float) -> List[str]:
    """Generate human-readable reasons for low confidence."""
    reasons: List[str] = []

    if confidence < 0.3:
        reasons.append("Very low confidence score — the answer may be unreliable or unsupported.")
    elif confidence < 0.5:
        reasons.append("Low confidence score — the answer should be treated with caution.")
    elif confidence < 0.7:
        reasons.append("Moderate confidence — the answer may be partially supported by the document.")

    if not answer or len(answer.strip()) < 10:
        reasons.append("The generated answer is very short or empty, suggesting limited relevant context.")

    return reasons


# ---------------------------------------------------------------------------
# Learning signals
# ---------------------------------------------------------------------------

from src.api.learning_signals import LearningSignalStore  # noqa: E402


def _capture_learning_signal(
    query: str,
    context: str,
    answer_text: str,
    sources: List[Dict[str, Any]],
    confidence: float,
    mode: str,
    template_name: Optional[str] = None,
    request_id: str = "",
) -> None:
    """Persist a learning signal for the online fine-tune loop."""
    try:
        store = LearningSignalStore()
        metadata: Dict[str, Any] = {
            "source": "standalone_api",
            "mode": mode,
            "template": template_name,
            "request_id": request_id,
            "confidence": confidence,
        }
        if confidence >= 0.7:
            store.record_high_quality(
                query=query,
                context=context,
                answer=answer_text,
                sources=sources,
                metadata=metadata,
            )
        else:
            store.record_low_confidence(
                query=query,
                context=context,
                answer=answer_text,
                reason="; ".join(_build_low_confidence_reasons(answer_text, confidence)) or "low confidence",
                metadata=metadata,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("_capture_learning_signal failed: %s", exc)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def process_document(
    content: bytes,
    filename: str,
    prompt: str,
    mode: str = "qa",
    subscription_id: str = "standalone",
    persist: bool = False,
    template=None,
    confidence_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Full pipeline: extract → intelligence → embed → retrieve → generate.

    Returns a dict matching the ``ProcessResponse`` schema fields.
    """
    import time

    request_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    collection_name = f"standalone_{document_id.replace('-', '_')}"

    t_start = time.monotonic()
    usage: Dict[str, int] = {
        "extraction_ms": 0,
        "intelligence_ms": 0,
        "retrieval_ms": 0,
        "generation_ms": 0,
        "total_ms": 0,
    }

    # 1. Extraction
    t0 = time.monotonic()
    extracted = extract_from_bytes(content, filename)
    usage["extraction_ms"] = int((time.monotonic() - t0) * 1000)

    # 2. Intelligence (optional enrichment)
    t0 = time.monotonic()
    _intel = run_intelligence(extracted, document_id)
    usage["intelligence_ms"] = int((time.monotonic() - t0) * 1000)

    # 3. Chunk + embed
    t0 = time.monotonic()
    chunk_count = chunk_and_embed(extracted, document_id, collection_name)
    usage["retrieval_ms"] = int((time.monotonic() - t0) * 1000)

    # 4. Build prompt
    structured_prompts = build_structured_prompt(
        mode=mode,
        user_prompt=prompt,
        document_text=extracted.full_text or "",
        template=template,
    )
    system_prompt = structured_prompts.get("system_prompt")
    effective_prompt = structured_prompts.get("user_prompt", prompt)

    # 5. Retrieve + generate
    t0 = time.monotonic()
    gen_result = retrieve_and_generate(
        query=effective_prompt,
        collection_name=collection_name,
        subscription_id=subscription_id,
        profile_id="standalone",
        document_id=document_id,
        system_prompt=system_prompt,
    )
    usage["generation_ms"] = int((time.monotonic() - t0) * 1000)
    usage["total_ms"] = int((time.monotonic() - t_start) * 1000)

    answer_text: str = gen_result.get("answer", "") or ""
    sources: List[Dict[str, Any]] = gen_result.get("sources", []) or []
    confidence: float = float(gen_result.get("confidence", 0.0) or 0.0)
    grounded: bool = bool(gen_result.get("grounded", False))
    context_found: bool = bool(gen_result.get("context_found", True))

    # 6. Parse structured output for non-qa modes
    structured_output: Optional[Dict[str, Any]] = None
    if mode != "qa":
        structured_output = _parse_structured_response(answer_text, mode)

    # 7. Confidence gate
    low_confidence = confidence < confidence_threshold if confidence_threshold > 0.0 else False
    low_confidence_reasons: List[str] = []
    if low_confidence or confidence < 0.5:
        low_confidence_reasons = _build_low_confidence_reasons(answer_text, confidence)

    # 8. Learning signal
    template_name = getattr(template, "name", None) if template else None
    _capture_learning_signal(
        query=prompt,
        context=extracted.full_text[:2000] if extracted.full_text else "",
        answer_text=answer_text,
        sources=sources,
        confidence=confidence,
        mode=mode,
        template_name=template_name,
        request_id=request_id,
    )

    # 9. Cleanup
    if not persist:
        cleanup_collection(collection_name)

    return {
        "request_id": request_id,
        "status": "completed",
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence,
        "grounded": grounded,
        "low_confidence": low_confidence,
        "low_confidence_reasons": low_confidence_reasons,
        "structured_output": structured_output,
        "document_id": document_id if persist else None,
        "output_format": "json",
        "usage": usage,
    }


def query_persisted_document(
    document_id: str,
    prompt: str,
    subscription_id: str,
    mode: str = "qa",
    confidence_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Query an already-indexed document.

    Skips extraction and embedding — goes straight to retrieval against the
    pre-existing Qdrant collection for *document_id*.
    """
    import time

    request_id = str(uuid.uuid4())
    collection_name = f"standalone_{document_id.replace('-', '_')}"

    t_start = time.monotonic()

    gen_result = retrieve_and_generate(
        query=prompt,
        collection_name=collection_name,
        subscription_id=subscription_id,
        profile_id="standalone",
        document_id=document_id,
    )

    total_ms = int((time.monotonic() - t_start) * 1000)

    answer_text: str = gen_result.get("answer", "") or ""
    sources: List[Dict[str, Any]] = gen_result.get("sources", []) or []
    confidence: float = float(gen_result.get("confidence", 0.0) or 0.0)
    grounded: bool = bool(gen_result.get("grounded", False))

    structured_output: Optional[Dict[str, Any]] = None
    if mode != "qa":
        structured_output = _parse_structured_response(answer_text, mode)

    low_confidence = confidence < confidence_threshold if confidence_threshold > 0.0 else False
    low_confidence_reasons: List[str] = []
    if low_confidence:
        low_confidence_reasons = _build_low_confidence_reasons(answer_text, confidence)

    _capture_learning_signal(
        query=prompt,
        context="",
        answer_text=answer_text,
        sources=sources,
        confidence=confidence,
        mode=mode,
        request_id=request_id,
    )

    return {
        "request_id": request_id,
        "status": "completed",
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence,
        "grounded": grounded,
        "low_confidence": low_confidence,
        "low_confidence_reasons": low_confidence_reasons,
        "structured_output": structured_output,
        "document_id": document_id,
        "output_format": "json",
        "usage": {"total_ms": total_ms},
    }
