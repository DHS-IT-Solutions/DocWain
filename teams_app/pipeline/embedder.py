"""Standalone embedding pipeline for Teams app.

Replaces train_on_document with a fast, LLM-free path:
chunk -> encode -> upsert to Qdrant.

No LLM calls, no section intelligence, no context understanding.
Typical documents embed in under 5 seconds.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Heuristic doc-type detection (no LLM)
# ---------------------------------------------------------------------------

_DOC_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "resume": [
        "experience", "education", "skills", "objective", "career",
        "employment", "qualification", "certif",
    ],
    "invoice": [
        "invoice", "bill to", "ship to", "subtotal", "total due",
        "payment terms", "tax", "qty", "unit price",
    ],
    "purchase_order": [
        "purchase order", "po number", "vendor", "delivery date",
        "requisition",
    ],
    "bank_statement": [
        "account number", "opening balance", "closing balance",
        "transaction", "statement period", "debit", "credit",
    ],
    "medical": [
        "patient", "diagnosis", "medication", "prescription",
        "dosage", "lab result", "clinical",
    ],
    "legal": [
        "hereby", "whereas", "indemnif", "jurisdiction",
        "governing law", "arbitration", "covenant",
    ],
    "insurance": [
        "policy number", "premium", "coverage", "exclusion",
        "deductible", "insured", "policyholder", "claim",
    ],
}

_FILENAME_HINTS: Dict[str, str] = {
    "resume": "resume",
    "cv": "resume",
    "invoice": "invoice",
    "receipt": "invoice",
    "po_": "purchase_order",
    "purchase_order": "purchase_order",
    "bank_statement": "bank_statement",
    "statement": "bank_statement",
    "policy": "insurance",
    "contract": "legal",
    "agreement": "legal",
}


def _detect_doc_type(text: str, filename: str = "") -> str:
    """Heuristic doc-type detection from text content and filename."""
    # Filename hint first
    if filename:
        fn_lower = filename.lower()
        for hint, dtype in _FILENAME_HINTS.items():
            if hint in fn_lower:
                return dtype

    if not text:
        return "general"

    text_lower = text[:10000].lower()
    scores: Dict[str, int] = {}
    for dtype, keywords in _DOC_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score >= 2:
            scores[dtype] = score

    if scores:
        return max(scores, key=scores.get)
    return "general"


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_item(item: Any) -> str:
    """Extract clean text from a texts list item (string, dict, or object)."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "full_text", "raw_text", "canonical_text"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""
    if hasattr(item, "text"):
        val = getattr(item, "text", "")
        if isinstance(val, str) and val.strip():
            return val
    if hasattr(item, "full_text"):
        val = getattr(item, "full_text", "")
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _extract_texts_from_docs(extracted_docs: Dict[str, Any]) -> Tuple[List[str], str]:
    """Pull text chunks and full_text from fileProcessor output.

    Returns (texts_list, full_text).
    """
    texts: List[str] = []
    full_text_parts: List[str] = []

    for _doc_name, doc_content in extracted_docs.items():
        if isinstance(doc_content, str):
            if doc_content.strip():
                full_text_parts.append(doc_content)
            continue

        if isinstance(doc_content, dict):
            # Structured payload from fileProcessor
            raw_texts = doc_content.get("texts") or []
            for t in raw_texts:
                clean = _extract_text_from_item(t)
                if clean.strip():
                    texts.append(clean)

            if not texts:
                raw = (
                    doc_content.get("full_text")
                    or doc_content.get("text")
                    or doc_content.get("content")
                )
                if isinstance(raw, str) and raw.strip():
                    full_text_parts.append(raw)
            else:
                ft = doc_content.get("full_text")
                if isinstance(ft, str) and ft.strip():
                    full_text_parts.append(ft)
            continue

        # ExtractedDocument object
        if hasattr(doc_content, "full_text"):
            ft = getattr(doc_content, "full_text", "")
            if isinstance(ft, str) and ft.strip():
                full_text_parts.append(ft)

    full_text = "\n\n".join(full_text_parts)
    return texts, full_text


# ---------------------------------------------------------------------------
# Simple chunking (no LLM, fast)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^(?:chapter\b|section\b|appendix\b|\d+(?:\.\d+)+|\d+\.|[ivxlcdm]+\.)\s+.+",
    re.IGNORECASE,
)
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9\s,:\-]{4,}$")

_TARGET_CHUNK_CHARS = 900
_MIN_CHUNK_CHARS = 100
_MAX_CHUNK_CHARS = 4000


def _is_heading(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    if _HEADING_RE.match(clean):
        return True
    if len(clean.split()) <= 10 and _ALL_CAPS_RE.match(clean):
        return True
    return False


def _simple_chunk(
    text: str,
    *,
    target: int = _TARGET_CHUNK_CHARS,
    min_chars: int = _MIN_CHUNK_CHARS,
    max_chars: int = _MAX_CHUNK_CHARS,
) -> List[Dict[str, Any]]:
    """Section-aware paragraph chunking.

    Returns list of dicts with 'text', 'section_title', 'chunk_index'.
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs
    paragraphs = re.split(r"\n{2,}", text)
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Document"
    current_paras: List[str] = []

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        # Check first line for heading
        first_line = stripped.split("\n")[0].strip()
        if _is_heading(first_line) and len(first_line) < 120:
            if current_paras:
                sections.append((current_title, current_paras))
            current_title = first_line
            rest = stripped[len(first_line):].strip()
            current_paras = [rest] if rest else []
        else:
            current_paras.append(stripped)

    if current_paras:
        sections.append((current_title, current_paras))

    # Build chunks from sections
    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0

    for title, paras in sections:
        section_text = "\n\n".join(paras)
        if not section_text.strip():
            continue

        if len(section_text) <= max_chars:
            # Whole section fits in one chunk
            if len(section_text) < min_chars and chunks:
                # Merge into previous chunk
                prev = chunks[-1]
                prev["text"] = prev["text"] + "\n\n" + section_text
            else:
                chunks.append({
                    "text": section_text,
                    "section_title": title,
                    "chunk_index": chunk_idx,
                })
                chunk_idx += 1
        else:
            # Split large section at paragraph boundaries
            current_chunk: List[str] = []
            current_len = 0
            for para in paras:
                para_len = len(para)
                if current_len + para_len > target and current_chunk:
                    chunks.append({
                        "text": "\n\n".join(current_chunk),
                        "section_title": title,
                        "chunk_index": chunk_idx,
                    })
                    chunk_idx += 1
                    current_chunk = [para]
                    current_len = para_len
                else:
                    current_chunk.append(para)
                    current_len += para_len
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text) < min_chars and chunks:
                    prev = chunks[-1]
                    prev["text"] = prev["text"] + "\n\n" + chunk_text
                else:
                    chunks.append({
                        "text": chunk_text,
                        "section_title": title,
                        "chunk_index": chunk_idx,
                    })
                    chunk_idx += 1

    # Filter out chunks that are too short (< 20 chars of real content)
    chunks = [c for c in chunks if len(c["text"].strip()) >= 20]

    # Re-index
    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    return chunks


def _try_section_chunker(
    full_text: str,
    doc_tag: str,
    filename: str,
) -> Optional[Tuple[List[str], List[Dict[str, Any]]]]:
    """Try to use the production SectionChunker; return None on failure."""
    try:
        from src.embedding.chunking.section_chunker import SectionChunker, normalize_text

        chunker = SectionChunker()
        section_chunks = chunker.chunk_document(
            full_text, doc_internal_id=doc_tag, source_filename=filename,
        )
        if not section_chunks:
            return None

        texts = []
        metadata = []
        for idx, sc in enumerate(section_chunks):
            text = normalize_text(sc.text)
            if not text or len(text.strip()) < 20:
                continue
            texts.append(text)
            metadata.append({
                "section_title": sc.section_title,
                "section_path": sc.section_path,
                "page_start": sc.page_start,
                "page_end": sc.page_end,
                "chunk_index": idx,
                "sentence_complete": text.strip()[-1] in {".", "?", "!"} if text.strip() else False,
            })
        if texts:
            # Re-index after filtering
            for i, m in enumerate(metadata):
                m["chunk_index"] = i
            return texts, metadata
        return None
    except Exception as exc:
        logger.debug("SectionChunker failed, using simple chunker: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Sparse vectors (keyword search support)
# ---------------------------------------------------------------------------

def _build_sparse_vectors(texts: List[str]) -> List[Dict[str, Any]]:
    """Build hashing-based sparse vectors for keyword search."""
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
        import numpy as np

        vectorizer = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )
        matrix = vectorizer.transform(texts)
        sparse_vectors = []
        for row in matrix:
            coo = row.tocoo()
            sparse_vectors.append({
                "indices": coo.col.tolist(),
                "values": coo.data.astype(np.float32).tolist(),
            })
        return sparse_vectors
    except Exception as exc:
        logger.debug("Sparse vector generation failed: %s", exc)
        return [{"indices": [], "values": []} for _ in texts]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def embed_document(
    extracted_docs: Dict[str, Any],
    filename: str,
    doc_tag: str,
    collection_name: str,
    profile_id: str,
) -> Tuple[int, str, str]:
    """Embed a document into Qdrant. Returns (chunks_count, quality_grade, doc_type).

    Fast, LLM-free pipeline: chunk -> encode -> upsert.
    Typical documents complete in under 5 seconds.
    """
    started = time.monotonic()
    logger.info("embed_document start: filename=%s doc_tag=%s", filename, doc_tag)

    # ── Step 1: Extract texts ────────────────────────────────────────────
    pre_chunked_texts, full_text = _extract_texts_from_docs(extracted_docs)

    if not pre_chunked_texts and not full_text:
        raise ValueError(f"No text content found in extracted_docs for {filename}")

    # ── Step 2: Chunk ────────────────────────────────────────────────────
    chunk_texts: List[str] = []
    chunk_meta: List[Dict[str, Any]] = []

    logger.info("Text extraction: pre_chunked=%d, full_text_len=%d", len(pre_chunked_texts), len(full_text))

    # Always chunk from full_text for better quality — fileProcessor chunks are often too granular or too few
    text_to_chunk = full_text or "\n\n".join(pre_chunked_texts)
    if text_to_chunk:

        # Try production SectionChunker first
        section_result = await asyncio.to_thread(
            _try_section_chunker, text_to_chunk, doc_tag, filename,
        )

        # Always run simple chunker too for comparison
        raw_chunks = _simple_chunk(text_to_chunk)
        simple_texts = [c["text"] for c in raw_chunks]
        simple_meta = [
            {
                "section_title": c["section_title"],
                "section_path": c["section_title"],
                "chunk_index": c["chunk_index"],
                "sentence_complete": c["text"].strip()[-1] in {".", "?", "!"} if c["text"].strip() else False,
            }
            for c in raw_chunks
        ]

        # Use whichever produced more chunks (better coverage)
        if section_result and len(section_result[0]) >= len(simple_texts):
            chunk_texts, chunk_meta = section_result
            chunker_used = "section"
        else:
            chunk_texts = simple_texts
            chunk_meta = simple_meta
            chunker_used = "simple"

    logger.info("Chunking result: %d chunks (chunker=%s, text_len=%d)", len(chunk_texts), chunker_used, len(text_to_chunk))

    if not chunk_texts:
        raise ValueError(f"Chunking produced no chunks for {filename}")

    # ── Step 3: Detect doc type (heuristic) ──────────────────────────────
    sample_text = full_text or "\n".join(chunk_texts[:5])
    doc_type = _detect_doc_type(sample_text, filename)

    # Try production content classifier as upgrade
    try:
        from src.embedding.pipeline.content_classifier import classify_doc_domain
        classified = await asyncio.to_thread(
            classify_doc_domain, sample_text[:5000], filename, doc_type,
        )
        if classified and classified != "generic":
            doc_type = classified
    except Exception:
        pass  # heuristic result is fine

    # ── Step 4: Encode chunks ────────────────────────────────────────────
    from src.embedding.model_loader import encode_with_fallback, get_embedding_model

    # Ensure model is loaded and get dimension
    _, vector_dim = await asyncio.to_thread(get_embedding_model)

    embeddings = await asyncio.to_thread(
        encode_with_fallback,
        chunk_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Build sparse vectors for hybrid search
    sparse_vectors = await asyncio.to_thread(_build_sparse_vectors, chunk_texts)

    # ── Step 5: Build payloads and upsert ────────────────────────────────
    from src.api.vector_store import QdrantVectorStore, build_collection_name
    from src.api.pipeline_models import ChunkRecord
    from src.api.config import Config
    from qdrant_client import QdrantClient

    import numpy as np

    # Normalize embedding matrix
    if isinstance(embeddings, np.ndarray):
        embedding_list = embeddings.tolist()
    else:
        embedding_list = [
            e.tolist() if hasattr(e, "tolist") else list(e)
            for e in embeddings
        ]

    # Compute doc version hash
    doc_version_hash = hashlib.sha1(
        (full_text or "\n".join(chunk_texts)).encode("utf-8")
    ).hexdigest()[:12]

    records: List[ChunkRecord] = []
    for idx, (text, meta, dense_vec, sparse_vec) in enumerate(
        zip(chunk_texts, chunk_meta, embedding_list, sparse_vectors)
    ):
        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        section_id = meta.get("section_id") or hashlib.sha1(
            f"{doc_tag}|{meta.get('section_path', 'Document')}".encode("utf-8")
        ).hexdigest()[:12]

        chunk_id = f"chunk_{hashlib.sha256('|'.join([collection_name, profile_id, doc_tag, section_id, str(idx), chunk_hash]).encode('utf-8')).hexdigest()}"

        payload = {
            "subscription_id": collection_name,
            "profile_id": profile_id,
            "document_id": doc_tag,
            "source_file": filename,
            "section_id": section_id,
            "section_title": meta.get("section_title", "Document"),
            "section_path": meta.get("section_path", "Document"),
            "chunk_index": idx,
            "chunk_count": len(chunk_texts),
            "chunk_hash": chunk_hash,
            "chunk_id": chunk_id,
            "chunk_type": "text",
            "chunk_kind": "section_text",
            "content": text,
            "embedding_text": text,
            "text_clean": text,
            "resolution": "chunk",
            "chunk_char_len": len(text),
            "sentence_complete": meta.get("sentence_complete", False),
            "doc_type": doc_type,
            "doc_domain": doc_type,
            "doc_version_hash": doc_version_hash,
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "page_number": meta.get("page_start"),
            "prev_chunk_id": None,
            "next_chunk_id": None,
        }

        records.append(ChunkRecord(
            chunk_id=chunk_id,
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            payload=payload,
        ))

    # Set prev/next chain
    for i, rec in enumerate(records):
        if i > 0:
            rec.payload["prev_chunk_id"] = records[i - 1].chunk_id
        if i < len(records) - 1:
            rec.payload["next_chunk_id"] = records[i + 1].chunk_id

    # Upsert to Qdrant
    client = QdrantClient(
        url=Config.Qdrant.URL,
        api_key=Config.Qdrant.API,
        timeout=120,
    )
    vector_store = QdrantVectorStore(client)
    vector_store.ensure_collection(collection_name, vector_dim)

    upserted = await asyncio.to_thread(
        vector_store.upsert_records, collection_name, records,
    )

    elapsed = time.monotonic() - started
    chunks_count = upserted

    # Quality grade based on chunk count
    quality_grade = "A" if chunks_count > 10 else "B" if chunks_count > 3 else "C"

    logger.info(
        "embed_document done: filename=%s chunks=%d grade=%s doc_type=%s elapsed=%.2fs",
        filename, chunks_count, quality_grade, doc_type, elapsed,
    )

    return chunks_count, quality_grade, doc_type
