"""
Fast path execution for SIMPLE queries.

Skips: Planner LLM call, RepairLoop, KG enrichment.
Does: embed query -> search Qdrant -> rerank top-3 -> LLM reason.
Target: sub-2s response time.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Generator, List, Optional

from src.api.vector_store import build_qdrant_filter
from src.retrieval.reranker import rerank_chunks
from src.retrieval.retriever import EvidenceChunk

logger = logging.getLogger(__name__)

_FAST_PATH_TOP_K = 10  # Qdrant retrieval limit
_FAST_PATH_RERANK_K = 3  # Rerank to top-3
_MAX_CONTEXT_CHARS = 4000  # Cap context window for fast responses


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are DocWain, a document intelligence assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Be concise and accurate. If the context does not contain enough "
    "information, say so clearly. Do not make up information."
)


def _build_fast_context(chunks: List[Dict[str, Any]], max_chunks: int = 3) -> str:
    """Build a minimal context string from the top chunks.

    Parameters
    ----------
    chunks:
        List of dicts with at least a ``text`` key.
    max_chunks:
        Maximum number of chunks to include.

    Returns
    -------
    Concatenated context string.
    """
    parts: list[str] = []
    total_chars = 0
    for chunk in chunks[:max_chunks]:
        text = chunk.get("text", "")
        if not text:
            continue
        if total_chars + len(text) > _MAX_CONTEXT_CHARS:
            remaining = _MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:
                parts.append(text[:remaining])
            break
        parts.append(text)
        total_chars += len(text)
    return "\n\n---\n\n".join(parts)


def _build_prompt(query: str, context: str) -> str:
    """Build user prompt with context and query."""
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def _embed_query(embedder: Any, query: str) -> list:
    """Embed query using the app's embedding model."""
    if hasattr(embedder, "encode"):
        vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=False)
        # Normalize vector output to flat list of floats
        return _vector_to_list(vec)
    raise RuntimeError("Embedder missing encode method")


def _vector_to_list(vec: Any) -> list:
    """Convert embedding output to a flat list of floats."""
    try:
        import torch
    except ImportError:
        torch = None
    try:
        import numpy as np
    except ImportError:
        np = None

    if torch is not None and torch.is_tensor(vec):
        return [float(v) for v in vec.detach().cpu().flatten().tolist()]
    if np is not None and isinstance(vec, __import__("numpy").ndarray):
        return [float(v) for v in vec.reshape(-1).tolist()]
    if isinstance(vec, list) and vec:
        first = vec[0]
        if torch is not None and torch.is_tensor(first):
            return [float(v) for v in first.detach().cpu().flatten().tolist()]
        if np is not None and isinstance(first, __import__("numpy").ndarray):
            return [float(v) for v in first.reshape(-1).tolist()]
        if isinstance(first, list):
            return [float(v) for v in first]
    if isinstance(vec, list):
        return [float(v) for v in vec]
    raise ValueError("Embedder returned unsupported vector type")


def _qdrant_search(
    qdrant_client: Any,
    vector: list,
    subscription_id: str,
    profile_id: str,
    top_k: int = _FAST_PATH_TOP_K,
) -> List[Any]:
    """Run a single-collection Qdrant vector search."""
    collection = str(subscription_id)
    q_filter = build_qdrant_filter(
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
    )
    results = qdrant_client.query_points(
        collection_name=collection,
        query=vector,
        using="content_vector",
        query_filter=q_filter,
        limit=int(top_k),
        with_payload=True,
        with_vectors=False,
    )
    return getattr(results, "points", None) or []


def _points_to_evidence(points: List[Any]) -> List[EvidenceChunk]:
    """Convert Qdrant points to EvidenceChunk objects."""
    chunks = []
    for point in points:
        if point is None:
            continue
        payload = getattr(point, "payload", None) or {}
        text = (
            payload.get("canonical_text")
            or payload.get("content")
            or payload.get("text")
            or payload.get("embedding_text")
            or ""
        )
        chunks.append(EvidenceChunk(
            text=text,
            source_name=payload.get("source_name") or "document",
            document_id=str(payload.get("document_id", "")),
            profile_id=str(payload.get("profile_id", "")),
            section=str(payload.get("section_id", "")),
            page_start=int(payload.get("page", 0) or 0),
            page_end=int(payload.get("page", 0) or 0),
            score=float(getattr(point, "score", 0.0) or 0.0),
            chunk_id=str(getattr(point, "id", "")),
            chunk_type=str(payload.get("chunk_kind", "text")),
        ))
    return chunks


def _chunks_to_sources(chunks: List[EvidenceChunk]) -> List[Dict[str, Any]]:
    """Build sources list from evidence chunks."""
    sources = []
    for c in chunks:
        sources.append({
            "document_id": c.document_id,
            "source_name": c.source_name,
            "page": c.page_start,
            "score": round(c.score, 4),
            "snippet": c.text[:160] if c.text else "",
        })
    return sources


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_fast_path(
    query: str,
    profile_id: str,
    subscription_id: str,
    app_state: Any,
) -> Dict[str, Any]:
    """Fast path for SIMPLE queries. No planner, no repair loop, no KG.

    Steps:
    1. Embed query
    2. Search Qdrant (single collection, top-10)
    3. Rerank to top-3
    4. Build minimal prompt
    5. Generate response via LLM

    Returns same format as CoreAgent.handle().
    """
    t0 = time.monotonic()

    # 1. Embed
    vector = _embed_query(app_state.embedding_model, query)

    # 2. Search Qdrant
    points = _qdrant_search(
        app_state.qdrant_client, vector,
        subscription_id, profile_id,
        top_k=_FAST_PATH_TOP_K,
    )
    evidence = _points_to_evidence(points)

    # 3. Rerank to top-3
    reranked = rerank_chunks(
        query, evidence,
        top_k=_FAST_PATH_RERANK_K,
        cross_encoder=getattr(app_state, "reranker", None),
    )

    # 4. Build prompt
    chunk_dicts = [{"text": c.text} for c in reranked]
    context = _build_fast_context(chunk_dicts, max_chunks=_FAST_PATH_RERANK_K)
    prompt = _build_prompt(query, context)

    # 5. Generate response
    llm = app_state.llm_gateway
    answer_text = llm.generate(
        prompt,
        system=_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=1024,
    )

    elapsed = time.monotonic() - t0
    sources = _chunks_to_sources(reranked)
    context_found = len(reranked) > 0

    logger.info(
        "[FAST_PATH] query=%r chunks=%d elapsed=%.2fs",
        query[:80], len(reranked), elapsed,
    )

    return {
        "response": answer_text,
        "answer": answer_text,
        "sources": sources,
        "query_type": "SIMPLE",
        "fast_path": True,
        "grounded": context_found,
        "context_found": context_found,
        "metadata": {
            "fast_path": True,
            "query_type": "SIMPLE",
            "elapsed_s": round(elapsed, 3),
            "chunks_used": len(reranked),
        },
    }


def execute_fast_path_stream(
    query: str,
    profile_id: str,
    subscription_id: str,
    app_state: Any,
) -> Generator[str, None, None]:
    """Streaming version of fast path. Yields response chunks."""
    t0 = time.monotonic()

    # 1. Embed
    vector = _embed_query(app_state.embedding_model, query)

    # 2. Search Qdrant
    points = _qdrant_search(
        app_state.qdrant_client, vector,
        subscription_id, profile_id,
        top_k=_FAST_PATH_TOP_K,
    )
    evidence = _points_to_evidence(points)

    # 3. Rerank to top-3
    reranked = rerank_chunks(
        query, evidence,
        top_k=_FAST_PATH_RERANK_K,
        cross_encoder=getattr(app_state, "reranker", None),
    )

    # 4. Build prompt
    chunk_dicts = [{"text": c.text} for c in reranked]
    context = _build_fast_context(chunk_dicts, max_chunks=_FAST_PATH_RERANK_K)
    prompt = _build_prompt(query, context)

    elapsed_prep = time.monotonic() - t0
    logger.info(
        "[FAST_PATH_STREAM] query=%r chunks=%d prep=%.2fs",
        query[:80], len(reranked), elapsed_prep,
    )

    # 5. Stream response
    llm = app_state.llm_gateway
    yield from llm.generate_stream(
        prompt,
        system=_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=1024,
    )
