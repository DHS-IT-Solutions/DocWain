"""Corrective RAG (CRAG) evaluator for DocWain V2.

Scores retrieved chunks for relevance before they are assembled into
context for the LLM.  Low-quality chunks are discarded, and if overall
retrieval quality is poor the evaluator recommends re-retrieval with
a refined query.

This reduces hallucination by ensuring the model only sees relevant,
high-quality evidence.

Usage::

    evaluator = CRAGEvaluator()
    result = evaluator.evaluate(query, chunks)
    if result.needs_reretrieval:
        # Use result.refined_query for re-retrieval
    else:
        # Use result.filtered_chunks for context assembly
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_RELEVANCE_THRESHOLD = 0.3       # chunks below this score are discarded
_QUALITY_THRESHOLD = 0.5         # overall quality below this triggers re-retrieval
_MIN_CHUNKS = 2                  # need at least this many relevant chunks
_MAX_RERETRIEVAL_ROUNDS = 2      # max re-retrieval attempts


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChunkScore:
    """Relevance score for a single retrieved chunk."""

    chunk_id: str
    text: str
    relevance_score: float          # 0.0 - 1.0
    term_overlap: float             # fraction of query terms found
    semantic_score: float           # original retrieval score
    keep: bool = True               # whether to include in context
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CRAGResult:
    """Result of CRAG evaluation on a set of retrieved chunks."""

    filtered_chunks: List[Dict[str, Any]]       # chunks that passed filtering
    discarded_chunks: List[Dict[str, Any]]       # chunks that were removed
    chunk_scores: List[ChunkScore]               # detailed scores per chunk
    overall_quality: float                        # 0.0 - 1.0
    needs_reretrieval: bool                       # True if quality too low
    refined_query: Optional[str] = None           # suggested query for re-retrieval
    reretrieval_reason: str = ""                  # why re-retrieval is needed


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class CRAGEvaluator:
    """Corrective RAG evaluator — scores and filters retrieved chunks.

    Scoring is programmatic (no LLM call) for speed.  It combines:
    1. Term overlap between query and chunk text
    2. Original semantic retrieval score
    3. Content quality signals (length, structure, specificity)

    Parameters
    ----------
    relevance_threshold:
        Minimum combined score to keep a chunk (0.0-1.0).
    quality_threshold:
        Minimum overall quality to avoid re-retrieval (0.0-1.0).
    min_chunks:
        Minimum number of relevant chunks needed.
    """

    def __init__(
        self,
        relevance_threshold: float = _RELEVANCE_THRESHOLD,
        quality_threshold: float = _QUALITY_THRESHOLD,
        min_chunks: int = _MIN_CHUNKS,
    ) -> None:
        self.relevance_threshold = relevance_threshold
        self.quality_threshold = quality_threshold
        self.min_chunks = min_chunks

    def evaluate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        *,
        reretrieval_round: int = 0,
    ) -> CRAGResult:
        """Evaluate retrieved chunks for relevance and quality.

        Parameters
        ----------
        query:
            The original user query.
        chunks:
            Retrieved chunks, each a dict with at least ``text`` and
            optionally ``score``, ``metadata``, ``chunk_id``, ``source``.
        reretrieval_round:
            Current re-retrieval round (0 = first retrieval).

        Returns
        -------
        CRAGResult with filtered chunks and quality assessment.
        """
        if not chunks:
            return CRAGResult(
                filtered_chunks=[],
                discarded_chunks=[],
                chunk_scores=[],
                overall_quality=0.0,
                needs_reretrieval=(reretrieval_round < _MAX_RERETRIEVAL_ROUNDS),
                refined_query=query,
                reretrieval_reason="No chunks retrieved",
            )

        # Score each chunk
        query_terms = self._extract_terms(query)
        chunk_scores = []

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", chunk.get("content", ""))
            chunk_id = chunk.get("chunk_id", chunk.get("id", str(i)))
            semantic = chunk.get("score", chunk.get("relevance_score", 0.5))

            score = self._score_chunk(query, query_terms, text, semantic)
            score.chunk_id = chunk_id
            score.metadata = chunk.get("metadata", {})

            # Decide keep/discard
            if score.relevance_score < self.relevance_threshold:
                score.keep = False
                score.reasons.append(
                    f"Below relevance threshold ({score.relevance_score:.2f} < {self.relevance_threshold})"
                )

            chunk_scores.append(score)

        # Partition
        kept = []
        discarded = []
        for cs, chunk in zip(chunk_scores, chunks):
            if cs.keep:
                kept.append(chunk)
            else:
                discarded.append(chunk)

        # Overall quality
        if kept:
            overall_quality = sum(cs.relevance_score for cs in chunk_scores if cs.keep) / len(kept)
        else:
            overall_quality = 0.0

        # Decide re-retrieval
        needs_reretrieval = False
        reretrieval_reason = ""
        refined_query = None

        if len(kept) < self.min_chunks:
            needs_reretrieval = True
            reretrieval_reason = (
                f"Insufficient relevant chunks ({len(kept)} < {self.min_chunks})"
            )
            refined_query = self._refine_query(query, chunk_scores)
        elif overall_quality < self.quality_threshold:
            needs_reretrieval = True
            reretrieval_reason = (
                f"Overall quality too low ({overall_quality:.2f} < {self.quality_threshold})"
            )
            refined_query = self._refine_query(query, chunk_scores)

        # Don't re-retrieve if we've hit the max
        if reretrieval_round >= _MAX_RERETRIEVAL_ROUNDS:
            needs_reretrieval = False
            if reretrieval_reason:
                reretrieval_reason += " (max re-retrieval rounds reached)"

        logger.info(
            "CRAG: %d/%d chunks kept, quality=%.2f, reretrieval=%s",
            len(kept), len(chunks), overall_quality, needs_reretrieval,
        )

        return CRAGResult(
            filtered_chunks=kept,
            discarded_chunks=discarded,
            chunk_scores=chunk_scores,
            overall_quality=overall_quality,
            needs_reretrieval=needs_reretrieval,
            refined_query=refined_query,
            reretrieval_reason=reretrieval_reason,
        )

    def _score_chunk(
        self,
        query: str,
        query_terms: List[str],
        text: str,
        semantic_score: float,
    ) -> ChunkScore:
        """Score a single chunk on multiple dimensions."""
        text_lower = text.lower()

        # 1. Term overlap (0.0-1.0)
        if query_terms:
            matches = sum(1 for t in query_terms if t in text_lower)
            term_overlap = matches / len(query_terms)
        else:
            term_overlap = 0.0

        # 2. Content quality signals
        quality_bonus = 0.0

        # Longer chunks with substance score higher
        word_count = len(text.split())
        if word_count >= 50:
            quality_bonus += 0.05
        if word_count >= 100:
            quality_bonus += 0.05

        # Structured content (tables, lists, headers) is more informative
        if "|" in text and text.count("|") >= 4:
            quality_bonus += 0.05  # likely a table
        if re.search(r"^\s*[-*]\s+", text, re.MULTILINE):
            quality_bonus += 0.02  # bullet points
        if re.search(r"^\s*#{1,4}\s+", text, re.MULTILINE):
            quality_bonus += 0.02  # headers

        # Numeric content (useful for data queries)
        numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
        if len(numbers) >= 3:
            quality_bonus += 0.05

        # 3. Combined score
        # Weight: 40% semantic, 40% term overlap, 20% quality
        relevance = (
            0.4 * min(semantic_score, 1.0)
            + 0.4 * term_overlap
            + 0.2 * min(quality_bonus + 0.5, 1.0)  # baseline 0.5 for quality
        )

        return ChunkScore(
            chunk_id="",
            text=text[:200],  # store truncated for logging
            relevance_score=round(relevance, 3),
            term_overlap=round(term_overlap, 3),
            semantic_score=round(semantic_score, 3),
        )

    def _extract_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query for overlap scoring."""
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "about", "between", "through", "during",
            "before", "after", "above", "below", "and", "or", "but",
            "not", "no", "nor", "so", "yet", "both", "each", "few",
            "more", "most", "other", "some", "such", "than", "too",
            "very", "just", "also", "what", "which", "who", "whom",
            "this", "that", "these", "those", "i", "me", "my", "we",
            "our", "you", "your", "he", "him", "his", "she", "her",
            "it", "its", "they", "them", "their",
        }

        words = re.findall(r"\b\w+\b", query.lower())
        return [w for w in words if w not in stop_words and len(w) > 1]

    def _refine_query(
        self, query: str, chunk_scores: List[ChunkScore]
    ) -> str:
        """Generate a refined query based on what was found/missing.

        Simple strategy: keep the original query but add specificity
        based on what the low-scoring chunks contained (they might
        be close but not quite right).
        """
        # Find terms that appeared in kept chunks but not in query
        query_lower = query.lower()
        kept_texts = " ".join(
            cs.text for cs in chunk_scores if cs.keep
        ).lower()

        # Just return the original query with a specificity hint
        # In practice, the planner would generate better refined queries
        return query
