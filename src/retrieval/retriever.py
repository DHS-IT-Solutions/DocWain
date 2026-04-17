"""Unified retriever with hybrid dense + keyword fallback search."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.api.vector_store import build_collection_name

logger = logging.getLogger(__name__)


@dataclass
class EvidenceChunk:
    """A single piece of retrieved evidence."""

    text: str
    source_name: str
    document_id: str
    profile_id: str
    section: str
    page_start: int
    page_end: int
    score: float
    chunk_id: str
    chunk_type: str = "text"
    profile_name: str = ""


@dataclass
class RetrievalResult:
    """Aggregated retrieval output."""

    chunks: List[EvidenceChunk]
    profiles_searched: List[str]
    total_found: int


class UnifiedRetriever:
    """Single retriever combining dense vector search with keyword fallback."""

    # Minimum number of high-quality results before triggering keyword fallback
    _DENSE_MIN = 3
    # Score threshold for "high quality" dense result
    _HIGH_QUALITY_THRESHOLD = 0.5

    # Negative cache entries expire after this many seconds so that
    # collections created by the embedding pipeline become visible
    # without requiring a server restart.
    _NEGATIVE_CACHE_TTL = 30

    # RRF constant for reciprocal-rank fusion across retrievers (dense + PPR).
    # Standard value from the original RRF paper.
    _RRF_K = 60
    # How many chunks to pull from PPR per profile before fusion.
    _PPR_TOP_K = 30

    def __init__(self, qdrant_client, embedder, *, enable_ppr: Optional[bool] = None):
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        # Maps collection_name → (exists: bool, checked_at: float)
        self._collection_exists_cache: dict[str, tuple[bool, float]] = {}
        # PPR is opt-in: explicit flag overrides, otherwise falls back to
        # Config.Retrieval.ENABLE_PPR (default False) so existing deployments
        # aren't affected until the operator flips it on.
        if enable_ppr is None:
            try:
                from src.api.config import Config
                enable_ppr = bool(getattr(Config.Retrieval, "ENABLE_PPR", False))
            except Exception:
                enable_ppr = False
        self.enable_ppr = enable_ppr
        self._ppr_retriever = None  # lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        subscription_id: str,
        profile_ids: List[str],
        *,
        document_ids: Optional[List[str]] = None,
        top_k: int = 50,
        correlation_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Search across one or more profiles and return merged results."""
        if not subscription_id or not str(subscription_id).strip():
            raise ValueError("subscription_id is required for retrieval")

        collection_name = build_collection_name(subscription_id)

        # Guard: verify collection exists before querying Qdrant.
        # Positive results are cached permanently; negative results expire
        # after _NEGATIVE_CACHE_TTL seconds so newly-created collections
        # (from the embedding pipeline) become visible without a restart.
        now = time.monotonic()
        cached = self._collection_exists_cache.get(collection_name)
        need_check = (
            cached is None
            or (not cached[0] and (now - cached[1]) > self._NEGATIVE_CACHE_TTL)
        )
        if need_check:
            try:
                exists = self.qdrant_client.collection_exists(collection_name)
                self._collection_exists_cache[collection_name] = (exists, now)
            except Exception:
                logger.warning(
                    "Could not verify collection existence: %s", collection_name,
                )
                self._collection_exists_cache[collection_name] = (False, now)

        if not self._collection_exists_cache[collection_name][0]:
            logger.warning(
                "Collection %s does not exist — returning empty results for subscription=%s",
                collection_name, subscription_id,
            )
            return RetrievalResult(
                chunks=[],
                profiles_searched=list(profile_ids),
                total_found=0,
            )

        query_vector = self.embedder.encode([query])[0]

        all_chunks: List[EvidenceChunk] = []
        per_profile = max(1, top_k // max(len(profile_ids), 1))

        for pid in profile_ids:
            dense_chunks = self._search_profile(
                collection_name, query, query_vector, subscription_id, pid,
                document_ids=document_ids,
                top_k=per_profile,
                correlation_id=correlation_id,
            )
            if self.enable_ppr:
                ppr_chunks = self._ppr_search_profile(
                    collection_name, query, subscription_id, pid,
                    document_ids=document_ids,
                    top_k=self._PPR_TOP_K,
                )
                fused = self._rrf_fuse(dense_chunks, ppr_chunks, top_k=per_profile)
                all_chunks.extend(fused)
            else:
                all_chunks.extend(dense_chunks)

        # Fill missing documents: fetch one chunk from each doc not yet in results
        all_chunks = self._fill_missing_documents(
            collection_name, all_chunks, subscription_id, profile_ids,
        )

        # Ensure document diversity: every document gets at least one chunk
        all_chunks = self._ensure_document_diversity(all_chunks, top_k)

        return RetrievalResult(
            chunks=all_chunks,
            profiles_searched=list(profile_ids),
            total_found=len(all_chunks),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        subscription_id: str,
        profile_id: str,
        document_ids: Optional[List[str]] = None,
        *,
        chunks_only: bool = False,
    ) -> Filter:
        """Build Qdrant filter scoped to subscription + single profile."""
        must = [
            FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
            FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
        ]
        if document_ids:
            if len(document_ids) == 1:
                must.append(FieldCondition(key="document_id", match=MatchValue(value=document_ids[0])))
            else:
                must.append(FieldCondition(key="document_id", match=MatchAny(any=document_ids)))
        if chunks_only:
            # Without this, doc_index points (which are whole-document summaries
            # optimised for query similarity) tend to dominate the top-K and
            # push real chunks out before the post-filter can see them. They
            # score higher than chunk-level embeddings by design.
            must.append(FieldCondition(key="resolution", match=MatchValue(value="chunk")))
        return Filter(must=must)

    def _search_profile(
        self,
        collection_name: str,
        query: str,
        query_vector: list,
        subscription_id: str,
        profile_id: str,
        *,
        document_ids: Optional[List[str]] = None,
        top_k: int = 30,
        correlation_id: Optional[str] = None,
    ) -> List[EvidenceChunk]:
        """Dense search for a single profile, with keyword fallback."""
        qfilter = self._build_filter(subscription_id, profile_id, document_ids,
                                     chunks_only=True)

        try:
            result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="content_vector",
                query_filter=qfilter,
                limit=top_k,
                with_payload=True,
            )
            points = result.points if hasattr(result, "points") else []
        except Exception:
            logger.exception(
                "Dense search failed collection=%s profile=%s cid=%s",
                collection_name, profile_id, correlation_id,
            )
            points = []

        # Filter out doc_index/doc_intelligence points — those are fetched separately
        points = [
            pt for pt in points
            if (pt.payload or {}).get("resolution", "chunk") not in ("doc_index", "doc_intelligence")
        ]
        chunks = [self._point_to_chunk(pt, profile_id) for pt in points]

        # Keyword fallback when dense returns too few high-quality hits
        high_quality = [c for c in chunks if c.score >= self._HIGH_QUALITY_THRESHOLD]
        if len(high_quality) < self._DENSE_MIN:
            fallback = self._keyword_fallback(
                collection_name, query, qfilter, top_k, existing_ids={c.chunk_id for c in chunks},
            )
            chunks.extend(fallback)

        return chunks

    def _fill_missing_documents(
        self,
        collection_name: str,
        existing_chunks: List[EvidenceChunk],
        subscription_id: str,
        profile_ids: List[str],
    ) -> List[EvidenceChunk]:
        """Scroll the profile to find documents not yet represented in results."""
        covered = set()
        for c in existing_chunks:
            doc_id = c.document_id or (c.meta or {}).get("document_id", "")
            if doc_id:
                covered.add(doc_id)

        logger.info("_fill_missing: %d chunks covering %d docs initially", len(existing_chunks), len(covered))

        added = 0
        for pid in profile_ids:
            qfilter = self._build_filter(subscription_id, pid)
            try:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=qfilter,
                    limit=500,
                    with_payload=True,
                )
                records = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
            except Exception as exc:
                logger.warning("_fill_missing scroll failed: %s", exc)
                continue

            logger.info("_fill_missing: scrolled %d records for profile %s", len(records or []), pid)

            for record in (records or []):
                payload = record.payload or {}
                # Try multiple field names for document_id
                doc_id = (
                    payload.get("document_id")
                    or payload.get("doc_id")
                    or (payload.get("chunk", {}) or {}).get("document_id", "")
                    or ""
                )
                if doc_id and doc_id not in covered:
                    chunk = self._point_to_chunk(record, pid)
                    chunk.score = 0.1
                    existing_chunks.append(chunk)
                    covered.add(doc_id)
                    added += 1

        logger.info("_fill_missing: added %d missing docs, now %d total docs", added, len(covered))
        return existing_chunks

    # Fraction of top_k reserved for score-sorted dense/PPR hits before any
    # diversity round-robin kicks in. With ratio=0.6 and top_k=10, the first
    # 6 slots respect score order strictly; diversity fill gets the bottom 4.
    _SCORE_PRIORITY_RATIO = 0.6
    # Score-0.1 fill chunks (from _fill_missing_documents) are demoted below
    # any real retrieval hit; anything at/above this threshold is a real hit.
    _REAL_HIT_SCORE = 0.15

    @classmethod
    def _ensure_document_diversity(
        cls,
        chunks: List[EvidenceChunk],
        top_k: int,
    ) -> List[EvidenceChunk]:
        """Respect score order first, fill doc diversity only in the tail.

        The previous policy round-robined "best chunk per document" into the
        head, which was the right call for generic queries but destroyed
        ranking for specific queries: a score-0.1 fill chunk from an
        unrelated doc could land above the dense-top-scored chunks. Now the
        head of the result list is strictly score-ordered (real hits stay on
        top), and diversity fill only reaches the tail slots.
        """
        if not chunks:
            return chunks

        score_sorted = sorted(chunks, key=lambda x: x.score, reverse=True)
        head_limit = max(1, int(top_k * cls._SCORE_PRIORITY_RATIO))

        head: List[EvidenceChunk] = []
        seen = set()
        for c in score_sorted:
            if len(head) >= head_limit:
                break
            head.append(c)
            seen.add(id(c))

        # Tail: prefer chunks from documents not yet represented, but only
        # among "real" hits; score-0.1 fill chunks remain last-resort.
        head_docs = {c.document_id for c in head}
        tail_candidates = [c for c in score_sorted if id(c) not in seen]
        new_doc_real = [c for c in tail_candidates
                        if c.document_id not in head_docs and c.score >= cls._REAL_HIT_SCORE]
        new_doc_fill = [c for c in tail_candidates
                        if c.document_id not in head_docs and c.score < cls._REAL_HIT_SCORE]
        same_doc = [c for c in tail_candidates if c.document_id in head_docs]

        tail_order = new_doc_real + same_doc + new_doc_fill
        result = head + tail_order
        return result[:top_k]

    def _keyword_fallback(
        self,
        collection_name: str,
        query: str,
        qfilter: Filter,
        top_k: int,
        existing_ids: set,
    ) -> List[EvidenceChunk]:
        """Scroll-based keyword fallback for low-confidence dense results."""
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=qfilter,
                limit=top_k,
                with_payload=True,
            )
            records = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
        except Exception:
            logger.exception("Keyword fallback scroll failed collection=%s", collection_name)
            return []

        query_tokens = set(query.lower().split())
        fallback_chunks: List[EvidenceChunk] = []

        for record in records:
            payload = record.payload or {}
            text = payload.get("canonical_text") or payload.get("embedding_text") or ""
            chunk_id = (payload.get("chunk") or {}).get("id", "")
            if chunk_id in existing_ids:
                continue

            # Simple keyword overlap score
            text_tokens = set(text.lower().split())
            overlap = query_tokens & text_tokens
            if not overlap:
                continue

            precision = len(overlap) / len(query_tokens) if query_tokens else 0
            recall = len(overlap) / len(text_tokens) if text_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

            chunk = self._point_to_chunk(record, payload.get("profile_id", ""))
            chunk.score = f1 * 0.5  # Scale keyword scores below dense scores
            fallback_chunks.append(chunk)

        fallback_chunks.sort(key=lambda c: c.score, reverse=True)
        return fallback_chunks[:top_k]

    def _ppr_search_profile(
        self,
        collection_name: str,
        query: str,
        subscription_id: str,
        profile_id: str,
        *,
        document_ids: Optional[List[str]] = None,
        top_k: int = 30,
    ) -> List[EvidenceChunk]:
        """Run PPR over the KG and hydrate the chunk IDs into EvidenceChunks.

        PPR returns ``[{chunk_id, document_id, score}]`` keyed against the
        Neo4j ``Chunk`` nodes written by the section-intelligence pipeline.
        We fetch the matching Qdrant points in one scroll-with-filter to get
        the text and provenance needed for downstream reasoning, preserving
        the PPR score as the EvidenceChunk score.
        """
        if self._ppr_retriever is None:
            try:
                from src.retrieval.ppr_retriever import PPRRetriever
                self._ppr_retriever = PPRRetriever()
            except Exception as exc:  # noqa: BLE001
                logger.warning("PPR init failed, skipping: %s", exc)
                return []

        try:
            ppr_hits = self._ppr_retriever.retrieve(
                query, subscription_id, profile_id, top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("PPR retrieve failed: %s", exc)
            return []

        if not ppr_hits:
            return []

        # Optional document_ids filter — PPR doesn't know to scope itself
        if document_ids:
            allowed = set(document_ids)
            ppr_hits = [h for h in ppr_hits if h.get("document_id") in allowed]
            if not ppr_hits:
                return []

        # Hydrate via Qdrant scroll filtered on chunk_id ∈ [...]
        chunk_ids = [str(h["chunk_id"]) for h in ppr_hits if h.get("chunk_id")]
        if not chunk_ids:
            return []

        score_by_id = {str(h["chunk_id"]): float(h.get("score") or 0.0) for h in ppr_hits}

        qfilter = Filter(must=[
            FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
            FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
            FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids)),
        ])
        try:
            records = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=qfilter,
                limit=max(len(chunk_ids), top_k) or 1,
                with_payload=True,
            )
            records = records[0] if isinstance(records, tuple) else records
        except Exception as exc:  # noqa: BLE001
            logger.warning("PPR hydration scroll failed: %s", exc)
            return []

        chunks: List[EvidenceChunk] = []
        for rec in records or []:
            chunk = self._point_to_chunk(rec, profile_id)
            if chunk.chunk_id:
                # Carry the PPR score instead of the (zero) scroll score so
                # downstream rank-based fusion sees the intended ordering.
                chunk.score = score_by_id.get(chunk.chunk_id, chunk.score or 0.0)
                chunks.append(chunk)

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]

    @classmethod
    def _rrf_fuse(
        cls,
        dense: List[EvidenceChunk],
        ppr: List[EvidenceChunk],
        *,
        top_k: int,
    ) -> List[EvidenceChunk]:
        """Reciprocal-rank fusion of dense and PPR chunk lists.

        A chunk appearing in both lists accumulates 1/(k+rank) from each
        source. The fused score replaces the raw score so downstream rankers
        see a unified magnitude. Retains the richer-populated EvidenceChunk
        (dense usually has the score from Qdrant; PPR hydration fills text
        from the same Qdrant points so either works — dense wins the tiebreak
        to preserve the original vector score on the object).
        """
        if not ppr:
            return dense[:top_k]
        if not dense:
            return ppr[:top_k]

        k = cls._RRF_K
        rrf_scores: dict[str, float] = {}
        chunk_by_id: dict[str, EvidenceChunk] = {}

        for rank, c in enumerate(dense, start=1):
            if not c.chunk_id:
                continue
            rrf_scores[c.chunk_id] = rrf_scores.get(c.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_by_id.setdefault(c.chunk_id, c)

        for rank, c in enumerate(ppr, start=1):
            if not c.chunk_id:
                continue
            rrf_scores[c.chunk_id] = rrf_scores.get(c.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_by_id.setdefault(c.chunk_id, c)

        ordered = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        fused: List[EvidenceChunk] = []
        for cid, score in ordered[:top_k]:
            ch = chunk_by_id.get(cid)
            if ch is None:
                continue
            ch.score = float(score)
            fused.append(ch)
        return fused

    @staticmethod
    def _point_to_chunk(point, profile_id: str) -> EvidenceChunk:
        """Convert a Qdrant point/record to an EvidenceChunk."""
        payload = point.payload or {}
        chunk_meta = payload.get("chunk") or {}
        section_meta = payload.get("section") or {}
        provenance = payload.get("provenance") or {}

        text = payload.get("canonical_text") or payload.get("embedding_text") or ""
        source_name = (
            payload.get("source_name")
            or provenance.get("source_file")
            or payload.get("source_file")
            or ""
        )

        # The ingest pipeline writes chunk_id as a flat payload field;
        # older payloads used a nested {"chunk": {"id": ...}} block. Check
        # both so fusion keys line up with whatever the live writer emitted.
        chunk_id = (
            payload.get("chunk_id")
            or chunk_meta.get("id")
            or ""
        )
        chunk_type = chunk_meta.get("type") or payload.get("chunk_kind") or "text"
        section_title = section_meta.get("title") or payload.get("section_title") or ""
        page_start = provenance.get("page_start") or payload.get("page") or 0

        return EvidenceChunk(
            text=text,
            source_name=source_name,
            document_id=payload.get("document_id", ""),
            profile_id=payload.get("profile_id", profile_id),
            section=section_title,
            page_start=page_start,
            page_end=provenance.get("page_end", page_start),
            score=getattr(point, "score", 0.0) or 0.0,
            chunk_id=chunk_id,
            chunk_type=chunk_type,
        )
