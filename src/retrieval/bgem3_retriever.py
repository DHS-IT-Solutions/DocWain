"""BGE-M3 hybrid dense+sparse retriever for DocWain V2.

Replaces BGE-large with BGE-M3, providing combined dense (1024-dim)
and sparse (BM25-style) retrieval in a single model.  This enables
Qdrant hybrid search for better accuracy on both semantic and
keyword-exact queries.

Usage::

    retriever = BGEM3Retriever()
    results = retriever.search(query="contract renewal terms", top_k=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default model — BGE-M3 supports 8192 tokens, 1024-dim dense
_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
_DEFAULT_DENSE_DIM = 1024


@dataclass
class RetrievalResult:
    """A single retrieval result with scores and metadata."""

    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    page: Optional[int] = None
    chunk_id: str = ""


class BGEM3Retriever:
    """Hybrid dense+sparse retriever using BGE-M3.

    Encodes queries into both dense vectors (1024-dim) and sparse
    token-weight maps.  Searches Qdrant using hybrid mode when the
    collection supports it, otherwise falls back to dense-only.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to ``BAAI/bge-m3``.
    device:
        Torch device string (``cuda``, ``cpu``, or ``auto``).
    qdrant_client:
        Qdrant client instance for searching.
    collection_name:
        Default Qdrant collection to search.
    dense_weight:
        Weight for dense score in RRF fusion (default 0.6).
    sparse_weight:
        Weight for sparse score in RRF fusion (default 0.4).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
        qdrant_client: Any = None,
        collection_name: str = "documents",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load the BGE-M3 model."""
        if self._model is not None:
            return

        try:
            from FlagEmbedding import BGEM3FlagModel

            device = self.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading BGE-M3 model on %s", device)
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=(device == "cuda"),
            )
            logger.info("BGE-M3 model loaded successfully")
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed. Install with: pip install FlagEmbedding"
            )
            self._model = None

    def encode(
        self,
        texts: List[str],
        *,
        return_sparse: bool = True,
        return_dense: bool = True,
        max_length: int = 8192,
    ) -> Dict[str, Any]:
        """Encode texts into dense and/or sparse representations.

        Returns
        -------
        Dict with ``dense_vecs`` (numpy array) and ``lexical_weights``
        (list of dicts mapping token_id → weight).
        """
        self._load_model()
        if self._model is None:
            return {"dense_vecs": [], "lexical_weights": []}

        result = self._model.encode(
            texts,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
            max_length=max_length,
        )
        return result

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Search for relevant documents using hybrid dense+sparse retrieval.

        Parameters
        ----------
        query:
            The search query.
        top_k:
            Number of results to return.
        collection_name:
            Override the default collection.
        filter_conditions:
            Additional Qdrant filter conditions.
        subscription_id:
            Filter by subscription (added to filter).
        profile_id:
            Filter by profile (added to filter).

        Returns
        -------
        List of RetrievalResult sorted by combined score.
        """
        collection = collection_name or self.collection_name

        # Encode query
        encoding = self.encode([query])
        if not encoding.get("dense_vecs", []):
            logger.warning("BGE-M3 encoding failed, returning empty results")
            return []

        dense_vec = encoding["dense_vecs"][0].tolist()
        sparse_weights = (
            encoding.get("lexical_weights", [{}])[0]
            if encoding.get("lexical_weights")
            else {}
        )

        # Build Qdrant filter
        must_conditions = []
        if subscription_id:
            must_conditions.append({
                "key": "subscription_id",
                "match": {"value": subscription_id},
            })
        if profile_id:
            must_conditions.append({
                "key": "profile_id",
                "match": {"value": profile_id},
            })
        if filter_conditions:
            must_conditions.extend(
                filter_conditions.get("must", [])
            )

        qdrant_filter = {"must": must_conditions} if must_conditions else None

        if self.qdrant_client is None:
            logger.warning("No Qdrant client configured")
            return []

        # Try hybrid search (dense + sparse)
        try:
            results = self._hybrid_search(
                collection, dense_vec, sparse_weights, top_k, qdrant_filter
            )
        except Exception as exc:
            logger.warning("Hybrid search failed, falling back to dense: %s", exc)
            results = self._dense_search(
                collection, dense_vec, top_k, qdrant_filter
            )

        return results

    def _hybrid_search(
        self,
        collection: str,
        dense_vec: List[float],
        sparse_weights: Dict,
        top_k: int,
        qdrant_filter: Optional[Dict],
    ) -> List[RetrievalResult]:
        """Perform hybrid dense+sparse search via Qdrant."""
        from qdrant_client.models import (
            NamedSparseVector,
            NamedVector,
            Prefetch,
            Query,
            SparseVector,
        )

        # Build sparse indices and values
        indices = []
        values = []
        for token_id, weight in sparse_weights.items():
            indices.append(int(token_id))
            values.append(float(weight))

        prefetch = [
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k * 3,
                filter=qdrant_filter,
            ),
        ]

        if indices:
            prefetch.append(
                Prefetch(
                    query=SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=top_k * 3,
                    filter=qdrant_filter,
                )
            )

        # Use RRF fusion
        hits = self.qdrant_client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=Query(fusion="rrf"),
            limit=top_k,
        )

        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(RetrievalResult(
                text=payload.get("text", payload.get("content", "")),
                combined_score=hit.score,
                metadata=payload,
                source=payload.get("file_name", payload.get("source", "")),
                page=payload.get("page"),
                chunk_id=str(hit.id),
            ))

        return results

    def _dense_search(
        self,
        collection: str,
        dense_vec: List[float],
        top_k: int,
        qdrant_filter: Optional[Dict],
    ) -> List[RetrievalResult]:
        """Fallback dense-only search."""
        hits = self.qdrant_client.search(
            collection_name=collection,
            query_vector=dense_vec,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(RetrievalResult(
                text=payload.get("text", payload.get("content", "")),
                dense_score=hit.score,
                combined_score=hit.score,
                metadata=payload,
                source=payload.get("file_name", payload.get("source", "")),
                page=payload.get("page"),
                chunk_id=str(hit.id),
            ))

        return results

    def search_knowledge_pack(
        self,
        query: str,
        collection_name: str,
        *,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Search a knowledge pack collection (no profile/subscription filter).

        Knowledge pack collections contain authoritative domain content
        (NICE guidelines, legislation, etc.) that is shared across all
        tenants.
        """
        return self.search(
            query,
            top_k=top_k,
            collection_name=collection_name,
        )
