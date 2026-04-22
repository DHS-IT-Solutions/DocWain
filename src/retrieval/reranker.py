"""Ensemble reranking for retrieved evidence chunks.

Two reranker surfaces share this module:

* :func:`rerank_chunks` — the legacy dense+keyword+optional-cross-encoder
  ensemble used by the existing retrieval path; unchanged.
* :class:`CrossEncoderReranker` — the spec §7 stage 2 cross-encoder
  wrapper used by the SME retrieval layer (flag-gated via
  ``enable_cross_encoder_rerank``). Default model
  ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (resolved per spec §15 Q2);
  swap by passing a different ``model_name``.

The two are deliberately kept in one module so downstream code can import
either shape from ``src.retrieval.reranker`` without new paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

from src.retrieval.retriever import EvidenceChunk


#: Default cross-encoder model name (spec §15 Q2 resolution).
DEFAULT_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _keyword_f1(query: str, text: str) -> float:
    """Token-level F1 overlap between query and text."""
    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = q_tokens & t_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(q_tokens)
    recall = len(overlap) / len(t_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(scores: List[float]) -> List[float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def rerank_chunks(
    query: str,
    chunks: List[EvidenceChunk],
    top_k: int = 15,
    cross_encoder=None,
) -> List[EvidenceChunk]:
    """Ensemble rerank: dense score + keyword F1 + optional cross-encoder.

    Weights:
        Without cross-encoder: 0.7 * dense + 0.3 * keyword_f1
        With cross-encoder:    0.5 * dense + 0.2 * keyword_f1 + 0.3 * ce
    """
    if not chunks:
        return []

    dense_scores = _normalize([c.score for c in chunks])
    kw_scores = _normalize([_keyword_f1(query, c.text) for c in chunks])

    if cross_encoder is not None:
        # Pre-filter to top candidates by dense+keyword before expensive CE
        # CE on CPU costs ~0.6s per pair; cap at 10 pairs max
        _CE_MAX_PAIRS = 10
        _MAX_CE_CHARS = 1600  # MiniLM has 512 token limit

        if len(chunks) > _CE_MAX_PAIRS:
            # Score with dense+keyword first, CE only top candidates
            pre_scores = [0.7 * d + 0.3 * k for d, k in zip(dense_scores, kw_scores)]
            indexed = sorted(enumerate(pre_scores), key=lambda x: x[1], reverse=True)
            ce_indices = [idx for idx, _ in indexed[:_CE_MAX_PAIRS]]
            ce_index_set = set(ce_indices)

            pairs = [[query, chunks[i].text[:_MAX_CE_CHARS]] for i in ce_indices]
            raw_ce = cross_encoder.predict(pairs)
            ce_map = {ce_indices[j]: float(raw_ce[j]) for j in range(len(ce_indices))}
            raw_ce_all = [ce_map.get(i, 0.0) for i in range(len(chunks))]
            ce_scores = _normalize(raw_ce_all)
        else:
            pairs = [[query, c.text[:_MAX_CE_CHARS]] for c in chunks]
            raw_ce = cross_encoder.predict(pairs)
            ce_scores = _normalize(list(raw_ce))

        combined = [
            0.5 * d + 0.2 * k + 0.3 * ce
            for d, k, ce in zip(dense_scores, kw_scores, ce_scores)
        ]
    else:
        combined = [
            0.7 * d + 0.3 * k
            for d, k in zip(dense_scores, kw_scores)
        ]

    # Create new chunks with updated scores
    scored = []
    for chunk, new_score in zip(chunks, combined):
        scored.append(replace(chunk, score=new_score))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# SME cross-encoder reranker (spec §7 stage 2)
# ---------------------------------------------------------------------------


@dataclass
class RerankCandidate:
    """Input shape for the SME cross-encoder reranker.

    ``id`` identifies the upstream item (matches ``HybridResult.item_id``);
    ``text`` is the full candidate text the cross-encoder scores against the
    query. Additional per-candidate metadata rides on the caller side — the
    reranker only sees what it needs to score.
    """

    id: str
    text: str


def _load_model(model_name: str):
    """Load a sentence-transformers CrossEncoder. Isolated so tests can patch.

    Imported lazily inside the function body so importing this module never
    triggers a heavy download / GPU init. Also makes mock injection in tests
    a one-line patch against ``src.retrieval.reranker._load_model``.
    """
    from sentence_transformers import CrossEncoder  # noqa: WPS433 — lazy import

    return CrossEncoder(model_name)


@dataclass
class RerankScore:
    """Candidate + model score pair returned by :class:`CrossEncoderReranker`.

    The raw model output is exposed so callers that want to threshold or log
    absolute scores don't have to re-run the model. The SME retrieval layer
    treats the ranking as opaque and simply slices the top-N.
    """

    candidate: RerankCandidate
    score: float


class CrossEncoderReranker:
    """Cross-encoder reranker for SME retrieval (spec §7 stage 2).

    Lazy-loads the underlying model on the first :meth:`rerank` call so
    construction is cheap and tests don't need a live model. The default
    model name is :data:`DEFAULT_CROSS_ENCODER_MODEL` — swap by passing a
    different ``model_name`` to the constructor.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    ) -> None:
        self._name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._name

    def _ensure(self) -> None:
        if self._model is None:
            self._model = _load_model(self._name)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int = 10,
    ) -> List[RerankCandidate]:
        """Score every candidate against ``query`` and return the top-``top_k``.

        Returns a freshly-sorted list of :class:`RerankCandidate` in descending
        score order. Empty-candidate input short-circuits without loading the
        model so callers can treat the reranker as a cheap no-op when their
        upstream retrieval returned nothing.
        """
        if not candidates:
            return []
        self._ensure()
        pairs = [(query, c.text) for c in candidates]
        raw_scores = self._model.predict(pairs)
        scored = sorted(
            zip(candidates, raw_scores),
            key=lambda pair: -float(pair[1]),
        )
        return [cand for cand, _ in scored[:top_k]]

    def rerank_with_scores(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int = 10,
    ) -> List[RerankScore]:
        """Variant that returns (candidate, score) pairs sorted descending.

        Useful for layered retrieval stages that blend the cross-encoder
        score with other signals, or for diagnostics.
        """
        if not candidates:
            return []
        self._ensure()
        pairs = [(query, c.text) for c in candidates]
        raw_scores = self._model.predict(pairs)
        scored = sorted(
            (RerankScore(candidate=c, score=float(s)) for c, s in zip(candidates, raw_scores)),
            key=lambda rs: -rs.score,
        )
        return scored[:top_k]
