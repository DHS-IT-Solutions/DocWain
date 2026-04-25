"""Insight validators — enforced at the InsightStore writer.

Hard rule (spec Section 5.1, OQ1): every persisted insight has
non-empty evidence_doc_spans. KB refs are augmentation; they cannot
substitute for doc evidence.
"""
from __future__ import annotations

import hashlib
import re

from src.intelligence.insights.schema import Insight


class CitationViolation(ValueError):
    pass


class BodySeparationViolation(ValueError):
    pass


def require_doc_evidence(insight: Insight) -> None:
    if not insight.evidence_doc_spans:
        raise CitationViolation(
            f"insight {insight.insight_id} has no evidence_doc_spans"
        )


def _tokens(text: str) -> set:
    """Lowercase content tokens, ≥4 chars (filters articles, prepositions)."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
    return set(words)


_OVERLAP_THRESHOLD = 0.4


def require_body_grounded(insight: Insight) -> None:
    """Reject insights whose body introduces content not in evidence_doc_spans.

    Heuristic — counts overlap of meaningful tokens between body and
    concatenated quotes. ≥40% body tokens must appear in quotes.
    Per spec Section 8 (OQ1): KB-derived content goes to external_kb_refs,
    never into body text.
    """
    body_tokens = _tokens(insight.body)
    if not body_tokens:
        return
    quote_tokens: set = set()
    for span in insight.evidence_doc_spans:
        quote_tokens |= _tokens(span.quote)
    if not quote_tokens:
        return
    overlap = body_tokens & quote_tokens
    ratio = len(overlap) / len(body_tokens)
    if ratio < _OVERLAP_THRESHOLD:
        raise BodySeparationViolation(
            f"insight {insight.insight_id} body has insufficient overlap "
            f"with doc-span quotes ({ratio:.2f} < {_OVERLAP_THRESHOLD})"
        )


def compute_dedup_key(insight: Insight) -> str:
    """Stable dedup key from (profile_id, document_ids[], insight_type, headline_hash)."""
    sorted_docs = ",".join(sorted(insight.document_ids))
    headline_hash = hashlib.sha256(
        insight.headline.strip().lower().encode("utf-8")
    ).hexdigest()[:16]
    raw = f"{insight.profile_id}|{sorted_docs}|{insight.insight_type}|{headline_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
