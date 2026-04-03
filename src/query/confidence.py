"""Self-verification and confidence scoring for generated responses."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

_MAX_RE_RETRIEVAL_LOOPS = 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Outcome of response verification."""
    passed: bool
    confidence: float
    refined_query: Optional[str] = None
    reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_response(
    response: str,
    context: str,
    query: str,
    *,
    min_confidence: float = 0.7,
) -> VerificationResult:
    """Verify a generated response against the context and query.

    Checks:
        1. Evidence coverage — claims in the response should have supporting chunks.
        2. Citation presence — the response should reference evidence.
        3. Response length vs. query complexity — simple queries should not get
           excessively long answers, and complex queries should not get trivially
           short ones.

    If confidence falls below ``min_confidence``, a refined query is suggested
    targeting the evidence gap.

    Args:
        response: The generated response text.
        context: The assembled context string that was fed to the model.
        query: The original user query.
        min_confidence: Threshold below which a refined query is produced.

    Returns:
        VerificationResult with pass/fail, confidence score, and optional refined query.
    """
    reasons: List[str] = []
    scores: List[float] = []

    # ------------------------------------------------------------------
    # Check 1: Evidence coverage
    # ------------------------------------------------------------------
    coverage_score, coverage_reasons = _check_evidence_coverage(response, context)
    scores.append(coverage_score)
    reasons.extend(coverage_reasons)

    # ------------------------------------------------------------------
    # Check 2: Citation / grounding signals
    # ------------------------------------------------------------------
    citation_score, citation_reasons = _check_citation_presence(response, context)
    scores.append(citation_score)
    reasons.extend(citation_reasons)

    # ------------------------------------------------------------------
    # Check 3: Length appropriateness
    # ------------------------------------------------------------------
    length_score, length_reasons = _check_length_appropriateness(response, query)
    scores.append(length_score)
    reasons.extend(length_reasons)

    # ------------------------------------------------------------------
    # Check 4: Hedging / uncertainty language
    # ------------------------------------------------------------------
    hedge_score, hedge_reasons = _check_hedging(response)
    scores.append(hedge_score)
    reasons.extend(hedge_reasons)

    # Aggregate confidence (weighted average)
    weights = [0.40, 0.20, 0.15, 0.25]
    confidence = sum(s * w for s, w in zip(scores, weights))
    confidence = max(0.0, min(1.0, round(confidence, 3)))

    passed = confidence >= min_confidence
    refined_query = None

    if not passed:
        refined_query = _generate_refined_query(query, response, reasons)
        reasons.append(f"Confidence {confidence:.2f} < threshold {min_confidence:.2f}")

    return VerificationResult(
        passed=passed,
        confidence=confidence,
        refined_query=refined_query,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_evidence_coverage(response: str, context: str) -> tuple[float, List[str]]:
    """Check whether key claims in the response are supported by context evidence."""
    reasons: List[str] = []

    if not context.strip():
        reasons.append("No context evidence was provided")
        return 0.2, reasons

    if not response.strip():
        reasons.append("Response is empty")
        return 0.0, reasons

    # Extract key noun-phrase fragments from the response (simple heuristic:
    # look for capitalised multi-word phrases and numbers)
    claim_fragments = _extract_claim_fragments(response)
    if not claim_fragments:
        # Short or very simple response — give benefit of the doubt
        return 0.8, reasons

    context_lower = context.lower()
    supported = 0
    unsupported_examples: List[str] = []

    for frag in claim_fragments:
        if frag.lower() in context_lower:
            supported += 1
        else:
            unsupported_examples.append(frag)

    coverage = supported / len(claim_fragments) if claim_fragments else 1.0

    if unsupported_examples:
        shown = unsupported_examples[:3]
        reasons.append(
            f"Evidence coverage {coverage:.0%}: {len(unsupported_examples)} claim(s) "
            f"not found in context (e.g., {', '.join(shown)})"
        )
    else:
        reasons.append(f"Evidence coverage {coverage:.0%}: all sampled claims supported")

    # Map coverage ratio to a 0-1 score
    score = min(1.0, coverage + 0.2)  # slight leniency
    return round(score, 2), reasons


def _extract_claim_fragments(text: str) -> List[str]:
    """Extract verifiable fragments from the response text.

    Targets: numbers with context, proper nouns, dates, monetary values.
    """
    fragments: List[str] = []

    # Monetary values with surrounding context
    for m in re.finditer(r"[\$\u20ac\u00a3][\d,]+(?:\.\d+)?", text):
        fragments.append(m.group())

    # Dates (various formats)
    for m in re.finditer(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
        fragments.append(m.group())
    for m in re.finditer(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        text,
        re.IGNORECASE,
    ):
        fragments.append(m.group())

    # Percentages
    for m in re.finditer(r"\b\d+(?:\.\d+)?%", text):
        fragments.append(m.group())

    # Standalone numbers (4+ digits, likely data points)
    for m in re.finditer(r"\b\d[\d,]{3,}\b", text):
        fragments.append(m.group())

    # Deduplicate preserving order
    seen = set()
    unique: List[str] = []
    for f in fragments:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique[:15]  # cap to avoid over-checking


def _check_citation_presence(response: str, context: str) -> tuple[float, List[str]]:
    """Check whether the response implicitly or explicitly references evidence."""
    reasons: List[str] = []

    if not context.strip():
        return 0.5, reasons  # no context to cite

    # Look for signals that the response is grounded
    grounding_signals = [
        "document", "evidence", "section", "page", "report",
        "according to", "states that", "indicates", "shows that",
        "the data", "as noted", "specified",
    ]
    response_lower = response.lower()
    signal_count = sum(1 for s in grounding_signals if s in response_lower)

    if signal_count >= 3:
        score = 1.0
    elif signal_count >= 1:
        score = 0.8
    else:
        score = 0.5
        reasons.append("Response lacks explicit grounding language referencing evidence")

    return score, reasons


def _check_length_appropriateness(response: str, query: str) -> tuple[float, List[str]]:
    """Check whether response length matches query complexity."""
    reasons: List[str] = []
    response_words = len(response.split())
    query_words = len(query.split())

    # Simple query heuristic: short queries (< 10 words) are usually simple
    is_simple = query_words < 10 and "?" in query

    if response_words < 5:
        reasons.append("Response is suspiciously short")
        return 0.3, reasons

    if is_simple and response_words > 500:
        reasons.append(
            f"Response is very long ({response_words} words) for a simple query"
        )
        return 0.6, reasons

    if not is_simple and response_words < 20:
        reasons.append(
            f"Response is very short ({response_words} words) for a complex query"
        )
        return 0.5, reasons

    return 0.9, reasons


def _check_hedging(response: str) -> tuple[float, List[str]]:
    """Penalise excessive hedging / uncertainty language."""
    reasons: List[str] = []
    response_lower = response.lower()

    hedges = [
        "not specified in the documents",
        "not found in the",
        "unable to determine",
        "no information available",
        "i cannot find",
        "the documents do not mention",
        "not enough information",
        "cannot be determined",
    ]
    hedge_count = sum(1 for h in hedges if h in response_lower)

    if hedge_count == 0:
        return 1.0, reasons
    if hedge_count == 1:
        reasons.append("Response contains one hedging/uncertainty phrase")
        return 0.7, reasons
    if hedge_count <= 3:
        reasons.append(f"Response contains {hedge_count} hedging phrases")
        return 0.4, reasons

    reasons.append(f"Response heavily hedged ({hedge_count} uncertainty phrases)")
    return 0.2, reasons


# ---------------------------------------------------------------------------
# Refined query generation
# ---------------------------------------------------------------------------

def _generate_refined_query(
    original_query: str,
    response: str,
    reasons: List[str],
) -> str:
    """Produce a refined query that targets the evidence gap.

    This is a deterministic heuristic; the 27B model is NOT called here to keep
    Phase 2.5 lightweight.
    """
    # Identify what's missing
    gap_keywords: List[str] = []

    for reason in reasons:
        if "not found in context" in reason:
            # Extract the example fragments
            paren = reason.find("(e.g.,")
            if paren != -1:
                tail = reason[paren + 6:].rstrip(")")
                gap_keywords.extend(
                    k.strip() for k in tail.split(",") if k.strip()
                )

    if gap_keywords:
        extras = " ".join(gap_keywords[:3])
        return f"{original_query} specifically regarding {extras}"

    # Generic refinement: ask for more detail
    return f"{original_query} — provide detailed evidence and specific data points"


__all__ = ["VerificationResult", "verify_response"]
