"""Citation Verifier — separate verification pass for claim grounding.

Checks whether each factual claim in a generated response is supported
by the source chunks that were used to produce it.
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Set

logger = logging.getLogger(__name__)

VerificationStatus = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]

# Stop-words excluded from keyword overlap computation
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "if", "when", "while", "where", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "it", "its", "he", "she", "they", "them", "their", "we", "our",
    "you", "your", "i", "me", "my", "also", "there", "here",
}

# Phrases that indicate meta-commentary rather than factual claims
_META_PREFIXES = [
    "based on the documents",
    "based on the provided",
    "based on the context",
    "based on the information",
    "according to the documents",
    "according to the provided",
    "according to the context",
    "the documents show",
    "the documents indicate",
    "the document mentions",
    "as mentioned in",
    "as stated in",
    "as noted in",
]

_HEDGE_PHRASES = [
    "it's worth noting that",
    "it is worth noting that",
    "it should be noted that",
    "it's important to note that",
    "it is important to note that",
    "please note that",
    "note that",
    "keep in mind that",
    "it appears that",
    "it seems that",
]


@dataclass
class ClaimVerification:
    claim: str
    status: VerificationStatus
    supporting_chunk: str  # the chunk text that supports this claim (empty if unsupported)
    confidence: float  # 0.0-1.0


@dataclass
class VerificationResult:
    claims: List[ClaimVerification]
    overall_score: float  # ratio of SUPPORTED claims to total
    grounding_density: float  # claims per 100 words of response
    flagged_claims: List[str]  # claims that are UNSUPPORTED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def _significant_words(text: str) -> Set[str]:
    """Return set of non-stop-word tokens from *text*."""
    return {w for w in _tokenize(text) if w not in _STOP_WORDS and len(w) > 1}


def _keyword_overlap(claim_words: Set[str], chunk_text: str) -> float:
    """Fraction of *claim_words* that appear in *chunk_text*."""
    if not claim_words:
        return 0.0
    chunk_words = _significant_words(chunk_text)
    matched = claim_words & chunk_words
    return len(matched) / len(claim_words)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on common boundaries."""
    # Split on period/exclamation/question followed by whitespace or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Also split on newlines that separate content
    result: List[str] = []
    for part in parts:
        for sub in part.split("\n"):
            sub = sub.strip()
            if sub:
                result.append(sub)
    return result


def _strip_meta_prefix(sentence: str) -> str:
    """Remove leading meta-commentary phrases, returning the factual remainder."""
    lower = sentence.lower()
    for prefix in _META_PREFIXES:
        if lower.startswith(prefix):
            remainder = sentence[len(prefix):].lstrip(" ,;:-")
            if remainder:
                # Capitalise first char
                return remainder[0].upper() + remainder[1:] if len(remainder) > 1 else remainder.upper()
    return sentence


def _strip_hedge_prefix(sentence: str) -> str:
    """Remove leading hedge phrases, returning the factual remainder."""
    lower = sentence.lower()
    for hedge in _HEDGE_PHRASES:
        if lower.startswith(hedge):
            remainder = sentence[len(hedge):].lstrip(" ,;:-")
            if remainder:
                return remainder[0].upper() + remainder[1:] if len(remainder) > 1 else remainder.upper()
    return sentence


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?")


def _is_too_short(sentence: str, min_words: int = 3) -> bool:
    return len(sentence.split()) < min_words


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_claims(response_text: str) -> List[str]:
    """Extract individual factual claims from a response.

    Split response into sentences, filter to factual claims
    (skip questions, hedges, meta-commentary).
    """
    if not response_text or not response_text.strip():
        return []

    sentences = _split_sentences(response_text)
    claims: List[str] = []

    for sent in sentences:
        # Skip questions
        if _is_question(sent):
            continue

        # Strip meta-commentary prefix — keep the factual part
        cleaned = _strip_meta_prefix(sent)
        cleaned = _strip_hedge_prefix(cleaned)

        # After stripping, skip if only hedge/meta remains or too short
        if _is_too_short(cleaned):
            continue

        # Skip pure hedges that have no factual content after stripping
        lower = cleaned.lower()
        if any(lower == hedge.rstrip() for hedge in _HEDGE_PHRASES):
            continue

        claims.append(cleaned)

    return claims


def verify_claims(
    claims: List[str],
    source_chunks: List[Dict[str, Any]],
    llm_fn: Optional[Callable[..., str]] = None,
) -> List[ClaimVerification]:
    """Verify each claim against source chunks.

    For each claim, check if any source chunk contains supporting evidence.
    Uses keyword overlap first (fast), then LLM for ambiguous cases.
    """
    if not claims:
        return []

    results: List[ClaimVerification] = []

    for claim in claims:
        claim_words = _significant_words(claim)
        best_overlap = 0.0
        best_chunk_text = ""

        for chunk in source_chunks:
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                continue
            overlap = _keyword_overlap(claim_words, chunk_text)
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk_text = chunk_text

        # Fast path: high overlap -> SUPPORTED
        if best_overlap >= 0.6:
            results.append(ClaimVerification(
                claim=claim,
                status="SUPPORTED",
                supporting_chunk=best_chunk_text,
                confidence=min(best_overlap, 1.0),
            ))
        # Ambiguous: moderate overlap, try LLM if available
        elif best_overlap >= 0.3:
            if llm_fn is not None:
                try:
                    verdict = _llm_verify(claim, best_chunk_text, llm_fn)
                    results.append(ClaimVerification(
                        claim=claim,
                        status=verdict,
                        supporting_chunk=best_chunk_text if verdict != "UNSUPPORTED" else "",
                        confidence=best_overlap,
                    ))
                except Exception:
                    logger.warning("LLM verification failed for claim, falling back to PARTIAL", exc_info=True)
                    results.append(ClaimVerification(
                        claim=claim,
                        status="PARTIAL",
                        supporting_chunk=best_chunk_text,
                        confidence=best_overlap,
                    ))
            else:
                # No LLM available — mark PARTIAL
                results.append(ClaimVerification(
                    claim=claim,
                    status="PARTIAL",
                    supporting_chunk=best_chunk_text,
                    confidence=best_overlap,
                ))
        # No match
        else:
            results.append(ClaimVerification(
                claim=claim,
                status="UNSUPPORTED",
                supporting_chunk="",
                confidence=best_overlap,
            ))

    return results


def verify_response(
    response_text: str,
    source_chunks: List[Dict[str, Any]],
    llm_fn: Optional[Callable[..., str]] = None,
) -> VerificationResult:
    """Full verification pipeline: extract claims, verify each, compute scores."""
    claims = extract_claims(response_text)

    if not claims:
        return VerificationResult(
            claims=[],
            overall_score=1.0,  # no claims = nothing to dispute
            grounding_density=0.0,
            flagged_claims=[],
        )

    verifications = verify_claims(claims, source_chunks, llm_fn=llm_fn)

    supported_count = sum(1 for v in verifications if v.status == "SUPPORTED")
    overall_score = supported_count / len(verifications) if verifications else 1.0

    word_count = len(response_text.split())
    grounding_density = (len(claims) / word_count * 100) if word_count > 0 else 0.0

    flagged = [v.claim for v in verifications if v.status == "UNSUPPORTED"]

    return VerificationResult(
        claims=verifications,
        overall_score=overall_score,
        grounding_density=grounding_density,
        flagged_claims=flagged,
    )


# ---------------------------------------------------------------------------
# LLM verification helper
# ---------------------------------------------------------------------------

def _llm_verify(claim: str, chunk_text: str, llm_fn: Callable[..., str]) -> VerificationStatus:
    """Ask the LLM whether *chunk_text* supports *claim*."""
    prompt = (
        "Does the following source support this claim? "
        f"Claim: {claim}. "
        f"Source: {chunk_text}. "
        "Answer YES, PARTIAL, or NO."
    )
    response = llm_fn(prompt).strip().upper()

    if "YES" in response:
        return "SUPPORTED"
    elif "PARTIAL" in response:
        return "PARTIAL"
    else:
        return "UNSUPPORTED"
