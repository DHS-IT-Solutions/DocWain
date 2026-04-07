"""Lightweight rule-based query classifier.

Categorises incoming queries into SIMPLE, COMPLEX, ANALYTICAL, or
CONVERSATIONAL at <50 ms with zero LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

QueryType = Literal["SIMPLE", "COMPLEX", "ANALYTICAL", "CONVERSATIONAL"]


@dataclass
class QueryClassification:
    query_type: QueryType
    confidence: float  # 0.0-1.0
    signals: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_GREETING = re.compile(
    r"^(hi|hello|hey|hiya|yo|greetings|howdy|good\s+(morning|afternoon|evening))"
    r"[!.,?\s]*$",
    re.IGNORECASE,
)

_THANKS = re.compile(
    r"\b(thanks|thank\s+you|thx|ty|cheers|appreciate\s+it|much\s+appreciated)\b",
    re.IGNORECASE,
)

_BYE = re.compile(
    r"^(bye|goodbye|see\s+you|take\s+care|later)[!.,?\s]*$",
    re.IGNORECASE,
)

_AFFIRMATION = re.compile(
    r"^(ok|okay|yes|no|sure|got\s+it|alright|right|yep|nope|nah)[!.,?\s]*$",
    re.IGNORECASE,
)

_ANALYTICAL_KEYWORDS = re.compile(
    r"\b(suggest|improve|recommend|analyze|analyse|compare\s+trends|identify\s+risks"
    r"|what\s+should|areas?\s+(?:for|need(?:ing)?)\s+improvement|opportunities|concerns"
    r"|based\s+on\s+the\s+data|looking\s+at\s+the\s+numbers"
    r"|strengths?\s+and\s+weaknesses?"
    r"|evaluate|assessment|implications|insights?)\b",
    re.IGNORECASE,
)

_MULTI_DOC = re.compile(
    r"\b(compare|across|between\s+(?:documents?|the)|all\s+documents?"
    r"|each\s+document|differences?|versus|vs\.?)\b",
    re.IGNORECASE,
)

_QUESTION_WORD = re.compile(
    r"\b(what|who|when|where|which|how|why|is|are|was|were|did|does|do|can|could)\b",
    re.IGNORECASE,
)

_SINGLE_FACTOID = re.compile(
    r"^(what\s+is|who\s+is|when\s+was|when\s+is|where\s+is|what\s+was|what\s+are)\b",
    re.IGNORECASE,
)

_DOC_TERMS = re.compile(
    r"\b(document|report|file|data|revenue|profit|margin|cost|budget|sales"
    r"|employee|patient|contract|invoice|claim|policy|total|amount|balance"
    r"|summary|section|table|page|paragraph|chart|figure)\b",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(text.split())


def classify_query(
    query: str,
    conversation_history: Optional[list] = None,
) -> QueryClassification:
    """Classify *query* without any LLM call.

    Parameters
    ----------
    query:
        The raw user query string.
    conversation_history:
        Optional prior turns (unused today, reserved for future follow-up
        detection).

    Returns
    -------
    QueryClassification
    """
    text = query.strip()
    if not text:
        return QueryClassification(query_type="CONVERSATIONAL", confidence=0.9, signals=["empty_query"])

    words = _word_count(text)
    signals: list[str] = []

    # ------------------------------------------------------------------
    # 1. CONVERSATIONAL — greetings / thanks / short non-question
    # ------------------------------------------------------------------
    if _GREETING.search(text):
        signals.append("greeting_pattern")
        return QueryClassification(query_type="CONVERSATIONAL", confidence=0.95, signals=signals)

    if _BYE.search(text):
        signals.append("farewell_pattern")
        return QueryClassification(query_type="CONVERSATIONAL", confidence=0.95, signals=signals)

    if _AFFIRMATION.search(text):
        signals.append("short_affirmation")
        return QueryClassification(query_type="CONVERSATIONAL", confidence=0.90, signals=signals)

    if _THANKS.search(text) and words < 10:
        signals.append("thanks_pattern")
        return QueryClassification(query_type="CONVERSATIONAL", confidence=0.90, signals=signals)

    # Short, no question word, no document terms → conversational
    if words < 5 and not _QUESTION_WORD.search(text) and not _DOC_TERMS.search(text):
        signals.append("short_non_question")
        return QueryClassification(query_type="CONVERSATIONAL", confidence=0.80, signals=signals)

    # ------------------------------------------------------------------
    # 2. ANALYTICAL — suggestions / recommendations / analysis
    # ------------------------------------------------------------------
    if _ANALYTICAL_KEYWORDS.search(text):
        signals.append("analytical_keyword")
        conf = 0.90 if words > 8 else 0.80
        return QueryClassification(query_type="ANALYTICAL", confidence=conf, signals=signals)

    # ------------------------------------------------------------------
    # 3. COMPLEX — multi-doc / long / multiple questions
    # ------------------------------------------------------------------
    if _MULTI_DOC.search(text):
        signals.append("multi_doc_keyword")
        return QueryClassification(query_type="COMPLEX", confidence=0.90, signals=signals)

    question_marks = text.count("?")
    if question_marks >= 2:
        signals.append("multiple_questions")
        return QueryClassification(query_type="COMPLEX", confidence=0.85, signals=signals)

    if words > 30:
        signals.append("long_query")
        return QueryClassification(query_type="COMPLEX", confidence=0.80, signals=signals)

    # ------------------------------------------------------------------
    # 4. SIMPLE — ONLY single-fact lookups (name, date, amount, yes/no)
    # ------------------------------------------------------------------
    if _SINGLE_FACTOID.search(text) and words < 10 and not _MULTI_DOC.search(text):
        signals.append("short_factoid")
        return QueryClassification(query_type="SIMPLE", confidence=0.85, signals=signals)

    # ------------------------------------------------------------------
    # 5. Default → COMPLEX (safe fallback)
    # ------------------------------------------------------------------
    signals.append("default_fallback")
    return QueryClassification(query_type="COMPLEX", confidence=0.60, signals=signals)
