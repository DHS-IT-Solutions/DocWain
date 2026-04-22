"""URL-as-prompt case selector: supplementary vs primary.

Pure function, no LLM call, no IO. Decides whether URL content in a user
query should be treated as *supplementary* to a strong profile pack or as
the *primary* driver when the profile pack is weak and the query is
URL-directed.

Inputs arrive from the CoreAgent after Stage-1 retrieval signals land.
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Literal


class CaseSelection(str, enum.Enum):
    NONE = "none"                    # no URLs in query -> ignore this machinery
    SUPPLEMENTARY = "supplementary"  # profile pack drives the answer
    PRIMARY = "primary"              # URL content drives the answer


# Accept Literal type for typed call sites as mentioned in the plan preamble.
CaseName = Literal["none", "supplementary", "primary"]


@dataclass(frozen=True)
class RetrievalSignal:
    sme_artifact_count: int = 0
    high_sim_chunk_count: int = 0


_URL_DIRECTED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bsummari[sz]e\s+(this|the|that|link|page|article|post|url)\b",
        r"\b(what|explain)\s+(does|is)\s+(this|that|the)\s+(page|link|article|post|url)\b",
        r"\btl;?\s*dr\b",
        r"\bread\s+(this|that)\b",
        r"\brender\s+this\b",
        r"\banalyze\s+this\b",
        r"\bextract\s+(text|content|from)\b",
    )
]

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "of", "to", "and", "or", "for", "on", "in",
    "at", "it", "this", "that", "these", "those", "please", "kindly",
    "from", "with", "be", "as", "by",
})


def _token_count(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    return sum(1 for t in tokens if t not in _STOPWORDS)


def _is_url_dominant(cleaned_query: str, *, token_cap: int) -> bool:
    if not cleaned_query.strip():
        return True
    for pat in _URL_DIRECTED_PATTERNS:
        if pat.search(cleaned_query):
            return True
    return _token_count(cleaned_query) <= token_cap


@dataclass
class CaseSelector:
    """Heuristic selector — explicit rule, no LLM call.

    A retrieval result is *strong* when either the profile yielded at least
    ``strong_artifact_count`` SME artifacts OR at least
    ``strong_high_sim_chunk_count`` chunks above the similarity threshold.

    A query is *URL-dominant* when fewer than ``url_dominant_token_cap``
    non-stopword tokens remain after URL removal, or when the cleaned query
    matches one of the URL-directed imperative patterns.

    Decision matrix:

    ============= ================ ===================
    Profile        Query shape      Case
    ============= ================ ===================
    strong         anything         supplementary
    weak           URL-dominant     primary
    weak           not URL-dominant supplementary
    ============= ================ ===================
    """
    strong_artifact_count: int = 1
    strong_high_sim_chunk_count: int = 3
    url_dominant_token_cap: int = 8

    def select(
        self,
        *,
        cleaned_query: str,
        url_count: int,
        signal: RetrievalSignal,
    ) -> CaseSelection:
        if url_count <= 0:
            return CaseSelection.NONE

        strong = (
            signal.sme_artifact_count >= self.strong_artifact_count
            or signal.high_sim_chunk_count >= self.strong_high_sim_chunk_count
        )
        if strong:
            return CaseSelection.SUPPLEMENTARY

        if _is_url_dominant(
            cleaned_query, token_cap=self.url_dominant_token_cap
        ):
            return CaseSelection.PRIMARY

        return CaseSelection.SUPPLEMENTARY


def select_case(
    *,
    cleaned_query: str,
    url_count: int,
    signal: RetrievalSignal,
) -> CaseSelection:
    """Default-configured wrapper."""
    return CaseSelector().select(
        cleaned_query=cleaned_query,
        url_count=url_count,
        signal=signal,
    )
