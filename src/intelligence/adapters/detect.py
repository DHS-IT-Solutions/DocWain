"""Domain auto-detection for adapter routing.

Wraps the existing src.intelligence.domain_classifier to return a single
chosen-domain decision plus a fallback flag. Confidence threshold is 0.7
per spec Section 6.2.

The classifier's label space and our adapter name space are not identical;
_LABEL_TO_ADAPTER maps the former to the latter. Unknown labels fall to
generic. A keyword-evidence pre-pass corrects common confusions (HR docs
classifying as resume; contract docs as policy) when distinctive
domain-specific terms appear in the text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Tuple

DEFAULT_THRESHOLD = 0.7


@dataclass
class DetectionResult:
    domain: str
    confidence: float
    fallback_to_generic: bool


# Map classifier output labels to our adapter names
_LABEL_TO_ADAPTER = {
    "policy": "insurance",
    "insurance": "insurance",
    "medical": "medical",
    "clinical": "medical",
    "hr": "hr",
    "human_resources": "hr",
    "employment": "hr",
    "purchase_order": "procurement",
    "rfp": "procurement",
    "rfq": "procurement",
    "procurement": "procurement",
    "contract": "contract",
    "agreement": "contract",
    "legal": "contract",
    "resume": "resume",
    "cv": "resume",
}


# Keyword overrides — when distinctive domain terms appear with sufficient
# count, override the classifier's label. Prevents the common HR-vs-resume
# and contract-vs-policy confusions.
_KEYWORD_OVERRIDES = [
    ("hr", re.compile(
        r"\b(employee\s+id|hire\s+date|pto\s+balance|severance|at-will|"
        r"performance\s+review|probationary\s+period|fmla|exempt|"
        r"non-?compete\s+clause)\b", re.IGNORECASE), 2),
    ("insurance", re.compile(
        r"\b(policyholder|policy\s+number|deductible|premium|"
        r"sum\s+insured|liability\s+limit|exclusions?|endorsement|"
        r"underwriter|claims?\s+history)\b", re.IGNORECASE), 2),
    ("medical", re.compile(
        r"\b(patient|mrn|chief\s+complaint|diagnosis|prescription|"
        r"vitals?|lab\s+report|allergies|icd[-\s]?\d+|dosage)\b",
        re.IGNORECASE), 2),
    ("procurement", re.compile(
        r"\b(rfp|rfq|purchase\s+order|moq|incoterms|sla|"
        r"net\s+\d+\s+days|early-?pay\s+discount|vendor\s+consolidation)\b",
        re.IGNORECASE), 2),
    ("contract", re.compile(
        r"\b(parties?|effective\s+date|indemnification|governing\s+law|"
        r"dispute\s+resolution|force\s+majeure|severability|assignment\s+clause)\b",
        re.IGNORECASE), 3),
    ("resume", re.compile(
        r"\b(resume|curriculum\s+vitae|education|experience|certifications?|"
        r"summary\s*:?\s*\n)\b", re.IGNORECASE), 3),
]


def _keyword_override(text: str) -> str | None:
    """Return adapter name if a domain has clear keyword evidence."""
    best = None
    best_count = 0
    for adapter, rx, min_count in _KEYWORD_OVERRIDES:
        hits = len(rx.findall(text))
        if hits >= min_count and hits > best_count:
            best = adapter
            best_count = hits
    return best


def _default_classifier(text: str) -> Tuple[str, float]:
    from src.intelligence.domain_classifier import classify_domain
    result = classify_domain(text)
    label = getattr(result, "domain", None) or "generic"
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    return label, confidence


def detect_domain(
    text: str,
    *,
    classifier: Callable[[str], Tuple[str, float]] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> DetectionResult:
    cls = classifier or _default_classifier

    # 1. Keyword evidence override — strongest signal when present
    kw = _keyword_override(text)
    if kw is not None:
        return DetectionResult(domain=kw, confidence=0.95, fallback_to_generic=False)

    # 2. Otherwise consult the classifier
    label, confidence = cls(text)
    adapter = _LABEL_TO_ADAPTER.get(label, "generic")

    if confidence < threshold or adapter == "generic":
        return DetectionResult(
            domain="generic",
            confidence=confidence,
            fallback_to_generic=True,
        )
    return DetectionResult(
        domain=adapter,
        confidence=confidence,
        fallback_to_generic=False,
    )
