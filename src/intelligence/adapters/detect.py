"""Domain auto-detection for adapter routing.

Wraps the existing src.intelligence.domain_classifier to return a single
chosen-domain decision plus a fallback flag. Confidence threshold is 0.7
per spec Section 6.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

DEFAULT_THRESHOLD = 0.7


@dataclass
class DetectionResult:
    domain: str
    confidence: float
    fallback_to_generic: bool


def _default_classifier(text: str) -> Tuple[str, float]:
    from src.intelligence.domain_classifier import classify_domain
    label, conf = classify_domain(text)
    return label, float(conf or 0.0)


def detect_domain(
    text: str,
    *,
    classifier: Callable[[str], Tuple[str, float]] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> DetectionResult:
    cls = classifier or _default_classifier
    label, confidence = cls(text)
    if confidence < threshold:
        return DetectionResult(
            domain="generic",
            confidence=confidence,
            fallback_to_generic=True,
        )
    return DetectionResult(
        domain=label,
        confidence=confidence,
        fallback_to_generic=False,
    )
