"""Query-shape fingerprinting.

Deliberately simple: normalize tokens, drop stop words, sha1 the remaining
joined form. 16-hex-char prefix is enough collision resistance for cluster
grouping without carrying raw query text into analytics files.
"""
from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"[A-Za-z0-9_\-]+")

# Intentionally minimal stop list — we want query shapes, not topics.
_STOP: frozenset[str] = frozenset(
    {
        "a", "an", "the", "of", "for", "to", "in", "on", "at", "by",
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "done",
        "has", "have", "had",
        "i", "we", "you", "they", "he", "she", "it",
        "me", "us", "them", "him", "her",
        "my", "our", "your", "their", "his",
        "and", "or", "but", "if", "so", "than",
        "that", "this", "these", "those",
        "what", "which", "who", "when", "where", "how", "why",
    }
)


def normalize_tokens(text: str) -> list[str]:
    """Tokenize + lowercase + drop stop words.

    Preserves alphanumerics and dashes/underscores so identifiers like
    ``INV-2026-Q3-0048`` survive intact.
    """
    raw = _WORD.findall(text or "")
    out: list[str] = []
    for tok in raw:
        low = tok.lower()
        if low in _STOP:
            continue
        out.append(low)
    return out


def fingerprint_query(text: str) -> str:
    """Return a 16-hex-char sha1 prefix over normalized tokens.

    Stable across whitespace/case differences. Returns a fingerprint even when
    the input is stop-words-only (using an ``__empty_query__`` sentinel) so
    downstream code never sees an empty fingerprint.
    """
    toks = normalize_tokens(text)
    canonical = " ".join(toks) if toks else "__empty_query__"
    h = hashlib.sha1(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()
    return h[:16]
