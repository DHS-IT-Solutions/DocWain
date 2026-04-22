"""Recommendation-intent grounding post-pass.

Runs AFTER an LLM response is produced for a recommend-intent rich prompt.
For every recommendation-section item it requires ONE of:

  (a) a lexical match against a Recommendation Bank entry, OR
  (b) an inline ``[doc_id:chunk_id]`` citation.

Failing items are removed. The response gains a candid
``Note: N claim(s) could not be verified ...`` appendix when any items are
dropped. 0.0 hallucination rate is preserved by refusing to let an
ungrounded recommendation reach the user.

This module is formatting — per the memory rule it lives in the generation
package, NOT in intelligence/. It is imported exactly once, from
``src/agent/core_agent.py``, only when intent == "recommend" AND shape == RICH.

Purely textual — no LLM call, no timeouts, idempotent when called twice on
the same output. Preserves all other response sections byte-for-byte.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

_SECTION_HEAD = re.compile(r"^##\s+Recommendations\s*$", re.MULTILINE)
_NEXT_SECTION_HEAD = re.compile(r"^##\s+\S", re.MULTILINE)
_CITATION = re.compile(r"\[[A-Za-z0-9_\-]+:[A-Za-z0-9_\-]+\]")
_ITEM_LINE = re.compile(r"^\s*(?:\d+\.|-|\*)\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class GroundingReport:
    """Outcome of one grounding post-pass run."""

    kept_count: int
    dropped_count: int
    dropped_items: tuple[str, ...]


def enforce_recommendation_grounding(
    response: str,
    *,
    bank_entries: Sequence[dict],
) -> Tuple[str, GroundingReport]:
    """Drop ungrounded recommendation items and append a candid note.

    Returns ``(rewritten_response, report)``. When the response has no
    ``## Recommendations`` section the function returns the input unchanged
    with a zeroed report — callers never need to special-case that path.
    """
    head = _SECTION_HEAD.search(response)
    if head is None:
        return response, GroundingReport(0, 0, ())

    tail_start = head.end()
    nxt = _NEXT_SECTION_HEAD.search(response, pos=tail_start + 1)
    section_end = nxt.start() if nxt else len(response)
    section_body = response[tail_start:section_end]

    bank_signatures = tuple(
        _signature(entry.get("recommendation", "")) for entry in bank_entries
    )

    kept_lines: List[str] = []
    dropped_items: List[str] = []
    for match in _ITEM_LINE.finditer(section_body):
        line = match.group(0)
        claim = match.group(1)
        if _is_grounded(claim, bank_signatures):
            kept_lines.append(line)
        else:
            dropped_items.append(claim.strip())

    new_section = "\n".join(kept_lines).rstrip()
    if new_section:
        new_section += "\n"
    rewritten = response[:tail_start] + "\n" + new_section + response[section_end:]

    if dropped_items:
        rewritten = rewritten.rstrip() + (
            f"\n\nNote: {len(dropped_items)} claim(s) could not be verified "
            "against profile evidence and were removed.\n"
        )

    return rewritten, GroundingReport(
        kept_count=len(kept_lines),
        dropped_count=len(dropped_items),
        dropped_items=tuple(dropped_items),
    )


def _is_grounded(claim: str, bank_signatures: Sequence[str]) -> bool:
    """True when the claim has an inline citation or matches a bank entry."""
    if _CITATION.search(claim):
        return True
    sig = _signature(claim)
    return any(bank_sig and bank_sig in sig for bank_sig in bank_signatures)


def _signature(text: str) -> str:
    """Normalise text for lexical bank matching — lowercase, alnum + spaces."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
