"""Merge ephemeral URL chunks with profile retrieval output.

Stage-3-level helper called by CoreAgent after reranking. Concentrates the
merge semantics in one place so ``core_agent.py`` stays lean.

Semantics by case (from ``CaseSelection``):

* ``NONE``          — profile chunks only (URLs weren't processed anyway).
* ``SUPPLEMENTARY`` — profile chunks first, URL chunks appended.
* ``PRIMARY``       — URL chunks first, profile chunks appended as context.

Ephemeral chunks are shape-adapted to :class:`SimpleNamespace` so
downstream code that reads ``chunk.text`` / ``chunk.metadata`` works
without further conversion.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

from src.agent.url_case_selector import CaseSelection
from src.tools.url_ephemeral_source import EphemeralChunk


def _ephemeral_to_chunk(e: EphemeralChunk) -> Any:
    md = dict(e.metadata)
    md.setdefault("provenance", "ephemeral_url")
    md.setdefault("ephemeral", True)
    return SimpleNamespace(
        text=e.text,
        score=md.get("ephemeral_score", 0.7),
        metadata=md,
        document_id=md.get("source_url", "url"),
        chunk_id=f"url:{md.get('source_url', 'x')}#{md.get('chunk_index', 0)}",
        embedding=e.embedding,
    )


def merge_ephemeral(
    profile_chunks: List[Any],
    ephemeral_chunks: List[EphemeralChunk],
    *,
    case: CaseSelection,
) -> List[Any]:
    """Merge ``ephemeral_chunks`` into ``profile_chunks`` per the case."""
    if case == CaseSelection.NONE or not ephemeral_chunks:
        return list(profile_chunks)

    ephemeral_shaped = [_ephemeral_to_chunk(e) for e in ephemeral_chunks]

    if case == CaseSelection.PRIMARY:
        return ephemeral_shaped + list(profile_chunks)

    # SUPPLEMENTARY (default for URL-bearing queries with a strong profile).
    return list(profile_chunks) + ephemeral_shaped
