"""Parent-child chunker: splits text into parent chunks with overlapping child chunks.

Enables fine-grained retrieval (on children) with context expansion (to parents).
Parent chunks align to section boundaries when provided; children overlap for
continuity across chunk edges.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChildChunk:
    text: str
    chunk_id: str
    parent_id: str
    position: int  # 0-indexed position within parent
    overlap_before: str  # overlapping text from previous child
    overlap_after: str  # overlapping text from next child
    metadata: dict = field(default_factory=dict)


@dataclass
class ParentChunk:
    text: str
    chunk_id: str
    section_title: str
    children: List[ChildChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _approx_token_count(text: str) -> int:
    """Rough token estimate: split on whitespace."""
    return len(text.split())


def _split_into_words(text: str) -> List[str]:
    """Split text into words preserving whitespace by returning words."""
    return text.split()


def _words_to_text(words: List[str]) -> str:
    return " ".join(words)


def _extract_section_title(text: str) -> str:
    """Extract a short section title from the start of a chunk."""
    first_line = text.strip().split("\n", 1)[0]
    return first_line[:120].strip() if first_line else ""


def chunk_with_parents(
    text: str,
    document_id: str,
    parent_size: int = 1500,
    child_size: int = 300,
    overlap_size: int = 50,
    section_boundaries: Optional[List[int]] = None,
) -> List[ParentChunk]:
    """Split text into parent chunks at section boundaries, then split each
    parent into overlapping children.

    Parameters
    ----------
    text : str
        Full document text.
    document_id : str
        Unique document identifier used to build chunk IDs.
    parent_size : int
        Target parent chunk size in approximate tokens (word count).
    child_size : int
        Target child chunk size in approximate tokens.
    overlap_size : int
        Number of overlap tokens between consecutive children.
    section_boundaries : list[int] | None
        Character positions where section breaks occur.  When provided,
        parent chunks are split at these boundaries instead of at fixed
        token intervals.

    Returns
    -------
    list[ParentChunk]
        Parent chunks each containing their child chunks.
    """
    if not text or not text.strip():
        return []

    # --- Step 1: build parent-level text segments ---
    parent_texts: List[str] = []

    if section_boundaries:
        # Sort and deduplicate boundaries, add start/end sentinels
        bounds = sorted(set(section_boundaries))
        positions = [0] + [b for b in bounds if 0 < b < len(text)] + [len(text)]
        for i in range(len(positions) - 1):
            segment = text[positions[i]: positions[i + 1]].strip()
            if segment:
                parent_texts.append(segment)
    else:
        # Split at roughly parent_size word intervals
        words = _split_into_words(text)
        if not words:
            return []
        start = 0
        while start < len(words):
            end = min(start + parent_size, len(words))
            segment = _words_to_text(words[start:end]).strip()
            if segment:
                parent_texts.append(segment)
            start = end

    # If section-boundary splitting produced segments much larger than
    # parent_size, sub-split them so parents stay reasonably sized.
    final_parent_texts: List[str] = []
    for seg in parent_texts:
        seg_words = _split_into_words(seg)
        if len(seg_words) <= parent_size * 1.5:
            final_parent_texts.append(seg)
        else:
            idx = 0
            while idx < len(seg_words):
                chunk_end = min(idx + parent_size, len(seg_words))
                part = _words_to_text(seg_words[idx:chunk_end]).strip()
                if part:
                    final_parent_texts.append(part)
                idx = chunk_end

    # --- Step 2: create ParentChunks and their children ---
    parents: List[ParentChunk] = []

    for p_idx, p_text in enumerate(final_parent_texts):
        parent_id = f"{document_id}_p{p_idx}_{uuid.uuid4().hex[:8]}"
        section_title = _extract_section_title(p_text)

        parent = ParentChunk(
            text=p_text,
            chunk_id=parent_id,
            section_title=section_title,
        )

        # Split parent text into overlapping children
        words = _split_into_words(p_text)
        if not words:
            parents.append(parent)
            continue

        children: List[ChildChunk] = []
        step = max(child_size - overlap_size, 1)
        c_idx = 0
        position = 0

        while c_idx < len(words):
            c_end = min(c_idx + child_size, len(words))
            child_text = _words_to_text(words[c_idx:c_end])

            # Compute overlap_before: the tokens from previous child that
            # overlap into this one
            if c_idx > 0:
                overlap_start = max(c_idx - overlap_size, 0)
                overlap_before = _words_to_text(words[overlap_start:c_idx])
            else:
                overlap_before = ""

            # overlap_after is filled in a second pass once we know the next child
            child_id = f"{parent_id}_c{position}_{uuid.uuid4().hex[:8]}"
            child = ChildChunk(
                text=child_text,
                chunk_id=child_id,
                parent_id=parent_id,
                position=position,
                overlap_before=overlap_before,
                overlap_after="",  # filled below
            )
            children.append(child)
            position += 1
            c_idx += step

            # Avoid creating a tiny trailing chunk that is all overlap
            if c_idx < len(words) and (len(words) - c_idx) < overlap_size:
                # Extend current child to end
                child.text = _words_to_text(words[c_idx - step: len(words)])
                c_idx = len(words)

        # Second pass: fill overlap_after
        for i in range(len(children) - 1):
            next_words = _split_into_words(children[i + 1].text)
            overlap_after_tokens = next_words[:overlap_size]
            children[i].overlap_after = _words_to_text(overlap_after_tokens)

        parent.children = children
        parents.append(parent)

    return parents


def expand_to_parent(
    child_chunks: List[ChildChunk],
    parent_chunks: List[ParentChunk],
) -> List[str]:
    """Given retrieved child chunks, expand to their parent chunks (deduplicated).

    If multiple children from the same parent are retrieved, the parent text
    is returned only once.  Order follows the first occurrence of each parent.
    """
    parent_lookup = {p.chunk_id: p.text for p in parent_chunks}
    seen: set = set()
    result: List[str] = []

    for child in child_chunks:
        pid = child.parent_id
        if pid not in seen and pid in parent_lookup:
            seen.add(pid)
            result.append(parent_lookup[pid])

    return result
