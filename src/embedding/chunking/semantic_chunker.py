"""Semantic chunker: splits document text at meaningful boundaries.

Supports heading-based, paragraph-based, and hierarchical chunking.
Tables (lines containing |...|) are never split mid-block.
"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional

# Pattern: section/chapter headings or numbered headings (e.g. "1.", "1.2", "Chapter 3")
_HEADING_RE = re.compile(
    r"^(?:section\b|chapter\b|appendix\b|\d+(?:\.\d+)*\.?\s+\S|\d+\.\s+\S)",
    re.IGNORECASE,
)

# A table row contains at least one pipe character with content on both sides
_TABLE_ROW_RE = re.compile(r"\|.*\|")


def _is_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line.strip()))


def _is_table_row(line: str) -> bool:
    return bool(_TABLE_ROW_RE.search(line))


def _split_preserving_tables(text: str, max_chars: int) -> List[str]:
    """Split *text* into parts no larger than *max_chars*, never breaking mid-table.

    When the text has no newlines (a single long line of prose), falls back to
    word-boundary splitting so that oversized chunks are still divided.
    """
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines(keepends=True)

    # Single-line text: fall back to word-boundary splitting
    if len(lines) <= 1:
        return _split_on_words(text, max_chars)

    parts: List[str] = []
    current_lines: List[str] = []
    current_len = 0
    in_table = False

    for line in lines:
        line_is_table = _is_table_row(line)

        if line_is_table:
            in_table = True
        else:
            if in_table:
                # Leaving a table block — flush table + current together
                in_table = False

        projected = current_len + len(line)

        if projected > max_chars and current_lines and not in_table:
            # Safe to split here (not mid-table)
            parts.append("".join(current_lines).strip())
            current_lines = [line]
            current_len = len(line)
        else:
            current_lines.append(line)
            current_len += len(line)

    if current_lines:
        parts.append("".join(current_lines).strip())

    return [p for p in parts if p]


def _split_on_words(text: str, max_chars: int) -> List[str]:
    """Split prose text at word boundaries to stay within *max_chars*."""
    words = text.split()
    parts: List[str] = []
    current_words: List[str] = []
    current_len = 0

    for word in words:
        # +1 for the space separator
        needed = current_len + len(word) + (1 if current_words else 0)
        if needed > max_chars and current_words:
            parts.append(" ".join(current_words))
            current_words = [word]
            current_len = len(word)
        else:
            current_words.append(word)
            current_len = needed

    if current_words:
        parts.append(" ".join(current_words))

    return [p for p in parts if p]


class SemanticChunker:
    """Splits document text into semantically meaningful chunks.

    Parameters
    ----------
    min_chunk_chars:
        Chunks shorter than this are merged into the previous chunk.
    max_chunk_chars:
        Chunks longer than this are sub-divided (respecting table boundaries).
    """

    def __init__(self, min_chunk_chars: int = 100, max_chunk_chars: int = 2000) -> None:
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        *,
        hierarchical: bool = False,
        doc_id: str = "",
    ) -> List[dict]:
        """Chunk *text* and return a list of chunk dicts.

        Each dict contains:
            text, section_title, chunk_index, level, parent_chunk_id
        """
        if not text or not text.strip():
            return []

        raw_sections = self._split_into_sections(text)
        raw_sections = self._merge_small(raw_sections)
        chunks = self._flatten_large(raw_sections)
        return self._build_output(chunks, hierarchical=hierarchical, doc_id=doc_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_sections(self, text: str) -> List[dict]:
        """Split text at heading boundaries; fall back to blank lines."""
        lines = text.splitlines()
        has_headings = any(_is_heading(ln) for ln in lines if ln.strip())

        if has_headings:
            return self._split_at_headings(lines)
        return self._split_at_blank_lines(text)

    def _split_at_headings(self, lines: List[str]) -> List[dict]:
        sections: List[dict] = []
        current_title = ""
        current_lines: List[str] = []

        def _flush(title: str, body_lines: List[str]) -> None:
            body = "\n".join(body_lines).strip()
            if body:
                sections.append({"title": title, "text": body, "level": "section"})

        for line in lines:
            if _is_heading(line.strip()) and line.strip():
                _flush(current_title, current_lines)
                current_title = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

        _flush(current_title, current_lines)
        return sections

    def _split_at_blank_lines(self, text: str) -> List[dict]:
        paragraphs = re.split(r"\n\s*\n", text)
        sections: List[dict] = []
        for para in paragraphs:
            para = para.strip()
            if para:
                sections.append({"title": "", "text": para, "level": "paragraph"})
        return sections

    def _merge_small(self, sections: List[dict]) -> List[dict]:
        """Merge sections shorter than min_chunk_chars into the previous one."""
        merged: List[dict] = []
        for sec in sections:
            if (
                merged
                and len(sec["text"]) < self.min_chunk_chars
                and not _is_heading(sec["text"].splitlines()[0] if sec["text"] else "")
            ):
                prev = merged[-1]
                sep = "\n\n"
                prev["text"] = prev["text"] + sep + sec["text"]
                # Keep the primary title from the previous section
            else:
                merged.append(dict(sec))
        return merged

    def _flatten_large(self, sections: List[dict]) -> List[dict]:
        """Sub-divide sections larger than max_chunk_chars (no mid-table splits)."""
        result: List[dict] = []
        for sec in sections:
            if len(sec["text"]) <= self.max_chunk_chars:
                result.append(sec)
            else:
                parts = _split_preserving_tables(sec["text"], self.max_chunk_chars)
                for i, part in enumerate(parts):
                    result.append({
                        "title": sec["title"],
                        "text": part,
                        "level": "paragraph" if i > 0 else sec["level"],
                    })
        return result

    def _build_output(
        self,
        chunks: List[dict],
        *,
        hierarchical: bool,
        doc_id: str,
    ) -> List[dict]:
        output: List[dict] = []
        section_id: Optional[str] = None

        for idx, chunk in enumerate(chunks):
            level = chunk.get("level", "section")
            cid = f"{doc_id}:{idx}" if doc_id else str(idx)

            # Track the most recent section-level chunk as the parent
            if level == "section":
                section_id = cid

            parent_chunk_id = ""
            if hierarchical and level == "paragraph" and section_id and section_id != cid:
                parent_chunk_id = section_id

            output.append({
                "text": chunk["text"],
                "section_title": chunk.get("title", ""),
                "chunk_index": idx,
                "level": level,
                "parent_chunk_id": parent_chunk_id,
            })

        return output
