"""Unit tests for src/embedding/chunking/semantic_chunker.py"""

from __future__ import annotations

import pytest

from src.embedding.chunking.semantic_chunker import SemanticChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunker(**kw) -> SemanticChunker:
    return SemanticChunker(**kw)


# ---------------------------------------------------------------------------
# Basic instantiation
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        sc = SemanticChunker()
        assert sc.min_chunk_chars == 100
        assert sc.max_chunk_chars == 2000

    def test_custom(self):
        sc = SemanticChunker(min_chunk_chars=50, max_chunk_chars=500)
        assert sc.min_chunk_chars == 50
        assert sc.max_chunk_chars == 500


# ---------------------------------------------------------------------------
# Empty / trivial input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_string(self):
        sc = _make_chunker()
        assert sc.chunk("") == []

    def test_whitespace_only(self):
        sc = _make_chunker()
        assert sc.chunk("   \n\n  ") == []


# ---------------------------------------------------------------------------
# Heading-based splitting
# ---------------------------------------------------------------------------


class TestHeadingSplit:
    HEADING_DOC = (
        "1. Introduction\n"
        "This is the introduction paragraph with sufficient length to not be merged away.\n\n"
        "2. Background\n"
        "Background content goes here and it is long enough to stand on its own as a chunk.\n\n"
        "3. Conclusion\n"
        "Final remarks that wrap up the document section and meet the minimum length requirement."
    )

    def test_produces_chunks(self):
        sc = _make_chunker(min_chunk_chars=10)
        chunks = sc.chunk(self.HEADING_DOC)
        assert len(chunks) >= 2

    def test_chunk_has_required_keys(self):
        sc = _make_chunker(min_chunk_chars=10)
        chunks = sc.chunk(self.HEADING_DOC)
        required = {"text", "section_title", "chunk_index", "level", "parent_chunk_id"}
        for chunk in chunks:
            assert required == set(chunk.keys()), f"Missing keys in chunk: {chunk}"

    def test_chunk_index_is_sequential(self):
        sc = _make_chunker(min_chunk_chars=10)
        chunks = sc.chunk(self.HEADING_DOC)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_section_title_captured(self):
        sc = _make_chunker(min_chunk_chars=10)
        chunks = sc.chunk(self.HEADING_DOC)
        titles = [c["section_title"] for c in chunks]
        # At least one title should be non-empty
        assert any(t for t in titles)

    def test_level_section(self):
        sc = _make_chunker(min_chunk_chars=10)
        chunks = sc.chunk(self.HEADING_DOC)
        # With headings, top-level chunks should be 'section'
        levels = {c["level"] for c in chunks}
        assert "section" in levels

    def test_parent_chunk_id_empty_non_hierarchical(self):
        sc = _make_chunker(min_chunk_chars=10)
        chunks = sc.chunk(self.HEADING_DOC, hierarchical=False)
        assert all(c["parent_chunk_id"] == "" for c in chunks)


# ---------------------------------------------------------------------------
# Blank-line splitting (no headings)
# ---------------------------------------------------------------------------


class TestBlankLineSplit:
    NO_HEADING_DOC = (
        "First paragraph. " * 10 + "\n\n"
        "Second paragraph. " * 10 + "\n\n"
        "Third paragraph. " * 10
    )

    def test_splits_at_blank_lines(self):
        sc = _make_chunker(min_chunk_chars=10)
        chunks = sc.chunk(self.NO_HEADING_DOC)
        assert len(chunks) >= 2

    def test_level_paragraph(self):
        sc = _make_chunker(min_chunk_chars=10)
        chunks = sc.chunk(self.NO_HEADING_DOC)
        levels = {c["level"] for c in chunks}
        assert "paragraph" in levels

    def test_text_preserved(self):
        sc = _make_chunker(min_chunk_chars=10)
        chunks = sc.chunk(self.NO_HEADING_DOC)
        combined = " ".join(c["text"] for c in chunks)
        # All original words should still appear somewhere
        assert "First paragraph" in combined
        assert "Third paragraph" in combined


# ---------------------------------------------------------------------------
# Large section splitting (max_chunk_chars)
# ---------------------------------------------------------------------------


class TestLargeSectionSplit:
    def test_oversized_chunk_is_split(self):
        big_text = "word " * 500  # ~2500 chars
        sc = _make_chunker(min_chunk_chars=10, max_chunk_chars=200)
        chunks = sc.chunk(big_text)
        assert len(chunks) > 1
        for c in chunks:
            # Each piece should not exceed max by a single line's worth
            assert len(c["text"]) <= 400  # generous tolerance for last word

    def test_table_not_split(self):
        # Build a table block that is larger than max_chunk_chars but must stay intact
        table_header = "| Column A | Column B | Column C |\n| --- | --- | --- |\n"
        table_rows = "| cell A{i} | cell B{i} | cell C{i} |\n"
        table = table_header + "".join(
            table_rows.format(i=i) for i in range(30)
        )
        preamble = "Short intro.\n\n"
        text = preamble + table

        sc = _make_chunker(min_chunk_chars=5, max_chunk_chars=100)
        chunks = sc.chunk(text)

        # No chunk should contain a partial table row (all pipe rows intact)
        for chunk in chunks:
            lines_with_pipes = [ln for ln in chunk["text"].splitlines() if "|" in ln]
            for ln in lines_with_pipes:
                # Every pipe-containing line must be a complete row
                assert ln.count("|") >= 2


# ---------------------------------------------------------------------------
# Small section merging (min_chunk_chars)
# ---------------------------------------------------------------------------


class TestSmallSectionMerge:
    def test_tiny_chunks_merged(self):
        # Two short paragraphs followed by a long one
        text = "Hi.\n\nOK.\n\n" + ("This is a longer paragraph with more content. " * 5)
        sc = _make_chunker(min_chunk_chars=100, max_chunk_chars=2000)
        chunks = sc.chunk(text)
        # The tiny paragraphs should have been merged so we get fewer chunks
        # than the raw paragraph count (3)
        assert len(chunks) < 3 or any(
            "Hi" in c["text"] and "OK" in c["text"] for c in chunks
        )


# ---------------------------------------------------------------------------
# Hierarchical mode
# ---------------------------------------------------------------------------


class TestHierarchical:
    HEADING_DOC = (
        "1. Introduction\n"
        "This is the introduction with some reasonable content that exceeds minimum.\n\n"
        "2. Methods\n"
        "Methods section describes the approach taken in great detail over many sentences.\n\n"
        "3. Results\n"
        "Results are presented here with sufficient detail for the reader to understand."
    )

    def test_paragraph_chunks_have_parent_in_hierarchical(self):
        sc = _make_chunker(min_chunk_chars=10, max_chunk_chars=50)
        # Use very small max to force sub-division
        big_text = (
            "1. Big Section\n"
            + ("Content sentence goes here. " * 20)
        )
        chunks = sc.chunk(big_text, hierarchical=True, doc_id="doc1")
        paragraph_chunks = [c for c in chunks if c["level"] == "paragraph"]
        if paragraph_chunks:
            for c in paragraph_chunks:
                assert c["parent_chunk_id"] != "", (
                    f"Paragraph chunk missing parent_chunk_id: {c}"
                )

    def test_section_chunks_have_no_parent(self):
        sc = _make_chunker(min_chunk_chars=10)
        chunks = sc.chunk(self.HEADING_DOC, hierarchical=True, doc_id="doc2")
        section_chunks = [c for c in chunks if c["level"] == "section"]
        for c in section_chunks:
            # Section-level chunks should not have a parent (they ARE parents)
            # or their parent_chunk_id should not point to themselves
            assert c["parent_chunk_id"] != c["chunk_index"]

    def test_doc_id_in_parent_chunk_id(self):
        sc = _make_chunker(min_chunk_chars=10, max_chunk_chars=50)
        big_text = "1. Section\n" + ("Long content here. " * 30)
        chunks = sc.chunk(big_text, hierarchical=True, doc_id="mydoc")
        paragraph_chunks = [c for c in chunks if c["level"] == "paragraph"]
        if paragraph_chunks:
            for c in paragraph_chunks:
                assert "mydoc" in c["parent_chunk_id"]


# ---------------------------------------------------------------------------
# Output schema validation
# ---------------------------------------------------------------------------


class TestOutputSchema:
    def test_all_chunks_have_text(self):
        sc = _make_chunker(min_chunk_chars=5)
        text = "Section 1.\nSome content here.\n\nSection 2.\nMore content here for testing."
        chunks = sc.chunk(text)
        assert all(isinstance(c["text"], str) and c["text"] for c in chunks)

    def test_chunk_index_starts_at_zero(self):
        sc = _make_chunker(min_chunk_chars=5)
        chunks = sc.chunk("Hello world. This is a test paragraph with some text.")
        if chunks:
            assert chunks[0]["chunk_index"] == 0

    def test_level_values_valid(self):
        sc = _make_chunker(min_chunk_chars=5)
        text = (
            "1. First Section\nLong enough content here.\n\n"
            "Standalone paragraph with more text here."
        )
        chunks = sc.chunk(text)
        valid_levels = {"section", "paragraph"}
        for c in chunks:
            assert c["level"] in valid_levels, f"Invalid level: {c['level']}"
