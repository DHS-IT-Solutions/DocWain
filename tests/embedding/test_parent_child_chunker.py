"""Tests for parent-child chunker with overlap."""

import pytest

from src.embedding.chunking.parent_child_chunker import (
    ChildChunk,
    ParentChunk,
    chunk_with_parents,
    expand_to_parent,
)


def test_basic_parent_child_splitting():
    """Parents are created and children reference back to their parent."""
    text = "Section 1. " + "This is content. " * 100 + "Section 2. " + "More content here. " * 100

    parents = chunk_with_parents(
        text, document_id="doc1", parent_size=400, child_size=100, overlap_size=20
    )

    assert len(parents) >= 2
    for parent in parents:
        assert len(parent.children) >= 1
        assert all(c.parent_id == parent.chunk_id for c in parent.children)


def test_children_have_overlap():
    """Consecutive children share overlap text."""
    text = "word " * 500

    parents = chunk_with_parents(
        text, document_id="doc1", parent_size=300, child_size=80, overlap_size=20
    )

    parent = parents[0]
    if len(parent.children) >= 2:
        child1 = parent.children[0]
        child2 = parent.children[1]
        # Second child should have overlap text from first
        assert len(child2.overlap_before) > 0
        # First child should have overlap_after pointing into second
        assert len(child1.overlap_after) > 0


def test_expand_to_parent_deduplicates():
    """Multiple children from the same parent yield one parent text."""
    text = "Important content. " * 200

    parents = chunk_with_parents(
        text, document_id="doc1", parent_size=400, child_size=80, overlap_size=20
    )

    # Simulate retrieving 2 children from same parent
    if parents and len(parents[0].children) >= 2:
        retrieved = [parents[0].children[0], parents[0].children[1]]
        expanded = expand_to_parent(retrieved, parents)
        assert len(expanded) == 1  # deduplicated to single parent


def test_child_ids_are_unique():
    """Every child chunk has a globally unique ID."""
    text = "Content " * 300

    parents = chunk_with_parents(
        text, document_id="doc1", parent_size=300, child_size=80
    )

    all_ids = [c.chunk_id for p in parents for c in p.children]
    assert len(all_ids) == len(set(all_ids))


def test_section_boundaries_respected():
    """Section boundaries cause parent splits at the specified positions."""
    section1 = "First section content. " * 50
    section2 = "Second section content. " * 50
    text = section1 + section2
    boundary = len(section1)

    parents = chunk_with_parents(
        text,
        document_id="doc1",
        parent_size=800,
        child_size=100,
        section_boundaries=[boundary],
    )

    assert len(parents) >= 2


def test_empty_text_returns_empty():
    """Empty or whitespace-only text produces no chunks."""
    assert chunk_with_parents("", document_id="doc1") == []
    assert chunk_with_parents("   ", document_id="doc1") == []


def test_expand_to_parent_multiple_parents():
    """Children from different parents expand to multiple parent texts."""
    text = "Alpha content. " * 200 + "Beta content. " * 200

    parents = chunk_with_parents(
        text, document_id="doc2", parent_size=250, child_size=60, overlap_size=10
    )

    assert len(parents) >= 2
    # Pick one child from each parent
    retrieved = [parents[0].children[0], parents[-1].children[0]]
    expanded = expand_to_parent(retrieved, parents)
    assert len(expanded) == 2


def test_first_child_has_no_overlap_before():
    """The first child in a parent should have empty overlap_before."""
    text = "Some text here. " * 100

    parents = chunk_with_parents(
        text, document_id="doc1", parent_size=500, child_size=50, overlap_size=15
    )

    first_child = parents[0].children[0]
    assert first_child.overlap_before == ""
    assert first_child.position == 0


def test_last_child_has_no_overlap_after():
    """The last child in a parent should have empty overlap_after."""
    text = "Some text here. " * 100

    parents = chunk_with_parents(
        text, document_id="doc1", parent_size=500, child_size=50, overlap_size=15
    )

    last_child = parents[0].children[-1]
    assert last_child.overlap_after == ""
