"""Lean tests for the Phase 5 URL-citation and supplementary-prompt helpers."""
from __future__ import annotations

import pytest

from src.generation.prompts import (
    annotate_citation,
    build_supplementary_prompt,
)


def test_profile_citation_has_no_url():
    label = annotate_citation(doc_id="doc_1234", chunk_id="c_abcd1234")
    assert "http" not in label
    assert "doc_1234" in label


def test_url_citation_has_host_and_path():
    label = annotate_citation(source_url="https://docs.company.com/post/123")
    assert "docs.company.com" in label
    assert "/post" in label


def test_supplementary_prompt_mentions_primary_response_and_url_content():
    prompt = build_supplementary_prompt(
        primary_response="Q3 revenue rose 12%.",
        url_chunks=[
            {"text": "Article claims 15% growth in Q3.", "source_url": "https://ex/"},
        ],
    )
    assert "Q3 revenue rose 12%." in prompt
    assert "15% growth" in prompt
    assert "supplementary" in prompt.lower()
    assert "https://ex/" in prompt


def test_supplementary_prompt_rejects_empty_chunks():
    with pytest.raises(ValueError):
        build_supplementary_prompt(primary_response="x", url_chunks=[])
