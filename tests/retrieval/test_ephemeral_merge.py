"""Lean tests for ephemeral_merge."""
from __future__ import annotations

from types import SimpleNamespace

from src.agent.url_case_selector import CaseSelection
from src.retrieval.ephemeral_merge import merge_ephemeral
from src.tools.url_ephemeral_source import EphemeralChunk


def _profile_chunk(doc_id: str, text: str, score: float):
    return SimpleNamespace(
        document_id=doc_id,
        text=text,
        score=score,
        chunk_id=f"{doc_id}_c",
        metadata={"provenance": "profile"},
    )


def _ephemeral(url: str, text: str) -> EphemeralChunk:
    return EphemeralChunk(
        text=text,
        metadata={
            "ephemeral": True,
            "source_url": url,
            "chunk_index": 0,
        },
        embedding=[0.0] * 4,
    )


def test_supplementary_appends_ephemeral_after_profile():
    profile = [_profile_chunk(f"d{i}", f"t{i}", 0.9 - 0.01 * i) for i in range(3)]
    ephemeral = [_ephemeral("https://a.example/", "u1")]
    merged = merge_ephemeral(profile, ephemeral, case=CaseSelection.SUPPLEMENTARY)
    assert merged[:3] == profile
    assert merged[3].metadata["ephemeral"] is True
    assert merged[3].metadata["provenance"] == "ephemeral_url"


def test_primary_places_ephemeral_first():
    profile = [_profile_chunk("d1", "t1", 0.8)]
    ephemeral = [_ephemeral("https://a/", "u1"), _ephemeral("https://a/", "u2")]
    merged = merge_ephemeral(profile, ephemeral, case=CaseSelection.PRIMARY)
    assert merged[0].metadata["ephemeral"] is True
    assert merged[1].metadata["ephemeral"] is True
    assert merged[2:] == profile


def test_none_case_returns_profile_only_even_if_ephemeral_present():
    profile = [_profile_chunk("d1", "t", 0.9)]
    ephemeral = [_ephemeral("https://a/", "u1")]
    merged = merge_ephemeral(profile, ephemeral, case=CaseSelection.NONE)
    assert merged == profile


def test_empty_ephemeral_returns_profile_unchanged():
    profile = [_profile_chunk("d1", "t", 0.9)]
    merged = merge_ephemeral(profile, [], case=CaseSelection.SUPPLEMENTARY)
    assert merged == profile
