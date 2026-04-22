"""Lean tests for URL-as-prompt case selection."""
from __future__ import annotations

from src.agent.url_case_selector import (
    CaseSelection,
    CaseSelector,
    RetrievalSignal,
    select_case,
)


def _sig(artifacts: int = 0, high_sim: int = 0) -> RetrievalSignal:
    return RetrievalSignal(
        sme_artifact_count=artifacts, high_sim_chunk_count=high_sim,
    )


def test_no_urls_returns_none_case():
    assert select_case(
        cleaned_query="regular question",
        url_count=0,
        signal=_sig(),
    ) is CaseSelection.NONE


def test_strong_profile_always_supplementary():
    assert select_case(
        cleaned_query="summarize this",
        url_count=1,
        signal=_sig(artifacts=2),
    ) is CaseSelection.SUPPLEMENTARY
    assert select_case(
        cleaned_query="summarize this",
        url_count=1,
        signal=_sig(high_sim=5),
    ) is CaseSelection.SUPPLEMENTARY


def test_weak_profile_url_dominant_primary():
    assert select_case(
        cleaned_query="summarize this",
        url_count=1,
        signal=_sig(),
    ) is CaseSelection.PRIMARY
    assert select_case(
        cleaned_query="tl;dr",
        url_count=1,
        signal=_sig(),
    ) is CaseSelection.PRIMARY


def test_weak_profile_non_url_dominant_stays_supplementary():
    long_query = (
        "explain the accounting treatment for revenue recognition under "
        "asc 606 across quarterly filings and provide citations"
    )
    assert select_case(
        cleaned_query=long_query,
        url_count=1,
        signal=_sig(),
    ) is CaseSelection.SUPPLEMENTARY


def test_custom_thresholds_respected():
    selector = CaseSelector(strong_artifact_count=3)
    # Normally 1 artifact is strong; with cap=3 it becomes weak.
    decision = selector.select(
        cleaned_query="summarize this",
        url_count=1,
        signal=_sig(artifacts=1),
    )
    assert decision is CaseSelection.PRIMARY
