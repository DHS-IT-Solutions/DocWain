"""Lean sanity tests for recommendation_grounding.py."""
from src.generation.recommendation_grounding import (
    GroundingReport,
    enforce_recommendation_grounding,
)


def _response(body: str) -> str:
    return (
        "## Executive summary\nSome summary.\n\n"
        "## Recommendations\n"
        f"{body}\n\n"
        "## Rationale & evidence\nSome evidence.\n\n"
        "## Assumptions & caveats\nSome caveats.\n\n"
        "## Evidence\n- q3_pl:c2\n"
    )


_BANK = [
    {
        "recommendation": "Renegotiate top-3 vendor contracts",
        "evidence": ["q3_pl:c2"],
    },
    {
        "recommendation": "Freeze hiring in non-revenue roles",
        "evidence": ["q3_hr:c5"],
    },
]


def test_passes_grounded_items_unchanged():
    body = (
        "1. Renegotiate top-3 vendor contracts [q3_pl:c2].\n"
        "2. Freeze hiring in non-revenue roles [q3_hr:c5].\n"
    )
    out, report = enforce_recommendation_grounding(
        _response(body), bank_entries=_BANK
    )
    assert isinstance(report, GroundingReport)
    assert "Renegotiate" in out
    assert "Freeze hiring" in out
    assert report.dropped_count == 0


def test_drops_ungrounded_item_and_appends_note():
    body = (
        "1. Renegotiate top-3 vendor contracts [q3_pl:c2].\n"
        "2. Launch a new business unit in EMEA.\n"
    )
    out, report = enforce_recommendation_grounding(
        _response(body), bank_entries=_BANK
    )
    assert "Launch a new business unit" not in out
    assert "Renegotiate" in out
    assert report.dropped_count == 1
    assert "could not be verified" in out


def test_keeps_bank_match_even_without_inline_citation():
    body = "1. Renegotiate top-3 vendor contracts.\n"
    out, report = enforce_recommendation_grounding(
        _response(body), bank_entries=_BANK
    )
    assert "Renegotiate" in out
    assert report.dropped_count == 0


def test_returns_unchanged_without_recommendations_section():
    resp = "## Executive summary\nNothing here.\n"
    out, report = enforce_recommendation_grounding(resp, bank_entries=_BANK)
    assert out == resp
    assert report.dropped_count == 0


def test_empty_bank_only_accepts_inline_citations():
    body = (
        "1. Do X [doc_a:c1].\n"
        "2. Do Y.\n"
    )
    out, report = enforce_recommendation_grounding(
        _response(body), bank_entries=[]
    )
    assert "Do X" in out
    assert "Do Y" not in out
    assert report.dropped_count == 1
