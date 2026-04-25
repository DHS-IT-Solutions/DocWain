import pytest

from src.intelligence.insights.schema import (
    Insight,
    EvidenceSpan,
    KbRef,
    INSIGHT_TYPES,
    SEVERITIES,
    ACTION_TYPES,
)


def test_insight_types_complete():
    assert INSIGHT_TYPES == (
        "anomaly", "gap", "comparison", "scenario", "trend",
        "recommendation", "conflict", "projection", "next_action",
    )


def test_severities_ordered():
    assert SEVERITIES == ("info", "notice", "warn", "critical")


def test_action_types():
    assert ACTION_TYPES == ("artifact", "form_fill", "alert", "plan", "reminder")


def test_minimal_insight_construction():
    span = EvidenceSpan(
        document_id="DOC-1", page=1, char_start=0, char_end=10, quote="hello"
    )
    insight = Insight(
        insight_id="i-1",
        profile_id="p-1",
        subscription_id="s-1",
        document_ids=["DOC-1"],
        domain="generic",
        insight_type="anomaly",
        headline="Test headline",
        body="Body text grounded in doc",
        evidence_doc_spans=[span],
        confidence=0.9,
        severity="notice",
        adapter_version="generic@1.0",
    )
    assert insight.headline == "Test headline"
    assert insight.evidence_doc_spans[0].document_id == "DOC-1"
    assert insight.external_kb_refs == []


def test_invalid_insight_type_rejected():
    with pytest.raises(ValueError):
        Insight(
            insight_id="i-2",
            profile_id="p-1",
            subscription_id="s-1",
            document_ids=["DOC-1"],
            domain="generic",
            insight_type="bogus",
            headline="x",
            body="y",
            evidence_doc_spans=[
                EvidenceSpan(
                    document_id="DOC-1", page=1, char_start=0, char_end=1, quote="x"
                )
            ],
            confidence=0.5,
            severity="notice",
            adapter_version="generic@1.0",
        )


def test_invalid_severity_rejected():
    with pytest.raises(ValueError):
        Insight(
            insight_id="i-3",
            profile_id="p-1",
            subscription_id="s-1",
            document_ids=["DOC-1"],
            domain="generic",
            insight_type="anomaly",
            headline="x",
            body="y",
            evidence_doc_spans=[
                EvidenceSpan(document_id="DOC-1", page=1, char_start=0, char_end=1, quote="x")
            ],
            confidence=0.5,
            severity="meh",
            adapter_version="generic@1.0",
        )


def test_to_dict_round_trip():
    insight = Insight(
        insight_id="i-4",
        profile_id="p-1",
        subscription_id="s-1",
        document_ids=["DOC-1"],
        domain="insurance",
        insight_type="gap",
        headline="No flood coverage",
        body="The policy excludes flood damage. Flood damage exclusion is listed under exclusions.",
        evidence_doc_spans=[
            EvidenceSpan(document_id="DOC-1", page=1, char_start=100, char_end=130, quote="Excludes: flood damage")
        ],
        external_kb_refs=[KbRef(kb_id="insurance_taxonomy_v1", ref="exclusions/flood", label="Flood exclusion")],
        confidence=0.95,
        severity="warn",
        adapter_version="insurance@1.0",
    )
    d = insight.to_dict()
    assert d["insight_id"] == "i-4"
    assert d["insight_type"] == "gap"
    assert d["evidence_doc_spans"][0]["document_id"] == "DOC-1"
    assert d["external_kb_refs"][0]["kb_id"] == "insurance_taxonomy_v1"
