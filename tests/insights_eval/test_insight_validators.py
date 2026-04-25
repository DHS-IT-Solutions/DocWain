import pytest

from src.intelligence.insights.schema import Insight, EvidenceSpan, KbRef
from src.intelligence.insights.validators import (
    require_doc_evidence,
    require_body_grounded,
    compute_dedup_key,
    CitationViolation,
    BodySeparationViolation,
)


def _doc_span(doc_id="DOC-1", quote="hello"):
    return EvidenceSpan(
        document_id=doc_id, page=1, char_start=0, char_end=len(quote), quote=quote
    )


def _make(spans, kb_refs=None) -> Insight:
    return Insight(
        insight_id="i-1",
        profile_id="p-1",
        subscription_id="s-1",
        document_ids=["DOC-1"],
        domain="generic",
        insight_type="anomaly",
        headline="x",
        body="y derivable from quote: hello",
        evidence_doc_spans=spans,
        confidence=0.5,
        severity="notice",
        adapter_version="generic@1.0",
        external_kb_refs=kb_refs or [],
    )


def test_passes_with_at_least_one_span():
    insight = _make([_doc_span()])
    require_doc_evidence(insight)


def test_rejects_zero_spans():
    insight = _make([_doc_span()])
    insight.evidence_doc_spans = []  # mutate to simulate malformed write
    with pytest.raises(CitationViolation):
        require_doc_evidence(insight)


def test_rejects_kb_refs_without_doc_spans():
    insight = Insight(
        insight_id="i-2",
        profile_id="p-1",
        subscription_id="s-1",
        document_ids=["DOC-1"],
        domain="generic",
        insight_type="anomaly",
        headline="x",
        body="y",
        evidence_doc_spans=[_doc_span()],
        confidence=0.5,
        severity="notice",
        adapter_version="generic@1.0",
        external_kb_refs=[KbRef(kb_id="k1", ref="r1")],
    )
    insight.evidence_doc_spans = []
    with pytest.raises(CitationViolation):
        require_doc_evidence(insight)


def test_body_grounded_passes_when_quotes_overlap():
    span = EvidenceSpan(
        document_id="DOC-1", page=1, char_start=0, char_end=22,
        quote="The patient has Type 2 Diabetes.",
    )
    insight = _make([span])
    insight.body = "Patient diagnosed with Type 2 Diabetes."
    require_body_grounded(insight)


def test_body_grounded_rejects_unsupported_external_claims():
    span = EvidenceSpan(
        document_id="DOC-1", page=1, char_start=0, char_end=10,
        quote="Test data",
    )
    insight = _make([span])
    insight.body = "Patient is at risk of diabetic ketoacidosis"
    with pytest.raises(BodySeparationViolation):
        require_body_grounded(insight)


def test_body_grounded_passes_with_partial_overlap():
    span = EvidenceSpan(
        document_id="DOC-1", page=1, char_start=0, char_end=40,
        quote="Excludes: flood damage, earthquake, racing events.",
    )
    insight = _make([span])
    insight.body = "The policy excludes flood damage and earthquake coverage."
    require_body_grounded(insight)


def test_dedup_key_stable_for_same_inputs():
    insight = _make([_doc_span()])
    insight.profile_id = "p-1"
    insight.document_ids = ["DOC-1"]
    insight.insight_type = "anomaly"
    insight.headline = "Test headline"
    k1 = compute_dedup_key(insight)
    k2 = compute_dedup_key(insight)
    assert k1 == k2


def test_dedup_key_changes_with_headline():
    a = _make([_doc_span()])
    b = _make([_doc_span()])
    a.headline = "headline A"
    b.headline = "headline B"
    assert compute_dedup_key(a) != compute_dedup_key(b)


def test_dedup_key_independent_of_document_id_order():
    a = _make([_doc_span("D1"), _doc_span("D2")])
    b = _make([_doc_span("D2"), _doc_span("D1")])
    a.document_ids = ["D1", "D2"]
    b.document_ids = ["D2", "D1"]
    a.headline = b.headline = "same"
    assert compute_dedup_key(a) == compute_dedup_key(b)
