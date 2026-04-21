"""Lean sanity tests for src/generation/pack_summary.py."""
from src.generation.pack_summary import PackSummary
from src.retrieval.types import PackedItem


def _pi(text: str, **overrides) -> PackedItem:
    base = dict(
        text=text,
        provenance=((("d1", "c1"),) if "provenance" not in overrides else ()),
        layer="a",
        confidence=0.9,
        rerank_score=0.8,
        sme_backed=False,
        metadata={},
    )
    base.update(overrides)
    return PackedItem(**base)


def test_from_packed_items_empty_pack():
    s = PackSummary.from_packed_items([])
    assert s.total_chunks == 0
    assert s.distinct_docs == 0
    assert s.has_sme_artifacts is False
    assert s.bank_entries == ()
    assert s.evidence_items == ()
    assert s.insights == ()


def test_from_packed_items_splits_by_artifact_type():
    items = [
        _pi("Q3 revenue $5.3M", provenance=(("q3", "c1"),)),
        _pi(
            "QoQ growth 14%",
            provenance=(("q3", "c2"),),
            metadata={"artifact_type": "insight"},
            sme_backed=True,
        ),
        _pi(
            "Renegotiate top-3 vendor contracts",
            provenance=(("q3_pl", "c2"),),
            metadata={"artifact_type": "recommendation"},
            sme_backed=True,
        ),
    ]
    s = PackSummary.from_packed_items(items)
    assert s.total_chunks == 3
    # Two distinct doc ids: "q3" and "q3_pl"
    assert s.distinct_docs == 2
    assert s.has_sme_artifacts is True
    assert len(s.bank_entries) == 1
    assert (
        s.bank_entries[0]["recommendation"]
        == "Renegotiate top-3 vendor contracts"
    )
    assert s.bank_entries[0]["evidence"] == ["q3_pl:c2"]
    assert len(s.insights) == 1
    assert s.insights[0].text == "QoQ growth 14%"
    # Every item with provenance lands in evidence_items
    assert len(s.evidence_items) == 3


def test_sme_backed_without_artifact_type_still_flags_has_sme():
    items = [_pi("generic chunk", sme_backed=True, provenance=(("d1", "c1"),))]
    s = PackSummary.from_packed_items(items)
    assert s.has_sme_artifacts is True
    assert s.bank_entries == ()
    assert s.insights == ()
