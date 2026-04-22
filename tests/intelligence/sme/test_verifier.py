"""Tests for SMEVerifier — one representative case per check.

Public API is the canonical ERRATA §3 surface: ``verify(item, ctx)`` and
``verify_batch(items, ctx)`` returning :class:`Verdict` instances.
"""
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef
from src.intelligence.sme.verifier import (
    SMEVerifier,
    VerifierChunkStore,
    VerifierContext,
)


def _item(**over):
    base = dict(
        item_id="i1",
        artifact_type="insight",
        subscription_id="sub_a",
        profile_id="prof_x",
        text="Revenue rose 12% in Q3.",
        evidence=[
            EvidenceRef(
                doc_id="d1", chunk_id="c1", quote="Q3 revenue up 12%"
            )
        ],
        confidence=0.9,
    )
    base.update(over)
    return ArtifactItem(**base)


@pytest.fixture
def cs():
    store = MagicMock(spec=VerifierChunkStore)
    store.chunk_exists.return_value = True
    store.chunk_text.return_value = "Q3 revenue up 12% year over year."
    return store


@pytest.fixture
def ctx():
    return VerifierContext(subscription_id="sub_a", profile_id="prof_x")


@pytest.fixture
def v(cs):
    return SMEVerifier(chunk_store=cs, max_inference_hops=3)


def test_check1_evidence_presence(v, ctx):
    r = v.verify(_item(evidence=[]), ctx)
    assert not r.passed
    assert r.failing_check == "evidence_presence"
    assert r.drop_reason


def test_check2_chunk_missing(cs, v, ctx):
    cs.chunk_exists.return_value = False
    r = v.verify(_item(), ctx)
    assert not r.passed
    assert r.failing_check == "evidence_validity"


def test_check2_text_not_substantively_present(cs, v, ctx):
    cs.chunk_text.return_value = "Unrelated weather text."
    r = v.verify(_item(text="Revenue rose 12% in Q3."), ctx)
    assert not r.passed
    assert r.failing_check == "evidence_validity"


def test_check3_inference_provenance_exceeds_max_hops(v, ctx):
    r = v.verify(_item(inference_path=[{"a": 1}] * 4), ctx)
    assert not r.passed
    assert r.failing_check == "inference_provenance"


def test_check4_rollback_single_source(v, ctx):
    r = v.verify(_item(confidence=0.9), ctx)
    assert r.passed
    assert r.adjusted_item.confidence <= 0.6


def test_check4_passes_with_two_sources(cs, v, ctx):
    # Make chunk_text return different content per chunk so both quotes find
    # their source text in the right chunk (otherwise check 2 would fail on
    # the second cited quote).
    cs.chunk_text.side_effect = lambda doc_id, chunk_id: {
        ("d1", "c1"): "Q3 revenue up 12% year over year.",
        ("d2", "c7"): "Q3 sales rose sharply in the quarter.",
    }[(doc_id, chunk_id)]
    r = v.verify(
        _item(
            confidence=0.9,
            evidence=[
                EvidenceRef(doc_id="d1", chunk_id="c1", quote="Q3 revenue up 12%"),
                EvidenceRef(doc_id="d2", chunk_id="c7", quote="Q3 sales rose"),
            ],
        ),
        ctx,
    )
    assert r.passed
    assert r.adjusted_item.confidence == pytest.approx(0.9)


def test_check5_contradiction_drops_lower_confidence(v, ctx):
    batch = [
        _item(
            item_id="i1",
            text="Q3 revenue rose.",
            confidence=0.95,
            evidence=[
                EvidenceRef(doc_id="d1", chunk_id="c1"),
                EvidenceRef(doc_id="d2", chunk_id="c2"),
            ],
        ),
        _item(item_id="i2", text="Q3 revenue fell.", confidence=0.5),
    ]
    vs = v.verify_batch(batch, ctx)
    # Order preserved — first input is first output.
    assert [x.item_id for x in vs] == ["i1", "i2"]
    assert [x.item_id for x in vs if x.passed] == ["i1"]
    failed = next(x for x in vs if not x.passed)
    assert failed.failing_check == "contradiction"


def test_ctx_overrides_max_hops(cs):
    # Verifier default allows 3 hops; ctx overrides to 2 so path of len 3 fails.
    v = SMEVerifier(chunk_store=cs, max_inference_hops=5)
    ctx = VerifierContext(max_inference_hops=2)
    r = v.verify(_item(inference_path=[{}, {}, {}]), ctx)
    assert not r.passed
    assert r.failing_check == "inference_provenance"


def test_conflict_tagged_items_bypass_contradiction(v, ctx):
    # An item tagged `conflict` is expected to contradict others — do not drop.
    batch = [
        _item(
            item_id="i1",
            text="Q3 revenue rose.",
            confidence=0.95,
            evidence=[
                EvidenceRef(doc_id="d1", chunk_id="c1"),
                EvidenceRef(doc_id="d2", chunk_id="c2"),
            ],
        ),
        _item(
            item_id="i2",
            text="Q3 revenue fell.",
            confidence=0.5,
            domain_tags=["conflict"],
        ),
    ]
    vs = v.verify_batch(batch, ctx)
    assert [x.passed for x in vs] == [True, True]
