"""Tests for artifact models (ArtifactItem + EvidenceRef)."""
import pytest
from pydantic import ValidationError

from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef


def _kwargs(**over):
    base = dict(
        item_id="i1",
        artifact_type="insight",
        subscription_id="sub_a",
        profile_id="prof_x",
        text="Revenue rose 12% QoQ.",
        evidence=[EvidenceRef(doc_id="d1", chunk_id="c1")],
        confidence=0.85,
    )
    base.update(over)
    return base


def test_valid_artifact_item():
    item = ArtifactItem(**_kwargs())
    assert item.item_id == "i1"
    assert item.artifact_type == "insight"
    assert item.evidence[0].doc_id == "d1"
    assert item.inference_path == []
    assert item.domain_tags == []


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        ArtifactItem(**_kwargs(bogus=True))


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        ArtifactItem(**_kwargs(confidence=1.5))
    with pytest.raises(ValidationError):
        ArtifactItem(**_kwargs(confidence=-0.1))


def test_evidence_ref_frozen():
    ref = EvidenceRef(doc_id="d1", chunk_id="c1", quote="abc")
    with pytest.raises(ValidationError):
        ref.doc_id = "d2"
