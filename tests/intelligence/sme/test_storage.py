"""Tests for :class:`SMEArtifactStorage`.

Covers the canonical facade methods plus the ``persist_items`` convenience
wrapper. Blob / Qdrant / Neo4j are all mocked so we assert on call shape.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef
from src.intelligence.sme.storage import (
    SMEArtifactStorage,
    StorageDeps,
)


@pytest.fixture
def st() -> SMEArtifactStorage:
    return SMEArtifactStorage(
        StorageDeps(
            blob=MagicMock(),
            qdrant=MagicMock(),
            neo4j=MagicMock(),
        )
    )


def _insight(**overrides) -> ArtifactItem:
    base = dict(
        item_id="i1",
        artifact_type="insight",
        subscription_id="sub_a",
        profile_id="prof_x",
        text="Revenue rose 12% QoQ.",
        evidence=[EvidenceRef(doc_id="d1", chunk_id="c1")],
        confidence=0.75,
        domain_tags=["trend"],
    )
    base.update(overrides)
    return ArtifactItem(**base)


def test_persist_items_writes_blob_and_indexes_qdrant(st: SMEArtifactStorage) -> None:
    st.persist_items("sub_a", "prof_x", "insight", [_insight()], version=1)
    path, payload = st.deps.blob.write_text.call_args[0]
    assert path == "sme_artifacts/sub_a/prof_x/insight/1.json"
    body = json.loads(payload)
    assert body["subscription_id"] == "sub_a"
    assert body["profile_id"] == "prof_x"
    assert body["artifact_type"] == "insight"
    assert body["version"] == 1
    assert len(body["items"]) == 1
    # Qdrant point carries subscription + profile isolation keys.
    kw = st.deps.qdrant.upsert_points.call_args[1]
    assert kw["collection"] == "sme_artifacts_sub_a"
    point = kw["points"][0]
    assert point["payload"]["subscription_id"] == "sub_a"
    assert point["payload"]["profile_id"] == "prof_x"
    assert point["payload"]["synthesis_version"] == 1


def test_persist_items_writes_neo4j_only_for_kg_edge(st: SMEArtifactStorage) -> None:
    st.persist_items("s", "p", "insight", [_insight()], version=1)
    st.deps.neo4j.write_inferred_edges.assert_not_called()
    edge = _insight(
        item_id="e1",
        artifact_type="kg_edge",
        text="A indirectly funds B",
        metadata={
            "from_node": "a",
            "to_node": "b",
            "relation_type": "indirectly_funds",
        },
    )
    st.persist_items("s", "p", "kg_edge", [edge], version=2)
    st.deps.neo4j.write_inferred_edges.assert_called_once()
    payload = st.deps.neo4j.write_inferred_edges.call_args[0][0][0]
    assert payload["from_node"] == "a"
    assert payload["to_node"] == "b"
    assert payload["relation_type"] == "indirectly_funds"
    assert payload["subscription_id"] == "s"
    assert payload["profile_id"] == "p"


def test_put_snippet_writes_single_qdrant_point(st: SMEArtifactStorage) -> None:
    st.put_snippet("sub_a", "prof_x", _insight(), synthesis_version=3)
    kw = st.deps.qdrant.upsert_points.call_args[1]
    assert kw["collection"] == "sme_artifacts_sub_a"
    assert len(kw["points"]) == 1
    assert kw["points"][0]["payload"]["synthesis_version"] == 3


def test_put_canonical_writes_blob_only(st: SMEArtifactStorage) -> None:
    st.put_canonical("s", "p", "dossier", [_insight()], synthesis_version=5)
    path, payload = st.deps.blob.write_text.call_args[0]
    assert path == "sme_artifacts/s/p/dossier/5.json"
    assert json.loads(payload)["version"] == 5
    st.deps.qdrant.upsert_points.assert_not_called()


def test_put_manifest_writes_manifest_path(st: SMEArtifactStorage) -> None:
    st.put_manifest(
        "s",
        "p",
        {"synthesis_id": "s:p:7", "counts": {"dossier": 0}},
    )
    path, payload = st.deps.blob.write_text.call_args[0]
    assert path == "sme_artifacts/s/p/manifest/s:p:7.json"
    assert json.loads(payload)["counts"] == {"dossier": 0}


def test_put_manifest_rejects_missing_synthesis_id(st: SMEArtifactStorage) -> None:
    with pytest.raises(ValueError, match="synthesis_id"):
        st.put_manifest("s", "p", {"counts": {}})


def test_delete_version_clears_blob_and_qdrant(st: SMEArtifactStorage) -> None:
    st.delete_version("sub_a", "prof_x", "insight", version=1)
    st.deps.blob.delete.assert_called_once_with(
        "sme_artifacts/sub_a/prof_x/insight/1.json"
    )
    kw = st.deps.qdrant.delete_by_filter.call_args[1]
    assert kw["collection"] == "sme_artifacts_sub_a"
    must = kw["filter"]["must"]
    keys = {m["key"]: m["value"] for m in must}
    assert keys["subscription_id"] == "sub_a"
    assert keys["profile_id"] == "prof_x"
    assert keys["artifact_type"] == "insight"
    assert keys["synthesis_version"] == 1


def test_persist_items_empty_skips_qdrant(st: SMEArtifactStorage) -> None:
    st.persist_items("s", "p", "insight", [], version=1)
    st.deps.blob.write_text.assert_called_once()
    st.deps.qdrant.upsert_points.assert_not_called()
    st.deps.neo4j.write_inferred_edges.assert_not_called()
