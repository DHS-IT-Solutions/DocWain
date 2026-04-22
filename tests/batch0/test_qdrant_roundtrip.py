"""Qdrant roundtrip: write a chunk via payload_builder, read it via the
retriever's filter builder — asserts the read-side and write-side agree on
the indexed flat payload keys (chunk_id, resolution, section_kind, page,
source_name, doc_domain, etc.).

This test is the regression guard for commit b0c7211 — every subsequent
batch must keep it green.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from src.embedding.payload_builder import build_enriched_payload

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "invoice_fixture.json"

# Payload keys the retriever's FieldCondition filters rely on.
REQUIRED_FLAT_KEYS = {
    "subscription_id",
    "profile_id",
    "document_id",
    "chunk_id",
    "resolution",
    "chunk_kind",
    "section_id",
    "section_kind",
    "page",
    "source_name",
    "doc_domain",
    "embed_pipeline_version",
}


@pytest.fixture(scope="module")
def fixture_doc():
    return json.loads(FIXTURE.read_text())


def _build_payload(fixture_doc, chunk_idx):
    chunk = fixture_doc["chunks"][chunk_idx]
    return build_enriched_payload(
        chunk=chunk,
        chunk_index=chunk_idx,
        document_id=fixture_doc["document_id"],
        subscription_id=fixture_doc["subscription_id"],
        profile_id=fixture_doc["profile_id"],
        extraction_data={"entities": []},
        screening_summary={"entity_scores": {}, "domain_tags": [], "doc_category": "invoice"},
        source_name=fixture_doc["source_name"],
        doc_domain=fixture_doc["doc_domain"],
    )


def test_payload_has_all_indexed_flat_keys(fixture_doc):
    payload = _build_payload(fixture_doc, 0)
    missing = REQUIRED_FLAT_KEYS - set(payload.keys())
    assert not missing, f"Payload missing indexed keys: {missing}"


def test_payload_values_match_input(fixture_doc):
    payload = _build_payload(fixture_doc, 0)
    assert payload["subscription_id"] == fixture_doc["subscription_id"]
    assert payload["profile_id"] == fixture_doc["profile_id"]
    assert payload["document_id"] == fixture_doc["document_id"]
    assert payload["resolution"] == "chunk"
    assert payload["section_kind"] == fixture_doc["chunks"][0]["section"]["kind"]
    assert payload["source_name"] == fixture_doc["source_name"]
    assert payload["doc_domain"] == fixture_doc["doc_domain"]


def test_retriever_filter_uses_only_indexed_keys(fixture_doc):
    """The retriever's _build_filter must only reference keys present in
    the payload. Catches silent drift where the writer emits `document_id`
    but the reader filters on `doc_id`.
    """
    from src.retrieval.retriever import UnifiedRetriever

    class _StubQdrant:
        def collection_exists(self, _name):
            return True

    class _StubEmbedder:
        def encode(self, texts):
            return [[0.0] * 8 for _ in texts]

    r = UnifiedRetriever(qdrant_client=_StubQdrant(), embedder=_StubEmbedder())
    qfilter = r._build_filter(
        subscription_id=fixture_doc["subscription_id"],
        profile_id=fixture_doc["profile_id"],
        document_ids=[fixture_doc["document_id"]],
        chunks_only=True,
    )
    payload = _build_payload(fixture_doc, 0)
    for condition in qfilter.must:
        key = condition.key
        assert key in payload, (
            f"Retriever filter references key '{key}' which the writer "
            f"does not emit (payload keys: {sorted(payload.keys())})"
        )


def test_intelligence_retrieval_uses_only_indexed_keys(fixture_doc):
    """src/intelligence/retrieval.py must also only use keys the writer emits."""
    import inspect

    from src.intelligence import retrieval as intel_retrieval

    source = inspect.getsource(intel_retrieval)
    payload = _build_payload(fixture_doc, 0)
    # Any qdrant FieldCondition(key="X", ...) in the intelligence module
    # must use a key from the writer's payload. Scan the source for
    # obvious references to legacy names.
    FORBIDDEN = {"doc_id", "chunk.id", "section.id", "provenance.page_start"}
    for name in FORBIDDEN:
        if f'"{name}"' in source or f"'{name}'" in source:
            # Legacy key found — check it's either defensive (uses .get with
            # a fallback to an indexed key) or a bug we must fix.
            raise AssertionError(
                f"src/intelligence/retrieval.py references legacy payload "
                f"key '{name}' — rewrite to use an indexed flat key from "
                f"{sorted(payload.keys())}"
            )
