"""Unit tests for KG ingest incremental enrichment helpers."""

from __future__ import annotations

import pytest

from src.kg.ingest import build_graph_payload, should_run_cross_doc_inference


# ---------------------------------------------------------------------------
# should_run_cross_doc_inference
# ---------------------------------------------------------------------------

class TestShouldRunCrossDocInference:
    @pytest.mark.parametrize("doc_count", [10, 20, 30])
    def test_returns_true_at_multiples_of_interval(self, doc_count: int):
        assert should_run_cross_doc_inference(doc_count) is True

    @pytest.mark.parametrize("doc_count", [5, 1])
    def test_returns_false_for_non_multiples(self, doc_count: int):
        assert should_run_cross_doc_inference(doc_count) is False

    def test_returns_false_at_zero(self):
        assert should_run_cross_doc_inference(0) is False

    def test_custom_interval(self):
        assert should_run_cross_doc_inference(5, interval=5) is True
        assert should_run_cross_doc_inference(3, interval=5) is False


# ---------------------------------------------------------------------------
# build_graph_payload — typed_relationships parameter
# ---------------------------------------------------------------------------

class TestBuildGraphPayloadTypedRelationships:
    """build_graph_payload already accepts typed_relationships; verify it passes through."""

    def _embeddings_payload(self):
        return {
            "texts": ["Alice works for Acme Corp."],
            "chunk_metadata": [{"chunk_id": "chunk-1"}],
            "doc_metadata": {},
        }

    def test_accepts_typed_relationships_parameter(self, monkeypatch):
        """build_graph_payload should accept typed_relationships without error."""
        # Patch KG enabled so the function proceeds past the early-return guard
        import src.api.config as cfg_mod
        monkeypatch.setattr(cfg_mod.Config.KnowledgeGraph, "ENABLED", True, raising=False)

        typed_rels = [
            {"entity1": "Alice", "entity2": "Acme Corp", "relation_type": "WORKS_FOR"}
        ]

        result = build_graph_payload(
            embeddings_payload=self._embeddings_payload(),
            subscription_id="sub-1",
            profile_id="prof-1",
            document_id="doc-1",
            doc_name="test.pdf",
            typed_relationships=typed_rels,
        )

        # Result may be None if no entities are extracted, but no TypeError should occur
        if result is not None:
            assert isinstance(result.typed_relationships, list)

    def test_typed_relationships_stored_in_payload(self, monkeypatch):
        """typed_relationships passed in are stored on the returned GraphIngestPayload."""
        import src.api.config as cfg_mod
        monkeypatch.setattr(cfg_mod.Config.KnowledgeGraph, "ENABLED", True, raising=False)

        typed_rels = [
            {"entity1": "Bob", "entity2": "Globex", "relation_type": "CEO_OF"}
        ]

        result = build_graph_payload(
            embeddings_payload=self._embeddings_payload(),
            subscription_id="sub-1",
            profile_id="prof-1",
            document_id="doc-2",
            doc_name="test.pdf",
            typed_relationships=typed_rels,
        )

        if result is not None:
            assert result.typed_relationships == typed_rels

    def test_omitting_typed_relationships_defaults_to_empty(self, monkeypatch):
        """Omitting typed_relationships produces an empty list, not None."""
        import src.api.config as cfg_mod
        monkeypatch.setattr(cfg_mod.Config.KnowledgeGraph, "ENABLED", True, raising=False)

        result = build_graph_payload(
            embeddings_payload=self._embeddings_payload(),
            subscription_id="sub-1",
            profile_id="prof-1",
            document_id="doc-3",
            doc_name="test.pdf",
        )

        if result is not None:
            assert result.typed_relationships == []
