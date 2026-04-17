"""Verify ingest_payloads encodes sparse vectors into ChunkRecord.sparse_vector."""
from unittest.mock import MagicMock, patch

from src.embedding.pipeline import qdrant_ingestion


def test_ingest_payloads_populates_sparse_vector():
    raw_payloads = [
        {
            "subscription_id": "sub-1",
            "profile_id": "p-1",
            "document_id": "doc-1",
            "canonical_text": "Invoice INV-42 total $9,000.00",
            "embedding_text": "Invoice INV-42 total $9,000.00",
            "metadata": {"source_file": "inv42.pdf"},
        }
    ]

    fake_sparse_encoder = MagicMock()
    fake_sparse_encoder.encode.return_value = {"indices": [1, 5, 7], "values": [0.3, 0.9, 0.2]}

    fake_vector_store = MagicMock()

    captured_records = []

    def capture(collection_name, records, batch_size):
        captured_records.extend(records)
        return 1

    fake_vector_store.upsert_records.side_effect = capture
    fake_vector_store.ensure_collection = MagicMock()

    with patch.object(qdrant_ingestion, "QdrantVectorStore", return_value=fake_vector_store), \
         patch.object(qdrant_ingestion, "_ollama_embed", return_value=[[0.0] * 1024]), \
         patch("src.api.rag_state.get_app_state") as mock_get_state:
        mock_get_state.return_value = MagicMock(sparse_encoder=fake_sparse_encoder)
        qdrant_ingestion.ingest_payloads(raw_payloads)

    assert len(captured_records) == 1
    rec = captured_records[0]
    assert rec.sparse_vector is not None
    # sparse_to_qdrant returns a SparseVector-like object with indices and values attrs
    assert list(rec.sparse_vector.indices) == [1, 5, 7]
    assert list(rec.sparse_vector.values) == [0.3, 0.9, 0.2]


def test_ingest_payloads_graceful_when_sparse_encoder_missing():
    raw_payloads = [
        {
            "subscription_id": "sub-1",
            "profile_id": "p-1",
            "document_id": "doc-1",
            "canonical_text": "text content",
            "embedding_text": "text content",
            "metadata": {"source_file": "f.pdf"},
        }
    ]

    fake_vector_store = MagicMock()
    captured_records = []
    fake_vector_store.upsert_records.side_effect = lambda c, r, batch_size: captured_records.extend(r) or 1
    fake_vector_store.ensure_collection = MagicMock()

    with patch.object(qdrant_ingestion, "QdrantVectorStore", return_value=fake_vector_store), \
         patch.object(qdrant_ingestion, "_ollama_embed", return_value=[[0.0] * 1024]), \
         patch("src.api.rag_state.get_app_state") as mock_get_state:
        mock_get_state.return_value = MagicMock(sparse_encoder=None)
        qdrant_ingestion.ingest_payloads(raw_payloads)

    assert len(captured_records) == 1
    assert captured_records[0].sparse_vector is None


def test_ingest_payloads_graceful_when_app_state_missing():
    """Test path: running without a live AppState singleton (e.g. CLI or tests)."""
    raw_payloads = [
        {
            "subscription_id": "sub-1",
            "profile_id": "p-1",
            "document_id": "doc-1",
            "canonical_text": "text",
            "embedding_text": "text",
            "metadata": {"source_file": "f.pdf"},
        }
    ]

    fake_vector_store = MagicMock()
    captured_records = []
    fake_vector_store.upsert_records.side_effect = lambda c, r, batch_size: captured_records.extend(r) or 1
    fake_vector_store.ensure_collection = MagicMock()

    with patch.object(qdrant_ingestion, "QdrantVectorStore", return_value=fake_vector_store), \
         patch.object(qdrant_ingestion, "_ollama_embed", return_value=[[0.0] * 1024]), \
         patch("src.api.rag_state.get_app_state", return_value=None):
        qdrant_ingestion.ingest_payloads(raw_payloads)

    assert captured_records[0].sparse_vector is None
