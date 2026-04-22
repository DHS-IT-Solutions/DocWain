"""Lock dense+sparse+RRF hybrid retrieval behaviour in UnifiedRetriever."""
from unittest.mock import MagicMock

from src.retrieval.retriever import UnifiedRetriever


def _make_point(chunk_id: str, text: str, doc_id: str, score: float):
    pt = MagicMock()
    pt.payload = {
        "canonical_text": text,
        "chunk": {"id": chunk_id, "type": "text"},
        "document_id": doc_id,
        "profile_id": "p-1",
        "source_name": f"{chunk_id}.pdf",
        "subscription_id": "sub-1",
    }
    pt.score = score
    return pt


def test_hybrid_retrieve_uses_sparse_when_encoder_present():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    # Dense result
    dense_result = MagicMock()
    dense_result.points = [
        _make_point("cd1", "dense result 1", "doc-1", 0.9),
        _make_point("cd2", "dense result 2", "doc-1", 0.8),
    ]
    # Sparse result
    sparse_result = MagicMock()
    sparse_result.points = [
        _make_point("cs1", "sparse result 1", "doc-2", 0.85),
        _make_point("cd1", "dense result 1", "doc-1", 0.7),  # overlap with dense
    ]

    def query_points_side_effect(**kwargs):
        using = kwargs.get("using")
        if using == "content_vector":
            return dense_result
        elif using == "keywords_vector":
            return sparse_result
        return MagicMock(points=[])

    fake_qdrant.query_points.side_effect = query_points_side_effect
    fake_qdrant.scroll.return_value = ([], None)

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]

    fake_sparse = MagicMock()
    fake_sparse.encode.return_value = {"indices": [1, 2], "values": [0.5, 0.7]}

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        sparse_encoder=fake_sparse,
    )
    result = retriever.retrieve("test query", "sub-1", ["p-1"], top_k=10)

    # Sparse encoder called at least once
    assert fake_sparse.encode.called
    # Both content_vector and keywords_vector queries were made
    calls = fake_qdrant.query_points.call_args_list
    usings = [kw.get("using") for _, kw in calls]
    assert "content_vector" in usings
    assert "keywords_vector" in usings
    # Result contains chunks from both branches
    chunk_ids = {c.chunk_id for c in result.chunks}
    assert "cd1" in chunk_ids  # from dense
    assert "cs1" in chunk_ids  # from sparse


def test_hybrid_retrieve_degrades_to_dense_when_sparse_encoder_none():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value.points = [
        _make_point("cd1", "dense", "doc-1", 0.9),
    ]
    fake_qdrant.scroll.return_value = ([], None)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        sparse_encoder=None,
    )
    retriever.retrieve("test", "sub-1", ["p-1"], top_k=10)

    calls = fake_qdrant.query_points.call_args_list
    usings = [kw.get("using") for _, kw in calls]
    assert "content_vector" in usings
    assert "keywords_vector" not in usings


def test_hybrid_retrieve_degrades_when_sparse_search_raises():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True

    def query_side_effect(**kwargs):
        if kwargs.get("using") == "keywords_vector":
            raise RuntimeError("sparse server error")
        res = MagicMock()
        res.points = [_make_point("cd1", "dense", "doc-1", 0.9)]
        return res

    fake_qdrant.query_points.side_effect = query_side_effect
    fake_qdrant.scroll.return_value = ([], None)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]
    fake_sparse = MagicMock()
    fake_sparse.encode.return_value = {"indices": [1], "values": [0.5]}

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        sparse_encoder=fake_sparse,
    )
    # Must not raise
    result = retriever.retrieve("test", "sub-1", ["p-1"], top_k=10)
    # Dense result still present
    assert any(c.chunk_id == "cd1" for c in result.chunks)


def test_hybrid_retrieve_backward_compatible_constructor():
    """Existing callers that only pass qdrant_client + embedder must still work."""
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value.points = []
    fake_qdrant.scroll.return_value = ([], None)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]

    # Old signature — no sparse_encoder, no graph_augmenter — must succeed
    retriever = UnifiedRetriever(qdrant_client=fake_qdrant, embedder=fake_embedder)
    result = retriever.retrieve("test", "sub-1", ["p-1"], top_k=10)
    assert result is not None
