"""Lock KG entity expansion behaviour in UnifiedRetriever."""
from unittest.mock import MagicMock

from src.retrieval.retriever import UnifiedRetriever


def _make_dense_point(chunk_id: str, text: str, doc_id: str, score: float):
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


def _make_graph_hints(chunk_ids, doc_ids=None):
    from src.kg.retrieval import GraphHints, GraphSnippet
    snippets = [
        GraphSnippet(
            text=f"kg chunk text for {cid}",
            doc_id=(doc_ids[i] if doc_ids else f"doc-{cid}"),
            doc_name=f"{cid}.pdf",
            chunk_id=cid,
            relation="MENTIONS",
        )
        for i, cid in enumerate(chunk_ids)
    ]
    return GraphHints(
        evidence_chunk_ids=list(chunk_ids),
        doc_ids=doc_ids or [],
        graph_snippets=snippets,
    )


def test_kg_expansion_adds_chunks_from_graph_snippets():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    # Dense returns one chunk
    dense_result = MagicMock()
    dense_result.points = [_make_dense_point("cd1", "dense text", "doc-1", 0.9)]
    fake_qdrant.query_points.return_value = dense_result
    fake_qdrant.scroll.return_value = ([], None)

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]

    fake_augmenter = MagicMock()
    fake_augmenter.augment.return_value = _make_graph_hints(
        ["kg1", "kg2"], doc_ids=["doc-2", "doc-3"]
    )

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        graph_augmenter=fake_augmenter,
    )
    result = retriever.retrieve("find related entity", "sub-1", ["p-1"], top_k=10)

    ids = {c.chunk_id for c in result.chunks}
    assert "cd1" in ids  # dense still there
    assert "kg1" in ids  # kg expansion chunk 1
    assert "kg2" in ids  # kg expansion chunk 2


def test_kg_expansion_silent_when_augmenter_none():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value.points = []
    fake_qdrant.scroll.return_value = ([], None)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        graph_augmenter=None,
    )
    result = retriever.retrieve("query", "sub-1", ["p-1"], top_k=10)
    assert result is not None  # must not raise


def test_kg_expansion_silent_when_augmenter_raises():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value.points = []
    fake_qdrant.scroll.return_value = ([], None)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]
    fake_augmenter = MagicMock()
    fake_augmenter.augment.side_effect = RuntimeError("neo4j down")

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        graph_augmenter=fake_augmenter,
    )
    result = retriever.retrieve("query", "sub-1", ["p-1"], top_k=10)
    assert result is not None


def test_kg_expansion_does_not_duplicate_existing_chunks():
    """If a KG chunk_id is already in the dense/sparse results, don't add again."""
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    dense_result = MagicMock()
    dense_result.points = [_make_dense_point("shared", "in dense already", "doc-1", 0.9)]
    fake_qdrant.query_points.return_value = dense_result
    fake_qdrant.scroll.return_value = ([], None)

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]

    fake_augmenter = MagicMock()
    fake_augmenter.augment.return_value = _make_graph_hints(["shared", "kg_only"])

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        graph_augmenter=fake_augmenter,
    )
    result = retriever.retrieve("q", "sub-1", ["p-1"], top_k=10)

    # "shared" appears once, kg_only is added
    chunk_ids = [c.chunk_id for c in result.chunks]
    assert chunk_ids.count("shared") == 1
    assert "kg_only" in chunk_ids
