"""Verify AppState exposes sparse_encoder field."""
from src.api.rag_state import AppState


def test_appstate_has_sparse_encoder_field():
    """Constructor must accept sparse_encoder keyword."""
    state = AppState(
        embedding_model=None,
        reranker=None,
        qdrant_client=None,
        redis_client=None,
        ollama_client=None,
        rag_system=None,
        sparse_encoder="stub-sentinel",
    )
    assert state.sparse_encoder == "stub-sentinel"


def test_appstate_sparse_encoder_defaults_to_none():
    state = AppState(
        embedding_model=None,
        reranker=None,
        qdrant_client=None,
        redis_client=None,
        ollama_client=None,
        rag_system=None,
    )
    assert state.sparse_encoder is None
