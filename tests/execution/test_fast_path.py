"""Tests for the fast path execution pipeline."""
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qdrant_point(text: str, doc_id: str, score: float):
    """Create a mock Qdrant point."""
    point = MagicMock()
    point.payload = {
        "canonical_text": text,
        "document_id": doc_id,
        "source_name": f"{doc_id}.pdf",
        "profile_id": "profile_1",
        "section_id": "sec_1",
        "page": 1,
        "chunk_kind": "section_text",
    }
    point.score = score
    point.id = f"point_{doc_id}_{score}"
    return point


def _make_app_state(
    qdrant_points=None,
    llm_response="The revenue was $1.5M in Q2 2024.",
    reranker=None,
):
    """Build a mock app_state with all required components."""
    state = MagicMock()
    state.embedding_model.encode.return_value = [[0.1] * 1024]
    results_obj = MagicMock()
    results_obj.points = qdrant_points if qdrant_points is not None else [
        _make_qdrant_point("Revenue was $1.5M in Q2.", "doc1", 0.9),
    ]
    state.qdrant_client.query_points.return_value = results_obj
    state.reranker = reranker
    state.llm_gateway.generate.return_value = llm_response
    state.llm_gateway.generate_stream.return_value = iter(
        [llm_response[:20], llm_response[20:]]
    )
    return state


# ---------------------------------------------------------------------------
# Fast path unit tests
# ---------------------------------------------------------------------------

class TestExecuteFastPath:
    def test_returns_answer_with_fast_path_flag(self):
        """Fast path should return a response dict with answer and fast_path flag."""
        from src.execution.fast_path import execute_fast_path

        state = _make_app_state()
        result = execute_fast_path("What is the revenue?", "profile_1", "sub_1", state)

        assert result["fast_path"] is True
        assert result["query_type"] == "SIMPLE"
        assert "answer" in result
        assert len(result["answer"]) > 0
        assert "response" in result
        assert result["response"] == result["answer"]

    def test_returns_sources(self):
        """Fast path should include sources from retrieved chunks."""
        from src.execution.fast_path import execute_fast_path

        points = [
            _make_qdrant_point("Revenue was $1.5M.", "doc1", 0.9),
            _make_qdrant_point("Expenses were $500K.", "doc2", 0.8),
        ]
        state = _make_app_state(qdrant_points=points)
        result = execute_fast_path("What is the revenue?", "profile_1", "sub_1", state)

        assert "sources" in result
        assert len(result["sources"]) > 0
        assert result["sources"][0]["document_id"] == "doc1"

    def test_context_found_when_chunks_exist(self):
        """context_found should be True when chunks are retrieved."""
        from src.execution.fast_path import execute_fast_path

        state = _make_app_state()
        result = execute_fast_path("What is the revenue?", "profile_1", "sub_1", state)

        assert result["context_found"] is True
        assert result["grounded"] is True

    def test_context_found_false_when_no_chunks(self):
        """context_found should be False when no chunks retrieved."""
        from src.execution.fast_path import execute_fast_path

        state = _make_app_state(qdrant_points=[])
        result = execute_fast_path("What is the revenue?", "profile_1", "sub_1", state)

        assert result["context_found"] is False

    def test_metadata_includes_timing(self):
        """metadata should include elapsed_s and chunks_used."""
        from src.execution.fast_path import execute_fast_path

        state = _make_app_state()
        result = execute_fast_path("What is the revenue?", "profile_1", "sub_1", state)

        assert "metadata" in result
        assert "elapsed_s" in result["metadata"]
        assert "chunks_used" in result["metadata"]

    def test_calls_embedding_model(self):
        """Should call embedding model with the query."""
        from src.execution.fast_path import execute_fast_path

        state = _make_app_state()
        execute_fast_path("What is the revenue?", "profile_1", "sub_1", state)

        state.embedding_model.encode.assert_called_once()
        args = state.embedding_model.encode.call_args
        assert "What is the revenue?" in args[0][0]

    def test_calls_qdrant_with_subscription_as_collection(self):
        """Should use subscription_id as collection name."""
        from src.execution.fast_path import execute_fast_path

        state = _make_app_state()
        execute_fast_path("What is the revenue?", "profile_1", "sub_1", state)

        state.qdrant_client.query_points.assert_called_once()
        call_kwargs = state.qdrant_client.query_points.call_args
        assert call_kwargs.kwargs["collection_name"] == "sub_1"

    def test_skips_reranker_when_none(self):
        """When reranker is None, should still return results (keyword + dense only)."""
        from src.execution.fast_path import execute_fast_path

        state = _make_app_state(reranker=None)
        result = execute_fast_path("What is the revenue?", "profile_1", "sub_1", state)

        assert result["fast_path"] is True
        assert len(result["answer"]) > 0


# ---------------------------------------------------------------------------
# _build_fast_context tests
# ---------------------------------------------------------------------------

class TestBuildFastContext:
    def test_limits_to_max_chunks(self):
        """Should use at most max_chunks chunks."""
        from src.execution.fast_path import _build_fast_context

        chunks = [{"text": f"chunk {i}"} for i in range(10)]
        context = _build_fast_context(chunks, max_chunks=3)
        assert context.count("chunk") == 3

    def test_empty_chunks(self):
        """Should handle empty input gracefully."""
        from src.execution.fast_path import _build_fast_context

        assert _build_fast_context([], max_chunks=3) == ""

    def test_respects_char_limit(self):
        """Should not exceed the max context chars."""
        from src.execution.fast_path import _build_fast_context, _MAX_CONTEXT_CHARS

        chunks = [{"text": "x" * 2000} for _ in range(5)]
        context = _build_fast_context(chunks, max_chunks=5)
        assert len(context) <= _MAX_CONTEXT_CHARS + 100  # small separator overhead


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

class TestExecuteFastPathStream:
    def test_yields_chunks(self):
        """Streaming fast path should yield response chunks."""
        from src.execution.fast_path import execute_fast_path_stream

        state = _make_app_state()
        chunks = list(execute_fast_path_stream("What is the revenue?", "profile_1", "sub_1", state))

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

    def test_calls_generate_stream(self):
        """Should call llm_gateway.generate_stream, not generate."""
        from src.execution.fast_path import execute_fast_path_stream

        state = _make_app_state()
        list(execute_fast_path_stream("What is the revenue?", "profile_1", "sub_1", state))

        state.llm_gateway.generate_stream.assert_called_once()


# ---------------------------------------------------------------------------
# Router integration tests
# ---------------------------------------------------------------------------

class TestRouterFastPath:
    @patch("src.execution.router.FAST_PATH_ENABLED", True)
    @patch("src.execution.router._get_core_agent")
    def test_uses_fast_path_for_simple(self, mock_agent):
        """Router should use fast path for SIMPLE queries when enabled."""
        from src.execution.query_classifier import QueryClassification

        mock_app_state = _make_app_state()

        with patch("src.execution.fast_path.classify_query", create=True), \
             patch("src.api.rag_state.get_app_state", return_value=mock_app_state):

            # Patch classify_query at the import site in router
            with patch(
                "src.execution.query_classifier.classify_query",
                return_value=QueryClassification(
                    query_type="SIMPLE", confidence=0.9, signals=["short_factoid"]
                ),
            ):
                from src.execution.router import execute_request

                request = MagicMock()
                request.query = "What is the revenue?"
                request.subscription_id = "sub_1"
                request.profile_id = "profile_1"
                request.user_id = "user_1"
                request.agent_name = None
                request.document_id = None
                ctx = MagicMock()
                ctx.session_id = None

                result = execute_request(request, None, ctx)

                assert result.answer["fast_path"] is True
                assert result.answer["query_type"] == "SIMPLE"
                # CoreAgent should NOT be called
                mock_agent.return_value.handle.assert_not_called()

    @patch("src.execution.router.FAST_PATH_ENABLED", False)
    def test_skips_fast_path_when_disabled(self):
        """Router should skip fast path when FAST_PATH_ENABLED is False."""
        with patch("src.execution.router._get_core_agent") as mock_agent:
            mock_agent.return_value.handle.return_value = {
                "response": "full path", "metadata": {}
            }

            from src.execution.router import execute_request

            request = MagicMock()
            request.query = "What is the revenue?"
            request.subscription_id = "sub_1"
            request.profile_id = "profile_1"
            request.user_id = "user_1"
            request.agent_name = None
            request.document_id = None
            ctx = MagicMock()
            ctx.session_id = None

            execute_request(request, None, ctx)

            # CoreAgent SHOULD be called
            mock_agent.return_value.handle.assert_called_once()

    @patch("src.execution.router.FAST_PATH_ENABLED", True)
    def test_falls_back_on_complex_query(self):
        """Router should use CoreAgent for non-SIMPLE queries."""
        from src.execution.query_classifier import QueryClassification

        with patch("src.execution.router._get_core_agent") as mock_agent, \
             patch(
                 "src.execution.query_classifier.classify_query",
                 return_value=QueryClassification(
                     query_type="COMPLEX", confidence=0.9, signals=["multi_doc_keyword"]
                 ),
             ):
            mock_agent.return_value.handle.return_value = {
                "response": "full path", "metadata": {}
            }

            from src.execution.router import execute_request

            request = MagicMock()
            request.query = "Compare revenue across all documents"
            request.subscription_id = "sub_1"
            request.profile_id = "profile_1"
            request.user_id = "user_1"
            request.agent_name = None
            request.document_id = None
            ctx = MagicMock()
            ctx.session_id = None

            execute_request(request, None, ctx)

            mock_agent.return_value.handle.assert_called_once()
