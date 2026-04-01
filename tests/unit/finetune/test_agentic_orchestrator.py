"""Unit tests for Task 28 additions to agentic_orchestrator."""

import pytest

from src.finetune.agentic_orchestrator import (
    _BLOCKED_SOURCES,
    _TARGET_MODEL,
    enforce_data_policy,
    get_target_model,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_target_model_value(self):
        assert _TARGET_MODEL == "docwain:v2"

    def test_blocked_sources_contains_expected_keys(self):
        assert "document_content" in _BLOCKED_SOURCES
        assert "raw_text" in _BLOCKED_SOURCES
        assert "embedding_vector" in _BLOCKED_SOURCES

    def test_blocked_sources_is_set(self):
        assert isinstance(_BLOCKED_SOURCES, set)


# ---------------------------------------------------------------------------
# get_target_model()
# ---------------------------------------------------------------------------

class TestGetTargetModel:
    def test_returns_string(self):
        assert isinstance(get_target_model(), str)

    def test_returns_correct_model(self):
        assert get_target_model() == "docwain:v2"

    def test_consistent_across_calls(self):
        assert get_target_model() == get_target_model()


# ---------------------------------------------------------------------------
# enforce_data_policy() — allowed pairs
# ---------------------------------------------------------------------------

class TestEnforceDataPolicyAllow:
    def test_clean_pair_passes(self):
        pair = {"source": "user_feedback", "answer": "Short answer."}
        assert enforce_data_policy(pair) is True

    def test_empty_source_passes(self):
        pair = {"source": "", "answer": "Some answer."}
        assert enforce_data_policy(pair) is True

    def test_missing_source_passes(self):
        pair = {"answer": "Some answer."}
        assert enforce_data_policy(pair) is True

    def test_answer_exactly_2000_chars_passes(self):
        pair = {"source": "user_feedback", "answer": "x" * 2000}
        assert enforce_data_policy(pair) is True

    def test_answer_missing_passes(self):
        pair = {"source": "user_feedback"}
        assert enforce_data_policy(pair) is True

    def test_answer_none_passes(self):
        pair = {"source": "user_feedback", "answer": None}
        assert enforce_data_policy(pair) is True

    def test_unknown_source_passes(self):
        pair = {"source": "knowledge_base", "answer": "Valid answer."}
        assert enforce_data_policy(pair) is True


# ---------------------------------------------------------------------------
# enforce_data_policy() — blocked sources
# ---------------------------------------------------------------------------

class TestEnforceDataPolicyBlockedSource:
    def test_document_content_rejected(self):
        pair = {"source": "document_content", "answer": "Some text."}
        assert enforce_data_policy(pair) is False

    def test_raw_text_rejected(self):
        pair = {"source": "raw_text", "answer": "Some text."}
        assert enforce_data_policy(pair) is False

    def test_embedding_vector_rejected(self):
        pair = {"source": "embedding_vector", "answer": "Some text."}
        assert enforce_data_policy(pair) is False

    def test_blocked_source_with_short_answer_still_rejected(self):
        """Blocked source takes priority over answer length."""
        pair = {"source": "document_content", "answer": "OK"}
        assert enforce_data_policy(pair) is False

    def test_blocked_source_case_sensitive(self):
        """Source check is case-sensitive — uppercase variants should pass."""
        pair = {"source": "Document_Content", "answer": "Some text."}
        assert enforce_data_policy(pair) is True


# ---------------------------------------------------------------------------
# enforce_data_policy() — answer length
# ---------------------------------------------------------------------------

class TestEnforceDataPolicyAnswerLength:
    def test_answer_2001_chars_rejected(self):
        pair = {"source": "user_feedback", "answer": "x" * 2001}
        assert enforce_data_policy(pair) is False

    def test_very_long_answer_rejected(self):
        pair = {"source": "user_feedback", "answer": "a" * 10000}
        assert enforce_data_policy(pair) is False

    def test_answer_1999_chars_passes(self):
        pair = {"source": "user_feedback", "answer": "y" * 1999}
        assert enforce_data_policy(pair) is True

    def test_long_answer_with_blocked_source_rejected(self):
        """Both conditions fail — still rejected (source check first)."""
        pair = {"source": "raw_text", "answer": "z" * 5000}
        assert enforce_data_policy(pair) is False
