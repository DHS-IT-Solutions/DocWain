"""Unit tests for src/embedding/sparse.py

All heavy dependencies (transformers, model weights) are mocked so that
tests run without downloading anything.
"""

from __future__ import annotations

import types
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.embedding.sparse import SparseEncoder, sparse_to_qdrant


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 30522  # standard BERT vocab size


def _make_logits(seq_len: int = 8, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Return deterministic logits tensor shaped (1, seq_len, vocab_size)."""
    t = torch.zeros(1, seq_len, vocab_size)
    # Place positive values at a few known positions so we can assert on them.
    t[0, 0, 42] = 2.0
    t[0, 1, 42] = 1.5
    t[0, 2, 100] = 0.5
    # Negative values should be zeroed out by ReLU.
    t[0, 3, 200] = -1.0
    return t


def _make_attention_mask(seq_len: int = 8) -> torch.Tensor:
    """Return all-ones attention mask shaped (1, seq_len)."""
    return torch.ones(1, seq_len, dtype=torch.long)


def _make_model_output(logits: torch.Tensor) -> MagicMock:
    output = MagicMock()
    output.logits = logits
    return output


def _make_tokenizer_output(seq_len: int = 8) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "attention_mask": _make_attention_mask(seq_len),
        "token_type_ids": torch.zeros(1, seq_len, dtype=torch.long),
    }


def _build_encoder_with_mocks(
    seq_len: int = 8,
    custom_logits: torch.Tensor | None = None,
) -> tuple[SparseEncoder, MagicMock, MagicMock]:
    """Return (encoder, mock_tokenizer, mock_model) with pre-wired behaviour."""
    logits = custom_logits if custom_logits is not None else _make_logits(seq_len)
    tokenizer_out = _make_tokenizer_output(seq_len)

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = tokenizer_out

    mock_model = MagicMock()
    mock_model.return_value = _make_model_output(logits)
    # Simulate `.to()` returning itself and `.eval()` returning itself.
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    encoder = SparseEncoder(model_name="naver/splade-v3", device="cpu")
    encoder._model = mock_model
    encoder._tokenizer = mock_tokenizer
    encoder._resolved_device = "cpu"

    return encoder, mock_tokenizer, mock_model


# ---------------------------------------------------------------------------
# SparseEncoder.encode — basic correctness
# ---------------------------------------------------------------------------

class TestSparseEncoderEncode:
    def test_returns_dict_with_indices_and_values(self):
        encoder, _, _ = _build_encoder_with_mocks()
        result = encoder.encode("hello world")
        assert isinstance(result, dict)
        assert "indices" in result
        assert "values" in result

    def test_indices_are_ints(self):
        encoder, _, _ = _build_encoder_with_mocks()
        result = encoder.encode("test")
        assert all(isinstance(i, int) for i in result["indices"])

    def test_values_are_floats(self):
        encoder, _, _ = _build_encoder_with_mocks()
        result = encoder.encode("test")
        assert all(isinstance(v, float) for v in result["values"])

    def test_indices_and_values_same_length(self):
        encoder, _, _ = _build_encoder_with_mocks()
        result = encoder.encode("test")
        assert len(result["indices"]) == len(result["values"])

    def test_known_positive_indices_present(self):
        """Vocab positions 42 and 100 have positive logits → must appear."""
        encoder, _, _ = _build_encoder_with_mocks()
        result = encoder.encode("test")
        assert 42 in result["indices"]
        assert 100 in result["indices"]

    def test_negative_logit_position_absent(self):
        """Vocab position 200 has a negative logit → ReLU zeroes it out."""
        encoder, _, _ = _build_encoder_with_mocks()
        result = encoder.encode("test")
        assert 200 not in result["indices"]

    def test_all_values_positive(self):
        """SPLADE formula log(1+ReLU(x)) is always ≥ 0; stored values > 0."""
        encoder, _, _ = _build_encoder_with_mocks()
        result = encoder.encode("test")
        assert all(v > 0.0 for v in result["values"])

    def test_max_pool_takes_max_over_seq(self):
        """Position 42 appears in seq tokens 0 (logit=2.0) and 1 (logit=1.5).
        Max-pooling should pick the larger value: log(1 + 2.0) = log(3) ≈ 1.0986.
        """
        encoder, _, _ = _build_encoder_with_mocks()
        result = encoder.encode("test")
        idx_42_pos = result["indices"].index(42)
        expected = torch.log1p(torch.tensor(2.0)).item()
        assert abs(result["values"][idx_42_pos] - expected) < 1e-5

    def test_attention_mask_zeros_suppress_tokens(self):
        """Tokens with attention_mask=0 must not contribute."""
        seq_len = 4
        logits = torch.zeros(1, seq_len, VOCAB_SIZE)
        # Token 1 has a positive logit but is masked out.
        logits[0, 1, 999] = 3.0

        tokenizer_out = {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 0, 1, 1]], dtype=torch.long),
        }

        mock_tokenizer = MagicMock(return_value=tokenizer_out)
        mock_model = MagicMock(return_value=_make_model_output(logits))
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        encoder = SparseEncoder(device="cpu")
        encoder._model = mock_model
        encoder._tokenizer = mock_tokenizer
        encoder._resolved_device = "cpu"

        result = encoder.encode("test")
        assert 999 not in result["indices"]

    def test_empty_result_when_all_logits_nonpositive(self):
        """All-negative logits → empty sparse vector."""
        seq_len = 8  # must match default seq_len used by _build_encoder_with_mocks
        logits = torch.full((1, seq_len, VOCAB_SIZE), -1.0)
        encoder, _, _ = _build_encoder_with_mocks(custom_logits=logits)
        result = encoder.encode("test")
        assert result["indices"] == []
        assert result["values"] == []


# ---------------------------------------------------------------------------
# SparseEncoder.encode_batch
# ---------------------------------------------------------------------------

class TestSparseEncoderEncodeBatch:
    def test_returns_list_of_dicts(self):
        encoder, _, _ = _build_encoder_with_mocks()
        results = encoder.encode_batch(["foo", "bar", "baz"])
        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert "indices" in r
            assert "values" in r

    def test_empty_batch_returns_empty_list(self):
        encoder, _, _ = _build_encoder_with_mocks()
        results = encoder.encode_batch([])
        assert results == []

    def test_single_item_batch_matches_encode(self):
        encoder, _, _ = _build_encoder_with_mocks()
        single = encoder.encode("hello")
        batch = encoder.encode_batch(["hello"])
        assert batch[0]["indices"] == single["indices"]
        assert batch[0]["values"] == single["values"]

    def test_model_called_once_per_text(self):
        encoder, _, mock_model = _build_encoder_with_mocks()
        texts = ["a", "b", "c"]
        encoder.encode_batch(texts)
        assert mock_model.call_count == len(texts)


# ---------------------------------------------------------------------------
# Lazy loading (_ensure_loaded)
# ---------------------------------------------------------------------------

class TestEnsureLoaded:
    def test_model_not_loaded_before_encode(self):
        encoder = SparseEncoder(model_name="naver/splade-v3", device="cpu")
        assert encoder._model is None
        assert encoder._tokenizer is None

    def test_ensure_loaded_called_on_first_encode(self):
        encoder = SparseEncoder(device="cpu")

        load_call_count = {"n": 0}

        def fake_ensure_loaded():
            load_call_count["n"] += 1
            # Wire in mocks after "loading".
            logits = _make_logits()
            tokenizer_out = _make_tokenizer_output()
            mock_tokenizer = MagicMock(return_value=tokenizer_out)
            mock_model = MagicMock(return_value=_make_model_output(logits))
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = mock_model
            encoder._model = mock_model
            encoder._tokenizer = mock_tokenizer
            encoder._resolved_device = "cpu"

        encoder._ensure_loaded = fake_ensure_loaded  # type: ignore[method-assign]

        encoder.encode("hello")
        assert load_call_count["n"] == 1

    def test_transformers_loaded_lazily(self):
        """_ensure_loaded must import from transformers, not at module import."""
        mock_auto_model = MagicMock()
        mock_auto_tokenizer = MagicMock()

        logits = _make_logits()
        tokenizer_out = _make_tokenizer_output()

        mock_auto_tokenizer.from_pretrained.return_value = MagicMock(
            return_value=tokenizer_out
        )
        mock_auto_model_instance = MagicMock(
            return_value=_make_model_output(logits)
        )
        mock_auto_model_instance.to.return_value = mock_auto_model_instance
        mock_auto_model_instance.eval.return_value = mock_auto_model_instance
        mock_auto_model.from_pretrained.return_value = mock_auto_model_instance

        with patch(
            "src.embedding.sparse.AutoModelForMaskedLM", mock_auto_model, create=True
        ), patch(
            "src.embedding.sparse.AutoTokenizer", mock_auto_tokenizer, create=True
        ):
            # We need to reach inside _ensure_loaded without it actually importing.
            # Re-import the module so the patch applies cleanly.
            import importlib
            import src.embedding.sparse as sparse_mod

            with patch.dict(
                "sys.modules",
                {
                    "transformers": MagicMock(
                        AutoModelForMaskedLM=mock_auto_model,
                        AutoTokenizer=mock_auto_tokenizer,
                    )
                },
            ):
                enc = sparse_mod.SparseEncoder(device="cpu")
                enc._ensure_loaded()
                assert enc._model is not None
                assert enc._tokenizer is not None


# ---------------------------------------------------------------------------
# sparse_to_qdrant
# ---------------------------------------------------------------------------

class TestSparseToQdrant:
    def test_returns_sparse_vector_type(self):
        from qdrant_client.models import SparseVector

        result = sparse_to_qdrant({"indices": [1, 5, 42], "values": [0.1, 0.5, 0.9]})
        assert isinstance(result, SparseVector)

    def test_indices_preserved(self):
        from qdrant_client.models import SparseVector

        indices = [0, 10, 200, 999]
        values = [0.1, 0.2, 0.3, 0.4]
        result = sparse_to_qdrant({"indices": indices, "values": values})
        assert list(result.indices) == indices

    def test_values_preserved(self):
        from qdrant_client.models import SparseVector

        indices = [0, 10]
        values = [0.75, 1.25]
        result = sparse_to_qdrant({"indices": indices, "values": values})
        assert list(result.values) == values

    def test_empty_sparse_dict(self):
        from qdrant_client.models import SparseVector

        result = sparse_to_qdrant({"indices": [], "values": []})
        assert isinstance(result, SparseVector)
        assert list(result.indices) == []
        assert list(result.values) == []

    def test_roundtrip_from_encode(self):
        """encode → sparse_to_qdrant should produce a valid SparseVector."""
        from qdrant_client.models import SparseVector

        encoder, _, _ = _build_encoder_with_mocks()
        sparse_dict = encoder.encode("roundtrip test")
        sv = sparse_to_qdrant(sparse_dict)
        assert isinstance(sv, SparseVector)
        assert len(sv.indices) == len(sparse_dict["indices"])
        assert len(sv.values) == len(sparse_dict["values"])
