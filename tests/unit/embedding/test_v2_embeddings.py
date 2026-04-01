"""Unit tests for src/embedding/v2_embeddings.py

All heavy dependencies (transformers model weights, tokenizer downloads) are
mocked so that tests run without any network access or GPU requirement.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn

from src.embedding.v2_embeddings import V2Embedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 5120
PROJECTION_DIM = 1024
SEQ_LEN = 12


def _make_hidden_states(seq_len: int = SEQ_LEN, hidden_dim: int = HIDDEN_DIM) -> torch.Tensor:
    """Return deterministic hidden-state tensor shaped (seq_len, hidden_dim)."""
    torch.manual_seed(42)
    return torch.randn(seq_len, hidden_dim)


def _make_batch_hidden(seq_len: int = SEQ_LEN, hidden_dim: int = HIDDEN_DIM) -> torch.Tensor:
    """Return batched hidden states shaped (1, seq_len, hidden_dim)."""
    torch.manual_seed(42)
    return torch.randn(1, seq_len, hidden_dim)


def _make_embedder(**kwargs) -> V2Embedder:
    """Construct a V2Embedder forced to CPU regardless of hardware."""
    defaults = dict(
        model_name="test/model",
        projection_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )
    defaults.update(kwargs)
    return V2Embedder(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def embedder() -> V2Embedder:
    return _make_embedder()


# ===========================================================================
# __init__
# ===========================================================================

class TestInit:

    def test_default_params(self):
        emb = V2Embedder()
        assert emb.model_name == "docwain:v2"
        assert emb.ollama_host == "http://localhost:11434"
        assert emb.projection_dim == 1024
        assert emb.hidden_dim == 5120

    def test_custom_params(self):
        emb = V2Embedder(
            model_name="my/model",
            ollama_host="http://gpu01:11434",
            projection_dim=512,
            hidden_dim=4096,
            device="cpu",
        )
        assert emb.model_name == "my/model"
        assert emb.projection_dim == 512
        assert emb.hidden_dim == 4096

    def test_projection_layer_created(self, embedder):
        assert isinstance(embedder._projection, nn.Linear)
        assert embedder._projection.in_features == HIDDEN_DIM
        assert embedder._projection.out_features == PROJECTION_DIM

    def test_model_not_loaded_at_init(self, embedder):
        assert embedder._model is None
        assert embedder._tokenizer is None

    def test_device_auto_resolves(self):
        emb = V2Embedder(device="auto")
        # device should be cpu or cuda — just check it's a torch.device
        assert isinstance(emb.device, torch.device)

    def test_device_cpu_explicit(self, embedder):
        assert embedder.device == torch.device("cpu")


# ===========================================================================
# _projection shape & dtype
# ===========================================================================

class TestProjectionLayer:

    def test_projection_output_shape(self, embedder):
        x = torch.randn(1, HIDDEN_DIM)
        with torch.no_grad():
            out = embedder._projection(x)
        assert out.shape == (1, PROJECTION_DIM)

    def test_projection_output_dtype_float32(self, embedder):
        x = torch.randn(1, HIDDEN_DIM)
        with torch.no_grad():
            out = embedder._projection(x)
        assert out.dtype == torch.float32

    def test_projection_no_bias(self, embedder):
        assert embedder._projection.bias is None

    def test_projection_deterministic_given_same_weights(self, embedder):
        x = torch.randn(1, HIDDEN_DIM)
        with torch.no_grad():
            out1 = embedder._projection(x)
            out2 = embedder._projection(x)
        assert torch.allclose(out1, out2)


# ===========================================================================
# _encode_single — mocking _get_hidden_states
# ===========================================================================

class TestEncodeSingle:

    def test_output_is_list(self, embedder):
        hidden = _make_hidden_states()
        with patch.object(embedder, "_get_hidden_states", return_value=hidden):
            result = embedder._encode_single("hello world")
        assert isinstance(result, list)

    def test_output_length_equals_projection_dim(self, embedder):
        hidden = _make_hidden_states()
        with patch.object(embedder, "_get_hidden_states", return_value=hidden):
            result = embedder._encode_single("hello world")
        assert len(result) == PROJECTION_DIM

    def test_output_elements_are_floats(self, embedder):
        hidden = _make_hidden_states()
        with patch.object(embedder, "_get_hidden_states", return_value=hidden):
            result = embedder._encode_single("test")
        assert all(isinstance(v, float) for v in result)

    def test_mean_pool_over_sequence(self, embedder):
        """Verify the mean-pool step is actually applied."""
        seq_len = 4
        hidden_dim = HIDDEN_DIM
        # Construct hidden states where each token row is a constant multiple
        # of a base vector so we know what the mean should be.
        base = torch.ones(hidden_dim)
        hidden = torch.stack([base * float(i) for i in range(seq_len)])  # (4, hidden_dim)
        expected_mean = base * 1.5  # mean of [0,1,2,3]

        # Capture the input to the projection layer
        projection_inputs: list[torch.Tensor] = []
        original_forward = embedder._projection.forward

        def capturing_forward(x: torch.Tensor) -> torch.Tensor:
            projection_inputs.append(x.detach().clone())
            return original_forward(x)

        with patch.object(embedder, "_get_hidden_states", return_value=hidden):
            with patch.object(embedder._projection, "forward", side_effect=capturing_forward):
                embedder._encode_single("test")

        assert len(projection_inputs) == 1
        # shape should be (1, hidden_dim)
        pooled = projection_inputs[0].squeeze(0)
        assert torch.allclose(pooled, expected_mean, atol=1e-5)

    def test_different_inputs_produce_different_outputs(self, embedder):
        hidden_a = torch.randn(SEQ_LEN, HIDDEN_DIM)
        hidden_b = torch.randn(SEQ_LEN, HIDDEN_DIM) + 10.0  # clearly different

        with patch.object(embedder, "_get_hidden_states", side_effect=[hidden_a, hidden_b]):
            out_a = embedder._encode_single("text A")
            out_b = embedder._encode_single("text B")

        assert out_a != out_b


# ===========================================================================
# encode (public)
# ===========================================================================

class TestEncode:

    def test_encode_delegates_to_encode_single(self, embedder):
        expected = [0.5] * PROJECTION_DIM
        with patch.object(embedder, "_encode_single", return_value=expected) as mock:
            result = embedder.encode("hello")
        mock.assert_called_once_with("hello")
        assert result == expected

    def test_encode_returns_list_of_floats(self, embedder):
        hidden = _make_hidden_states()
        with patch.object(embedder, "_get_hidden_states", return_value=hidden):
            result = embedder.encode("sample text")
        assert isinstance(result, list)
        assert len(result) == PROJECTION_DIM


# ===========================================================================
# encode_batch (public)
# ===========================================================================

class TestEncodeBatch:

    def test_encode_batch_returns_list_of_lists(self, embedder):
        texts = ["text one", "text two", "text three"]
        hidden = _make_hidden_states()
        with patch.object(embedder, "_get_hidden_states", return_value=hidden):
            results = embedder.encode_batch(texts)
        assert isinstance(results, list)
        assert len(results) == len(texts)
        for vec in results:
            assert isinstance(vec, list)
            assert len(vec) == PROJECTION_DIM

    def test_encode_batch_empty_list(self, embedder):
        result = embedder.encode_batch([])
        assert result == []

    def test_encode_batch_single_item(self, embedder):
        hidden = _make_hidden_states()
        with patch.object(embedder, "_get_hidden_states", return_value=hidden):
            result = embedder.encode_batch(["only one"])
        assert len(result) == 1

    def test_encode_batch_calls_encode_single_per_text(self, embedder):
        texts = ["a", "b", "c"]
        dummy = [1.0] * PROJECTION_DIM
        with patch.object(embedder, "_encode_single", return_value=dummy) as mock:
            embedder.encode_batch(texts)
        assert mock.call_count == len(texts)

    def test_encode_batch_preserves_order(self, embedder):
        """Each text should produce a distinct vector; order must be preserved."""
        call_index = [0]

        def side_effect(text: str) -> list[float]:
            idx = call_index[0]
            call_index[0] += 1
            return [float(idx)] * PROJECTION_DIM

        with patch.object(embedder, "_encode_single", side_effect=side_effect):
            results = embedder.encode_batch(["a", "b", "c"])

        assert results[0] == [0.0] * PROJECTION_DIM
        assert results[1] == [1.0] * PROJECTION_DIM
        assert results[2] == [2.0] * PROJECTION_DIM


# ===========================================================================
# _ensure_loaded — lazy loading behaviour
# ===========================================================================

class TestEnsureLoaded:

    def _make_mock_model(self) -> MagicMock:
        model = MagicMock()
        model.eval.return_value = model
        # Simulate hidden_states output
        mock_hs = [torch.randn(1, SEQ_LEN, HIDDEN_DIM) for _ in range(13)]
        model.return_value.hidden_states = mock_hs
        return model

    def _make_mock_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        # tokenizer() returns a dict-like with input_ids etc.
        tok.return_value = {"input_ids": torch.ones(1, SEQ_LEN, dtype=torch.long)}
        return tok

    def test_ensure_loaded_sets_model_and_tokenizer(self, embedder):
        mock_model_cls = self._make_mock_model()
        mock_tok_cls = self._make_mock_tokenizer()

        with patch.dict(
            "sys.modules",
            {
                "transformers": types.ModuleType("transformers"),
            },
        ):
            import sys
            sys.modules["transformers"].AutoModel = MagicMock(return_value=mock_model_cls)
            sys.modules["transformers"].AutoTokenizer = MagicMock(return_value=mock_tok_cls)

            embedder._ensure_loaded()
            # After loading, _model and _tokenizer should be set
            assert embedder._model is not None
            assert embedder._tokenizer is not None

    def test_ensure_loaded_only_loads_once(self, embedder):
        """Calling _ensure_loaded twice should not reload."""
        sentinel = object()
        embedder._model = sentinel  # mark as already loaded
        embedder._tokenizer = sentinel

        with patch("src.embedding.v2_embeddings.V2Embedder._ensure_loaded") as mock:
            # Simulate second call — the guard should return early.
            # We can verify this by patching _ensure_loaded on an already-loaded embedder.
            pass  # guard is checked by _model is not None inside the method

        # Re-call directly; since _model is set it should return immediately.
        embedder._ensure_loaded()
        assert embedder._model is sentinel  # unchanged

    def test_ensure_loaded_raises_on_missing_transformers(self, embedder):
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises((RuntimeError, ImportError)):
                embedder._ensure_loaded()
