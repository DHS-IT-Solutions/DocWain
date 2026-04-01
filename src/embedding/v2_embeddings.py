"""DocWain V2 Model Embeddings.

Provides the V2Embedder class that wraps the DocWain V2 vision-grafted model
(Qwen3-14B based) and projects its 5120-dim hidden states down to a compact
1024-dim embedding space suitable for Qdrant storage.

Heavy dependencies (transformers, model weights) are lazy-loaded on first use
so that importing this module is always fast.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["V2Embedder"]


class V2Embedder:
    """Embed text using the DocWain V2 model.

    Args:
        model_name:     Ollama model tag or HuggingFace repo id for the base
                        model weights used by the tokenizer / forward pass.
        ollama_host:    Ollama API endpoint (informational; transformers path
                        loads weights directly).
        projection_dim: Output embedding dimension (default 1024).
        hidden_dim:     Hidden-state dimension of the base model (default 5120
                        for Qwen3-14B).
        device:         ``"auto"`` selects CUDA when available, else CPU.
                        Pass ``"cpu"`` or ``"cuda"`` to override.
    """

    def __init__(
        self,
        model_name: str = "docwain:v2",
        ollama_host: str = "http://localhost:11434",
        projection_dim: int = 1024,
        hidden_dim: int = 5120,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Projection layer is always initialised (small, no model weights needed)
        self._projection: nn.Linear = nn.Linear(
            hidden_dim, projection_dim, bias=False
        ).to(self.device)
        self._projection.eval()

        # Lazy-loaded heavy components
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the base model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for V2Embedder. "
                "Install it with: pip install transformers"
            ) from exc

        logger.info("Loading V2 model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            output_hidden_states=True,
            trust_remote_code=True,
        ).to(self.device)
        self._model.eval()  # type: ignore[union-attr]
        logger.info("V2 model loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hidden_states(self, text: str) -> torch.Tensor:
        """Tokenize *text* and return the penultimate hidden layer tensor.

        Returns:
            Tensor of shape (sequence_length, hidden_dim).
        """
        self._ensure_loaded()

        inputs = self._tokenizer(  # type: ignore[operator]
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)  # type: ignore[operator]

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors,
        # each shaped (batch, seq_len, hidden_dim).  Penultimate = index -2.
        hidden_states = outputs.hidden_states[-2]  # (1, seq_len, hidden_dim)
        return hidden_states[0]  # (seq_len, hidden_dim)

    def _encode_single(self, text: str) -> list[float]:
        """Encode a single string to a projected embedding vector.

        Steps:
            1. Obtain the penultimate hidden states (seq_len, hidden_dim).
            2. Mean-pool over the sequence dimension -> (hidden_dim,).
            3. Pass through the linear projection -> (projection_dim,).
            4. Return as a Python list of floats.
        """
        hidden = self._get_hidden_states(text)           # (seq_len, hidden_dim)
        pooled = hidden.mean(dim=0, keepdim=True)        # (1, hidden_dim)
        with torch.no_grad():
            projected = self._projection(pooled)         # (1, projection_dim)
        return projected.squeeze(0).cpu().tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[float]:
        """Encode a single text string.

        Args:
            text: Input text to embed.

        Returns:
            List of ``projection_dim`` floats.
        """
        return self._encode_single(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of text strings.

        Each string is encoded independently (no padding/batching at the
        transformer level — keeps the implementation simple and avoids
        padding-token contamination in the mean-pool step).

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, one per input string.
        """
        return [self._encode_single(t) for t in texts]
