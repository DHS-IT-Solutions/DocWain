"""SPLADE sparse embedding encoder.

Produces learned sparse representations using the SPLADE formula:
    w_i = max_j log(1 + ReLU(h_{ij})) over the sequence dimension

These are stored alongside dense vectors in Qdrant for hybrid retrieval.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "naver/splade-v3"
_MAX_LENGTH = 512


class SparseEncoder:
    """SPLADE-based sparse encoder with lazy model loading.

    Parameters
    ----------
    model_name:
        HuggingFace model ID for a SPLADE-compatible masked-LM.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.
        Pass ``"cpu"`` or ``"cuda"`` to override.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._resolved_device: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        """Lazy-load the tokenizer and model on first encode call."""
        if self._model is not None:
            return

        from transformers import AutoModelForMaskedLM, AutoTokenizer

        device = self._resolve_device()
        self._resolved_device = device

        offline = os.getenv("HF_HUB_OFFLINE") or os.getenv("TRANSFORMERS_OFFLINE")

        logger.info(
            "Loading SPLADE model: %s (device=%s)", self.model_name, device
        )

        load_kwargs: Dict[str, object] = {}
        if offline:
            load_kwargs["local_files_only"] = True

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **load_kwargs
        )
        self._model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        self._model.to(device)  # type: ignore[union-attr]
        self._model.eval()  # type: ignore[union-attr]

        logger.info("SPLADE model ready: %s", self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> Dict[str, List]:
        """Encode a single text string into a sparse vector.

        Parameters
        ----------
        text:
            Input text (will be truncated to ``_MAX_LENGTH`` tokens).

        Returns
        -------
        dict
            ``{"indices": [int, ...], "values": [float, ...]}`` where only
            non-zero dimensions are included.
        """
        self._ensure_loaded()

        device = self._resolved_device

        inputs = self._tokenizer(  # type: ignore[misc]
            text,
            return_tensors="pt",
            max_length=_MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)  # type: ignore[misc]

        logits = outputs.logits  # (1, seq_len, vocab_size)
        attention_mask = inputs["attention_mask"]  # (1, seq_len)

        # SPLADE formula: log(1 + ReLU(logits)) * attention_mask (broadcast)
        # attention_mask shape: (1, seq_len, 1) after unsqueeze
        weighted = (
            torch.log1p(torch.relu(logits))
            * attention_mask.unsqueeze(-1)
        )  # (1, seq_len, vocab_size)

        # Max-pool over sequence dimension → (1, vocab_size)
        sparse_vec = weighted.max(dim=1).values.squeeze(0)  # (vocab_size,)

        # Extract non-zero positions
        nonzero_mask = sparse_vec > 0
        indices = nonzero_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        values = sparse_vec[nonzero_mask].tolist()

        return {"indices": indices, "values": values}

    def encode_batch(self, texts: List[str]) -> List[Dict[str, List]]:
        """Encode a list of texts, returning one sparse dict per text.

        Parameters
        ----------
        texts:
            List of input strings.

        Returns
        -------
        list of dict
            One ``{"indices": [...], "values": [...]}`` dict per input text.
        """
        return [self.encode(text) for text in texts]


# ---------------------------------------------------------------------------
# Qdrant conversion helper
# ---------------------------------------------------------------------------


def sparse_to_qdrant(sparse_dict: Dict[str, List]) -> object:
    """Convert a sparse dict to a :class:`qdrant_client.models.SparseVector`.

    Parameters
    ----------
    sparse_dict:
        Dict with ``"indices"`` (list of ints) and ``"values"``
        (list of floats) as returned by :meth:`SparseEncoder.encode`.

    Returns
    -------
    qdrant_client.models.SparseVector
    """
    from qdrant_client.models import SparseVector

    return SparseVector(
        indices=sparse_dict["indices"],
        values=sparse_dict["values"],
    )


__all__ = ["SparseEncoder", "sparse_to_qdrant"]
