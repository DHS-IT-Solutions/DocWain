"""Base utilities for V2+ finetuning data generation.

Provides constants, a buffered JSONL writer, and formatting helpers for
SFT, DPO, and eval examples using the Qwen3 chat template.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCWAIN_SYSTEM_PROMPT: str = (
    "You are DocWain, an enterprise document intelligence assistant. "
    "You analyse documents with deep contextual understanding, extract "
    "structured information, identify patterns and anomalies, and provide "
    "holistic analysis grounded in evidence. You reason step-by-step before "
    "answering, state your confidence level, and cite specific sources. "
    "When information is insufficient, you say so clearly rather than guessing."
)

DOMAINS: List[str] = [
    "legal",
    "financial",
    "hr",
    "medical",
    "policy",
    "insurance",
    "procurement",
]

DOC_TYPES: List[str] = [
    "contract",
    "invoice",
    "purchase_order",
    "resume",
    "job_description",
    "policy_document",
    "compliance_report",
    "financial_statement",
    "audit_report",
    "medical_record",
    "insurance_claim",
    "patent",
    "lease_agreement",
    "employee_handbook",
    "meeting_minutes",
    "proposal",
    "sow",
    "nda",
]

# ---------------------------------------------------------------------------
# JSONLWriter
# ---------------------------------------------------------------------------

_DEFAULT_BUFFER_SIZE = 100


class JSONLWriter:
    """Buffered JSONL writer with context manager support.

    Creates parent directories automatically. Flushes buffer every
    ``buffer_size`` records and on close.
    """

    def __init__(self, path: str | Path, buffer_size: int = _DEFAULT_BUFFER_SIZE) -> None:
        self._path = Path(path)
        self._buffer_size = buffer_size
        self._buffer: List[Dict[str, Any]] = []
        self._count = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w", encoding="utf-8")

    # -- public API --

    def write(self, record: Dict[str, Any]) -> None:
        """Append a record to the buffer, flushing if full."""
        self._buffer.append(record)
        self._count += 1
        if len(self._buffer) >= self._buffer_size:
            self._flush()

    @property
    def count(self) -> int:
        """Total number of records written (including buffered)."""
        return self._count

    def close(self) -> None:
        """Flush remaining buffer and close the file handle."""
        self._flush()
        self._fh.close()

    # -- context manager --

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- internals --

    def _flush(self) -> None:
        for record in self._buffer:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._buffer.clear()
        self._fh.flush()


# ---------------------------------------------------------------------------
# Formatting helpers (Qwen3 chat template)
# ---------------------------------------------------------------------------


def format_sft_example(
    query: str,
    reasoning: str,
    answer: str,
    *,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """Format a supervised fine-tuning example using Qwen3 chat template.

    Returns a dict with a single ``text`` key containing the full
    conversation in ``<|im_start|>``/``<|im_end|>`` format with an
    embedded ``<think>`` block for chain-of-thought.
    """
    sys_prompt = system_prompt or DOCWAIN_SYSTEM_PROMPT
    text = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>"
    )
    return {"text": text}


def format_dpo_example(
    query: str,
    chosen_reasoning: str,
    chosen_answer: str,
    rejected_reasoning: str,
    rejected_answer: str,
    *,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """Format a DPO preference pair for TRL DPOTrainer.

    Returns a dict with ``prompt``, ``chosen``, and ``rejected`` keys.
    """
    sys_prompt = system_prompt or DOCWAIN_SYSTEM_PROMPT
    prompt = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    chosen = f"<think>\n{chosen_reasoning}\n</think>\n\n{chosen_answer}<|im_end|>"
    rejected = f"<think>\n{rejected_reasoning}\n</think>\n\n{rejected_answer}<|im_end|>"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def format_eval_example(
    benchmark: str,
    query: str,
    context: str,
    reference_answer: str,
    rubric: Dict[str, Any],
    *,
    expected_tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Format an evaluation benchmark example.

    Returns a dict with all eval fields needed by the eval runner.
    """
    return {
        "benchmark": benchmark,
        "query": query,
        "context": context,
        "reference_answer": reference_answer,
        "rubric": rubric,
        "expected_tools": expected_tools,
    }
