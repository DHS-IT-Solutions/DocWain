"""Data generator base for DocWain V2 finetuning."""

from src.finetune.v2.data_generator.base import (
    DOCWAIN_SYSTEM_PROMPT,
    format_sft_example,
    format_dpo_example,
)

__all__ = [
    "DOCWAIN_SYSTEM_PROMPT",
    "format_sft_example",
    "format_dpo_example",
]
