"""Round 1 — Conversational DPO.

LoRA fine-tuning using Direct Preference Optimisation on conversational
preference pairs.  Operates on the merged model weights (loaded in 4-bit)
to refine conversational quality after the main training phases.

Data source: ``conversational_dpo.jsonl`` with chosen/rejected response pairs.

Typical wall-time: ~4-6 hours on a single A100-80 GB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Round1Config:
    """Hyperparameters for Round 1 conversational DPO."""

    # DPO
    learning_rate: float = 1e-6
    beta: float = 0.05

    # Training loop
    epochs: int = 2
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16

    # Sequence lengths
    max_prompt_length: int = 2048
    max_response_length: int = 2048

    # Precision
    bf16: bool = True

    # Model strategy — LoRA + 4-bit for memory efficiency
    use_gradient_checkpointing: bool = True

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0

    # Paths
    data_path: Path = Path("finetune_data/v2/post_training/conversational_dpo.jsonl")
    merged_model_dir: Path = Path("finetune_artifacts/v2/merged")
    output_dir: Path = Path("finetune_artifacts/v2/post_round1")

    # Quality gate
    gate_conversation_quality: float = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_training_args(config: Round1Config, output_dir: Path) -> Dict[str, Any]:
    """Build a DPOConfig-compatible training arguments dictionary."""
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "beta": config.beta,
        "max_prompt_length": config.max_prompt_length,
        "max_length": config.max_prompt_length + config.max_response_length,
        "bf16": config.bf16,
        "fp16": False,
        "gradient_checkpointing": config.use_gradient_checkpointing,
        "logging_steps": 25,
        "save_steps": 200,
        "report_to": "none",
        "seed": 42,
    }


def _load_dpo_dataset(data_path: Path):
    """Load the conversational DPO dataset.

    Expects a JSONL file with ``prompt``, ``chosen``, and ``rejected`` keys.

    Returns ``None`` and logs a warning when the file is absent or unreadable.
    """
    from datasets import load_dataset  # type: ignore

    if not data_path.exists():
        logger.warning(
            "DPO dataset not found: %s — skipping dataset load.", data_path
        )
        return None

    try:
        ds = load_dataset("json", data_files=str(data_path), split="train")
        logger.info("Loaded %d DPO training samples from %s.", len(ds), data_path)
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load DPO dataset (%s) — skipping.", exc)
        return None


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def run_round1(
    config: Optional[Round1Config] = None,
    *,
    merged_model_dir: Optional[Path] = None,
) -> Path:
    """Execute Round 1 conversational DPO.

    1. Loads the merged model in 4-bit with LoRA via Unsloth.
    2. Loads the conversational DPO dataset.
    3. Trains with ``trl.DPOTrainer``.
    4. Saves the merged checkpoint and writes a ``.round1_complete`` marker.

    Parameters
    ----------
    config:
        Round 1 DPO training configuration.  Uses defaults if ``None``.
    merged_model_dir:
        Override path to the merged model directory.

    Returns
    -------
    Path to the output directory containing the DPO-tuned checkpoint.
    """
    if config is None:
        config = Round1Config()

    model_dir = merged_model_dir or config.merged_model_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Round 1: Conversational DPO ===")
    logger.info(
        "LR=%s  beta=%s  epochs=%d  batch=%d  grad_accum=%d  lora_r=%d",
        config.learning_rate,
        config.beta,
        config.epochs,
        config.per_device_batch_size,
        config.gradient_accumulation_steps,
        config.lora_r,
    )

    # --- Load merged model in 4-bit with LoRA ---------------------------------
    from unsloth import FastLanguageModel  # type: ignore

    max_seq = config.max_prompt_length + config.max_response_length

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        dtype=None,
        load_in_4bit=True,
        max_seq_length=max_seq,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq,
    )

    # --- Load DPO dataset ----------------------------------------------------
    dataset = _load_dpo_dataset(config.data_path)

    # --- Build training args and run DPOTrainer ------------------------------
    try:
        from trl import DPOConfig, DPOTrainer  # type: ignore
    except ImportError:
        logger.error("trl is not installed. Install with: pip install trl>=0.8")
        raise

    training_args_dict = _build_training_args(config, config.output_dir)
    dpo_cfg = DPOConfig(**training_args_dict)

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info(
        "Starting conversational DPO for %d epochs (beta=%s)...",
        config.epochs,
        config.beta,
    )
    trainer.train()

    # --- Save merged checkpoint -----------------------------------------------
    checkpoint_dir = config.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Merging LoRA and saving to %s", checkpoint_dir)
    model.save_pretrained_merged(
        str(checkpoint_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    marker = config.output_dir / ".round1_complete"
    marker.touch()

    logger.info("Round 1 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
