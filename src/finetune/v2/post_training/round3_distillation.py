"""Round 3 — Reasoning Distillation.

LoRA SFT fine-tuning on compressed reasoning examples to maintain quality while
improving inference speed.  Operates on the Round 2 output model loaded in 4-bit.

Data source: compressed reasoning examples derived from the full reasoning
traces, targeting shorter but equally accurate outputs.

Typical wall-time: ~2-3 hours on a single A100-80 GB.
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
class Round3Config:
    """Hyperparameters for Round 3 reasoning distillation."""

    # Optimiser
    learning_rate: float = 5e-7

    # Training loop
    epochs: int = 1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Sequence length
    max_seq_length: int = 4096

    # Precision
    bf16: bool = True

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0

    # Paths
    round2_dir: Path = Path("finetune_artifacts/v2/post_round2")
    output_dir: Path = Path("finetune_artifacts/v2/post_round3")

    # Quality gates
    gate_max_quality_drop: float = 0.03
    gate_min_speed_toks: float = 25.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_training_args(config: Round3Config, output_dir: Path) -> Dict[str, Any]:
    """Build an SFTConfig-compatible training arguments dictionary."""
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "max_seq_length": config.max_seq_length,
        "bf16": config.bf16,
        "fp16": False,
        "gradient_checkpointing": config.use_gradient_checkpointing,
        "logging_steps": 25,
        "save_steps": 200,
        "dataset_text_field": "text",
        "report_to": "none",
        "seed": 42,
    }


def _load_distillation_dataset(round2_dir: Path):
    """Load compressed reasoning examples for distillation.

    Looks for ``distillation_data.jsonl`` under *round2_dir* or a sibling
    data directory.

    Returns ``None`` and logs a warning when the file is absent or unreadable.
    """
    from datasets import load_dataset  # type: ignore

    jsonl_path = round2_dir / "distillation_data.jsonl"
    if not jsonl_path.exists():
        logger.warning(
            "Distillation dataset not found: %s — skipping dataset load.", jsonl_path
        )
        return None

    try:
        ds = load_dataset("json", data_files=str(jsonl_path), split="train")
        logger.info(
            "Loaded %d distillation training samples from %s.", len(ds), jsonl_path
        )
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load distillation dataset (%s) — skipping.", exc)
        return None


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def run_round3(
    config: Optional[Round3Config] = None,
    *,
    round2_dir: Optional[Path] = None,
) -> Path:
    """Execute Round 3 reasoning distillation.

    1. Loads the model from Round 2 output in 4-bit with LoRA via Unsloth.
    2. Loads compressed reasoning examples for distillation.
    3. Trains with ``trl.SFTTrainer`` for a single epoch.
    4. Saves the merged checkpoint and writes a ``.round3_complete`` marker.

    Parameters
    ----------
    config:
        Round 3 distillation training configuration.  Uses defaults if ``None``.
    round2_dir:
        Override path to the Round 2 output directory.

    Returns
    -------
    Path to the output directory containing the distilled checkpoint.
    """
    if config is None:
        config = Round3Config()

    r2_dir = (round2_dir or config.round2_dir).resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not (r2_dir / "checkpoint_final").exists():
        raise FileNotFoundError(
            f"Round 2 checkpoint not found: {r2_dir / 'checkpoint_final'}. "
            "Round 2 must complete successfully before Round 3."
        )

    logger.info("=== Round 3: Reasoning Distillation ===")
    logger.info(
        "LR=%s  epochs=%d  batch=%d  grad_accum=%d  lora_r=%d",
        config.learning_rate,
        config.epochs,
        config.per_device_batch_size,
        config.gradient_accumulation_steps,
        config.lora_r,
    )

    # --- Load model from Round 2 output in 4-bit with LoRA -------------------
    from unsloth import FastLanguageModel  # type: ignore

    model_path = (r2_dir / "checkpoint_final").resolve()
    logger.info("Loading Round 2 model from %s", model_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        dtype=None,
        load_in_4bit=True,
        max_seq_length=config.max_seq_length,
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
        max_seq_length=config.max_seq_length,
    )

    # --- Load distillation dataset -------------------------------------------
    dataset = _load_distillation_dataset(r2_dir)

    if dataset is None:
        logger.warning(
            "No distillation dataset available — skipping Round 3 training. "
            "The Round 2 model will be used as-is for final promotion."
        )
        marker = config.output_dir / ".round3_skipped"
        marker.touch()
        return config.output_dir

    # --- Build training args and run SFTTrainer ------------------------------
    try:
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError:
        logger.error("trl is not installed. Install with: pip install trl>=0.8")
        raise

    training_args_dict = _build_training_args(config, config.output_dir)
    sft_cfg = SFTConfig(**training_args_dict)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info(
        "Starting reasoning distillation for %d epoch(s)...",
        config.epochs,
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

    marker = config.output_dir / ".round3_complete"
    marker.touch()

    logger.info("Round 3 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
