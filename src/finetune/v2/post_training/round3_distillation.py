"""Round 3 — Reasoning Distillation.

SFT fine-tuning on compressed reasoning examples to maintain quality while
improving inference speed.  Operates on the Round 2 output model.

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
    """Build an SFTConfig-compatible training arguments dictionary.

    Parameters
    ----------
    config:
        Round 3 distillation training configuration.
    output_dir:
        Directory where checkpoints and the final model are written.

    Returns
    -------
    dict suitable for unpacking into ``trl.SFTConfig``.
    """
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

    1. Loads the model from Round 2 output via ``AutoModelForCausalLM``.
    2. Loads compressed reasoning examples for distillation.
    3. Trains with ``trl.SFTTrainer`` for a single epoch.
    4. Saves the final checkpoint and writes a ``.round3_complete`` marker.

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

    r2_dir = round2_dir or config.round2_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Round 3: Reasoning Distillation ===")
    logger.info(
        "LR=%s  epochs=%d  batch=%d  grad_accum=%d",
        config.learning_rate,
        config.epochs,
        config.per_device_batch_size,
        config.gradient_accumulation_steps,
    )

    # --- Load model from Round 2 output --------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    model_path = r2_dir / "checkpoint_final"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
    )

    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- Load distillation dataset -------------------------------------------
    dataset = _load_distillation_dataset(r2_dir)

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
        tokenizer=tokenizer,
    )

    logger.info(
        "Starting reasoning distillation for %d epoch(s)...",
        config.epochs,
    )
    trainer.train()

    # --- Save checkpoint and completion marker --------------------------------
    checkpoint_dir = config.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    marker = config.output_dir / ".round3_complete"
    marker.touch()

    logger.info("Round 3 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
