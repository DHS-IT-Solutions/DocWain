"""Round 2 — Confidence Calibration SFT.

SFT fine-tuning on confidence-calibrated examples so the model learns to
express appropriate uncertainty.  Operates on the Round 1 output model.

Data source: ``confidence_sft.jsonl`` with calibrated confidence examples.

Typical wall-time: ~3-4 hours on a single A100-80 GB.
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
class Round2Config:
    """Hyperparameters for Round 2 confidence calibration SFT."""

    # Optimiser
    learning_rate: float = 1e-6

    # Training loop
    epochs: int = 2
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Sequence length
    max_seq_length: int = 4096

    # Precision
    bf16: bool = True

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Paths
    data_path: Path = Path("finetune_data/v2/post_training/confidence_sft.jsonl")
    round1_dir: Path = Path("finetune_artifacts/v2/post_round1")
    output_dir: Path = Path("finetune_artifacts/v2/post_round2")

    # Quality gate — Expected Calibration Error (lower is better)
    gate_ece: float = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_training_args(config: Round2Config, output_dir: Path) -> Dict[str, Any]:
    """Build an SFTConfig-compatible training arguments dictionary.

    Parameters
    ----------
    config:
        Round 2 confidence SFT training configuration.
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


def _load_confidence_dataset(data_path: Path):
    """Load the confidence calibration SFT dataset.

    Expects a JSONL file with ``prompt`` and ``completion`` keys where
    completions include appropriately calibrated confidence expressions.

    Returns ``None`` and logs a warning when the file is absent or unreadable.
    """
    from datasets import load_dataset  # type: ignore

    if not data_path.exists():
        logger.warning(
            "Confidence dataset not found: %s — skipping dataset load.", data_path
        )
        return None

    try:
        ds = load_dataset("json", data_files=str(data_path), split="train")
        logger.info(
            "Loaded %d confidence training samples from %s.", len(ds), data_path
        )
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load confidence dataset (%s) — skipping.", exc)
        return None


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def run_round2(
    config: Optional[Round2Config] = None,
    *,
    round1_dir: Optional[Path] = None,
) -> Path:
    """Execute Round 2 confidence calibration SFT.

    1. Loads the model from Round 1 output via ``AutoModelForCausalLM``.
    2. Loads the confidence calibration SFT dataset.
    3. Trains with ``trl.SFTTrainer``.
    4. Saves the final checkpoint and writes a ``.round2_complete`` marker.

    Parameters
    ----------
    config:
        Round 2 confidence SFT training configuration.  Uses defaults if ``None``.
    round1_dir:
        Override path to the Round 1 output directory.

    Returns
    -------
    Path to the output directory containing the confidence-calibrated checkpoint.
    """
    if config is None:
        config = Round2Config()

    r1_dir = round1_dir or config.round1_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Round 2: Confidence Calibration SFT ===")
    logger.info(
        "LR=%s  epochs=%d  batch=%d  grad_accum=%d",
        config.learning_rate,
        config.epochs,
        config.per_device_batch_size,
        config.gradient_accumulation_steps,
    )

    # --- Load model from Round 1 output --------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    model_path = r1_dir / "checkpoint_final"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
    )

    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- Load confidence dataset ---------------------------------------------
    dataset = _load_confidence_dataset(config.data_path)

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
        "Starting confidence calibration SFT for %d epochs...",
        config.epochs,
    )
    trainer.train()

    # --- Save checkpoint and completion marker --------------------------------
    checkpoint_dir = config.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    marker = config.output_dir / ".round2_complete"
    marker.touch()

    logger.info("Round 2 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
