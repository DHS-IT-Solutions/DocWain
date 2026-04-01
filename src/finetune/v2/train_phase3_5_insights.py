"""Phase 3.5 — Insight Generation SFT.

Trains LoRA adapters (projection frozen from Phase 3) on insight generation data
so the model learns to produce high-quality insights across five categories:
pattern recognition, anomaly detection, trend analysis, comparative analysis,
and gap analysis.

Data source: ``insight_training.jsonl`` produced by the V2 data pipeline.

Typical wall-time: ~5-7 hours on a single A100-80 GB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

INSIGHT_CATEGORIES = [
    "pattern_recognition",
    "anomaly_detection",
    "trend_analysis",
    "comparative_analysis",
    "gap_analysis",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InsightPhaseConfig:
    """Hyperparameters for Phase 3.5 insight generation SFT."""

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128

    # Optimiser
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10

    # Training loop
    epochs: int = 4
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Sequence length
    max_seq_length: int = 4096

    # Precision
    bf16: bool = True

    # Output
    output_dir: Path = Path("runs/v2/phase3_5_insights")

    # Quality gates — minimum metrics to proceed beyond Phase 3.5
    gate_insight_precision: float = 0.80
    gate_insight_recall: float = 0.60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_training_args(config: InsightPhaseConfig, output_dir: Path) -> Dict[str, Any]:
    """Build an SFTConfig-compatible training arguments dictionary.

    Parameters
    ----------
    config:
        Phase 3.5 insight SFT training configuration.
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
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "max_seq_length": config.max_seq_length,
        "bf16": config.bf16,
        "fp16": False,
        "logging_steps": 25,
        "save_steps": 200,
        "report_to": "none",
        "seed": 42,
    }


def _load_insight_dataset(data_dir: Path):
    """Load the insight training dataset from *data_dir*.

    Expects an ``insight_training.jsonl`` file under *data_dir*.  Each record
    should have ``prompt`` and ``completion`` keys covering one of the five
    insight categories defined in :data:`INSIGHT_CATEGORIES`.

    Returns ``None`` and logs a warning when the file is absent or unreadable.
    """
    from datasets import load_dataset  # type: ignore

    jsonl_path = data_dir / "insight_training.jsonl"
    if not jsonl_path.exists():
        logger.warning(
            "Insight dataset not found: %s — skipping dataset load.", jsonl_path
        )
        return None

    try:
        ds = load_dataset("json", data_files=str(jsonl_path), split="train")
        logger.info("Loaded %d insight training samples from %s.", len(ds), jsonl_path)
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load insight dataset (%s) — skipping.", exc)
        return None


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def run_phase3_5(
    config: Optional[InsightPhaseConfig] = None,
    *,
    phase3_dir: Optional[Path] = None,
) -> Path:
    """Execute Phase 3.5 insight generation SFT.

    1. Loads the model from the Phase 3 checkpoint directory.
    2. Freezes the projection (preserve doc-intel + tool-calling alignment).
    3. Applies LoRA adapters for insight generation fine-tuning.
    4. Loads the insight training dataset (``insight_training.jsonl``).
    5. Trains with ``trl.SFTTrainer`` using the config hyperparameters.
    6. Saves the final checkpoint and writes a ``.phase3_5_complete`` marker.

    Parameters
    ----------
    config:
        Insight SFT training configuration.  Uses defaults if ``None``.
    phase3_dir:
        Override path to the Phase 3 output directory.  Falls back to
        ``runs/v2/phase3`` when not provided.

    Returns
    -------
    Path to the output directory containing the insight-tuned checkpoint.
    """
    if config is None:
        config = InsightPhaseConfig()

    p3_dir = phase3_dir or Path("runs/v2/phase3")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 3.5: Insight Generation SFT ===")
    logger.info(
        "LoRA r=%d  alpha=%d  LR=%s  epochs=%d  batch=%d  grad_accum=%d",
        config.lora_r,
        config.lora_alpha,
        config.learning_rate,
        config.epochs,
        config.per_device_batch_size,
        config.gradient_accumulation_steps,
    )
    logger.info("Insight categories: %s", INSIGHT_CATEGORIES)

    # --- Load model from Phase 3 output ---------------------------------------
    from .vision_graft import GraftConfig, VisionGraftedModel

    graft_cfg = GraftConfig(freeze_vision=True, freeze_text=False)
    model = VisionGraftedModel(graft_cfg)
    model.load_vision_encoder()

    proj_ckpt = p3_dir / "projection.pt"
    model.load_projection(checkpoint=proj_ckpt)

    # Freeze projection — keep doc-intel + tool-calling alignment intact
    if model._projection is not None:
        for p in model._projection.parameters():
            p.requires_grad = False

    model.load_text_model()
    model.add_lora(r=config.lora_r, lora_alpha=config.lora_alpha)

    # --- Load insight training dataset ----------------------------------------
    data_dir = p3_dir / "insight_data"
    dataset = _load_insight_dataset(data_dir)

    # --- Build training args and run SFTTrainer -------------------------------
    try:
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError:
        logger.error("trl is not installed. Install with: pip install trl>=0.8")
        raise

    training_args_dict = _build_training_args(config, config.output_dir)
    sft_cfg = SFTConfig(**training_args_dict)

    trainer = SFTTrainer(
        model=model._text_model,
        args=sft_cfg,
        train_dataset=dataset,
        tokenizer=model._tokenizer,
    )

    logger.info(
        "Starting insight generation SFT for %d epochs across %d categories...",
        config.epochs,
        len(INSIGHT_CATEGORIES),
    )
    trainer.train()

    # --- Save checkpoint and completion marker --------------------------------
    checkpoint_dir = config.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model._text_model.save_pretrained(str(checkpoint_dir))
    if model._tokenizer is not None:
        model._tokenizer.save_pretrained(str(checkpoint_dir))

    marker = config.output_dir / ".phase3_5_complete"
    marker.touch()

    logger.info("Phase 3.5 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
