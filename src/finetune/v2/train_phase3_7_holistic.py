"""Phase 3.7 — Holistic Reasoning SFT.

Trains LoRA adapters (projection frozen from Phase 3.5) on holistic reasoning
data covering synthesis, intent alignment, depth calibration, and domain
accuracy.  This phase uses a longer sequence length (8192) to accommodate
multi-document reasoning tasks.

Data source: ``holistic_training.jsonl`` produced by the V2 data pipeline.

Typical wall-time: ~8-10 hours on a single A100-80 GB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HolisticConfig:
    """Hyperparameters for Phase 3.7 holistic reasoning SFT."""

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Optimiser
    learning_rate: float = 8e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training loop
    epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Sequence length — longer than other phases for multi-doc reasoning
    max_seq_length: int = 8192

    # Precision
    bf16: bool = True

    # Checkpointing
    checkpoint_steps: int = 300

    # Paths
    data_dir: Path = Path("finetune_data/v2/holistic")
    phase35_dir: Path = Path("finetune_artifacts/v2/phase3_5")
    output_dir: Path = Path("finetune_artifacts/v2/phase3_7")

    # Quality gates — minimum metrics to proceed beyond Phase 3.7
    gate_synthesis_coherence: float = 0.80
    gate_intent_alignment: float = 0.85
    gate_depth_calibration: float = 0.75
    gate_domain_accuracy: float = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_training_args(config: HolisticConfig, output_dir: Path) -> Dict[str, Any]:
    """Build an SFTConfig-compatible training arguments dictionary.

    Parameters
    ----------
    config:
        Phase 3.7 holistic reasoning SFT training configuration.
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
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "max_seq_length": config.max_seq_length,
        "bf16": config.bf16,
        "fp16": False,
        "logging_steps": 25,
        "save_steps": config.checkpoint_steps,
        "report_to": "none",
        "seed": 42,
    }


def _load_holistic_dataset(data_dir: Path):
    """Load the holistic reasoning training dataset from *data_dir*.

    Expects a ``holistic_training.jsonl`` file under *data_dir*.  Each record
    should have ``prompt`` and ``completion`` keys covering multi-document
    reasoning, synthesis, and depth-calibrated responses.

    Returns ``None`` and logs a warning when the file is absent or unreadable.
    """
    from datasets import load_dataset  # type: ignore

    jsonl_path = data_dir / "holistic_training.jsonl"
    if not jsonl_path.exists():
        logger.warning(
            "Holistic dataset not found: %s — skipping dataset load.", jsonl_path
        )
        return None

    try:
        ds = load_dataset("json", data_files=str(jsonl_path), split="train")
        logger.info(
            "Loaded %d holistic training samples from %s.", len(ds), jsonl_path
        )
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load holistic dataset (%s) — skipping.", exc)
        return None


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def run_phase3_7(
    config: Optional[HolisticConfig] = None,
    *,
    phase35_dir: Optional[Path] = None,
) -> Path:
    """Execute Phase 3.7 holistic reasoning SFT.

    1. Loads the model from the Phase 3.5 checkpoint directory.
    2. Freezes the projection (preserve doc-intel + tool-calling + insight alignment).
    3. Applies LoRA adapters for holistic reasoning fine-tuning.
    4. Loads the holistic training dataset (``holistic_training.jsonl``).
    5. Trains with ``trl.SFTTrainer`` using the config hyperparameters.
    6. Saves the final checkpoint and writes a ``.phase3_7_complete`` marker.

    Parameters
    ----------
    config:
        Holistic SFT training configuration.  Uses defaults if ``None``.
    phase35_dir:
        Override path to the Phase 3.5 output directory.  Falls back to
        ``finetune_artifacts/v2/phase3_5`` when not provided.

    Returns
    -------
    Path to the output directory containing the holistic-reasoning-tuned checkpoint.
    """
    if config is None:
        config = HolisticConfig()

    p35_dir = phase35_dir or config.phase35_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 3.7: Holistic Reasoning SFT ===")
    logger.info(
        "LoRA r=%d  alpha=%d  LR=%s  epochs=%d  batch=%d  grad_accum=%d  max_seq=%d",
        config.lora_r,
        config.lora_alpha,
        config.learning_rate,
        config.epochs,
        config.per_device_batch_size,
        config.gradient_accumulation_steps,
        config.max_seq_length,
    )

    # --- Load model from Phase 3.5 output ------------------------------------
    from .vision_graft import GraftConfig, VisionGraftedModel

    graft_cfg = GraftConfig(freeze_vision=True, freeze_text=False)
    model = VisionGraftedModel(graft_cfg)
    model.load_vision_encoder()

    proj_ckpt = p35_dir / "projection.pt"
    model.load_projection(checkpoint=proj_ckpt)

    # Freeze projection — keep doc-intel + tool-calling + insight alignment intact
    if model._projection is not None:
        for p in model._projection.parameters():
            p.requires_grad = False

    model.load_text_model()
    model.add_lora(r=config.lora_r, lora_alpha=config.lora_alpha)

    # --- Load holistic training dataset --------------------------------------
    dataset = _load_holistic_dataset(config.data_dir)

    # --- Build training args and run SFTTrainer ------------------------------
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
        "Starting holistic reasoning SFT for %d epochs (max_seq_length=%d)...",
        config.epochs,
        config.max_seq_length,
    )
    trainer.train()

    # --- Save checkpoint and completion marker --------------------------------
    checkpoint_dir = config.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model._text_model.save_pretrained(str(checkpoint_dir))
    if model._tokenizer is not None:
        model._tokenizer.save_pretrained(str(checkpoint_dir))

    marker = config.output_dir / ".phase3_7_complete"
    marker.touch()

    logger.info("Phase 3.7 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
