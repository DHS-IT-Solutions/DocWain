"""Phase 2 — Document Intelligence SFT.

Trains the projection MLP + LoRA adapters on document-understanding data:
table extraction, layout analysis, OCR correction, and cross-reference tasks.

The projection is initialised from the Phase 1 checkpoint so vision-language
alignment is preserved while the model learns document-specific skills.

Typical wall-time: ~6-10 hours on a single A100-80 GB.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Phase2Config:
    """Hyperparameters for Phase 2 document intelligence SFT."""

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    learning_rate: float = 2e-5
    epochs: int = 8
    per_device_batch_size: int = 4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    bf16: bool = True
    lr_scheduler_type: str = "cosine"
    checkpoint_steps: int = 500

    # Curriculum
    curriculum_stages: int = 4

    # Dataset mix — proportions must sum to 1.0
    dataset_mix: Dict[str, float] = field(default_factory=lambda: {
        "table": 0.40,
        "layout": 0.25,
        "ocr": 0.20,
        "cross_ref": 0.15,
    })

    # Data
    data_dir: Path = Path("finetune_data/v2/doc_intelligence")
    phase1_checkpoint: Path = Path("finetune_artifacts/v2/phase1/projection.pt")

    # Output
    output_dir: Path = Path("finetune_artifacts/v2/phase2")
    save_steps: int = 500
    logging_steps: int = 25
    eval_steps: int = 500

    # Quality gates — minimum metrics to proceed to Phase 3
    gate_docvqa_accuracy: float = 0.75
    gate_table_f1: float = 0.80
    gate_layout_map: float = 0.70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_curriculum_epochs(config: Phase2Config) -> List[Tuple[int, int]]:
    """Divide total epochs into ``curriculum_stages`` equal ranges.

    Each stage gets a contiguous slice of epoch indices.  When epochs is not
    evenly divisible by stages, the last stage absorbs the remainder.

    Parameters
    ----------
    config:
        Phase 2 training configuration.

    Returns
    -------
    List of ``(start_epoch, end_epoch)`` tuples (end is exclusive), one per
    curriculum stage.

    Examples
    --------
    >>> cfg = Phase2Config(epochs=8, curriculum_stages=4)
    >>> _get_curriculum_epochs(cfg)
    [(0, 2), (2, 4), (4, 6), (6, 8)]
    """
    stages = max(1, config.curriculum_stages)
    base = config.epochs // stages
    remainder = config.epochs % stages

    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(stages):
        # Distribute the remainder across the last stages (one extra each)
        extra = 1 if i >= (stages - remainder) and remainder > 0 else 0
        end = start + base + extra
        if start < end:  # skip degenerate zero-width stages
            ranges.append((start, end))
        start = end

    return ranges


def _build_training_args(config: Phase2Config, output_dir: Path) -> Dict:
    """Build a SFTTrainer-compatible training arguments dictionary.

    Parameters
    ----------
    config:
        Phase 2 training configuration.
    output_dir:
        Directory where checkpoints and final model are written.

    Returns
    -------
    dict suitable for unpacking into ``trl.SFTConfig`` or
    ``transformers.TrainingArguments``.
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
        "bf16": config.bf16,
        "fp16": False,
        "logging_steps": config.logging_steps,
        "save_steps": config.checkpoint_steps,
        "eval_steps": config.eval_steps,
        "max_seq_length": config.max_seq_length,
        "dataset_text_field": "text",
        "report_to": "none",
        "seed": 42,
    }


def _load_stage_dataset(stage_name: str, data_dir: Path):
    """Load a JSONL dataset file for a given curriculum stage.

    Parameters
    ----------
    stage_name:
        Name of the dataset category (e.g. ``"table"``, ``"layout"``).
    data_dir:
        Root directory containing per-category JSONL files.

    Returns
    -------
    A ``datasets.Dataset`` if the file exists, otherwise ``None``.
    """
    from datasets import load_dataset  # type: ignore

    jsonl_path = data_dir / f"{stage_name}.jsonl"
    if not jsonl_path.exists():
        logger.warning(
            "Dataset file not found for stage '%s': %s — skipping stage.",
            stage_name,
            jsonl_path,
        )
        return None

    try:
        ds = load_dataset("json", data_files=str(jsonl_path), split="train")
        logger.info("Loaded %d examples for stage '%s'.", len(ds), stage_name)
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to load dataset for stage '%s' (%s) — skipping stage.",
            stage_name,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def run_phase2(
    config: Optional[Phase2Config] = None,
    *,
    phase1_checkpoint: Optional[Path] = None,
) -> Path:
    """Execute Phase 2 document intelligence SFT.

    1. Loads the vision encoder + projection from Phase 1.
    2. Loads the text model and applies LoRA adapters.
    3. Unfreezes the projection MLP for continued training.
    4. Iterates through curriculum stages, training on per-stage datasets.
    5. Saves adapters and updated projection to ``config.output_dir``.

    Parameters
    ----------
    config:
        Training configuration. Uses defaults if ``None``.
    phase1_checkpoint:
        Override path to the Phase 1 projection checkpoint.

    Returns
    -------
    Path to the output directory containing adapters and projection.
    """
    if config is None:
        config = Phase2Config()

    proj_ckpt = phase1_checkpoint or config.phase1_checkpoint
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 2: Document Intelligence SFT ===")
    logger.info(
        "LoRA r=%d  alpha=%d  LR=%s  epochs=%d  batch=%d  stages=%d",
        config.lora_r,
        config.lora_alpha,
        config.learning_rate,
        config.epochs,
        config.per_device_batch_size,
        config.curriculum_stages,
    )
    logger.info("Dataset mix: %s", config.dataset_mix)

    # --- Load model -----------------------------------------------------------
    from .vision_graft import GraftConfig, VisionGraftedModel

    graft_cfg = GraftConfig(freeze_vision=True, freeze_text=False)
    model = VisionGraftedModel(graft_cfg)
    model.load_vision_encoder()
    model.load_projection(checkpoint=proj_ckpt)
    model.load_text_model()
    model.add_lora(r=config.lora_r, lora_alpha=config.lora_alpha)

    # Unfreeze projection MLP so it continues to adapt alongside LoRA
    if model._projection is not None:
        for param in model._projection.parameters():
            param.requires_grad = True
        logger.info("Projection MLP unfrozen for Phase 2 training.")

    # --- Curriculum training loop --------------------------------------------
    curriculum = _get_curriculum_epochs(config)
    stage_names = list(config.dataset_mix.keys())
    logger.info(
        "Curriculum stages: %d  |  epoch ranges: %s",
        len(curriculum),
        curriculum,
    )

    try:
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError:
        logger.error(
            "trl is not installed. Install with: pip install trl>=0.8"
        )
        raise

    for stage_idx, (epoch_start, epoch_end) in enumerate(curriculum):
        stage_epochs = epoch_end - epoch_start

        # Round-robin through dataset categories by stage index
        stage_name = stage_names[stage_idx % len(stage_names)]
        logger.info(
            "--- Curriculum stage %d/%d: '%s'  epochs %d-%d ---",
            stage_idx + 1,
            len(curriculum),
            stage_name,
            epoch_start,
            epoch_end - 1,
        )

        dataset = _load_stage_dataset(stage_name, config.data_dir)
        if dataset is None:
            logger.warning(
                "Skipping stage %d ('%s') — no dataset available.",
                stage_idx + 1,
                stage_name,
            )
            continue

        stage_output_dir = config.output_dir / f"stage_{stage_idx + 1}_{stage_name}"
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        training_args_dict = _build_training_args(config, stage_output_dir)
        training_args_dict["num_train_epochs"] = stage_epochs

        sft_cfg = SFTConfig(**training_args_dict)

        trainer = SFTTrainer(
            model=model._text_model,
            tokenizer=model._tokenizer,
            train_dataset=dataset,
            args=sft_cfg,
        )

        logger.info(
            "Training stage %d ('%s') for %d epochs...",
            stage_idx + 1,
            stage_name,
            stage_epochs,
        )
        trainer.train()

        stage_ckpt = stage_output_dir / "checkpoint_final"
        model._text_model.save_pretrained(str(stage_ckpt))
        if model._tokenizer is not None:
            model._tokenizer.save_pretrained(str(stage_ckpt))
        logger.info(
            "Stage %d checkpoint saved to %s", stage_idx + 1, stage_ckpt
        )

    # --- Save final outputs ---------------------------------------------------
    model.save_all(config.output_dir)
    logger.info("Phase 2 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
