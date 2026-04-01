"""Phase 2.5 — DPO Contrastive Preference Training.

Teaches the model to distinguish high-quality extractions from corrupted ones
by training on chosen/rejected preference pairs via Direct Preference Optimisation.

Loads from the Phase 2 checkpoint and uses the format_dpo_pair helper from
dataset_preprocess.py to structure each contrastive pair.

Typical wall-time: ~3-5 hours on a single A100-80 GB.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DPOPhaseConfig:
    """Hyperparameters for Phase 2.5 DPO contrastive preference training."""

    # DPO-specific
    beta: float = 0.1

    # Optimiser
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10

    # Training loop
    epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16

    # Sequence lengths
    max_prompt_length: int = 2048
    max_response_length: int = 2048

    # Precision
    bf16: bool = True

    # Output
    output_dir: Path = Path("runs/v2/phase2_5_dpo")

    # Quality gates — minimum thresholds before proceeding to Phase 3
    gate_hallucination_rate: float = 0.05
    gate_extraction_f1_improvement: float = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dpo_training_args(config: DPOPhaseConfig, output_dir: Path) -> Dict[str, Any]:
    """Build a DPOConfig-compatible training arguments dictionary.

    Parameters
    ----------
    config:
        Phase 2.5 DPO training configuration.
    output_dir:
        Directory where checkpoints and the final model are written.

    Returns
    -------
    dict suitable for unpacking into ``trl.DPOConfig``.
    """
    max_length = config.max_prompt_length + config.max_response_length
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "beta": config.beta,
        "max_prompt_length": config.max_prompt_length,
        "max_length": max_length,
        "bf16": config.bf16,
        "fp16": False,
        "logging_steps": 25,
        "save_steps": 200,
        "report_to": "none",
        "seed": 42,
    }


def corrupt_extraction(good: Dict[str, Any], *, seed: int = 0) -> Dict[str, Any]:
    """Programmatically corrupt a good extraction to create a rejected sample.

    Applies exactly 2 randomly chosen corruptions from the following set:
    - ``drop_entity``: remove a random entity from the entities list.
    - ``hallucinate_value``: replace a random field value with a fake string.
    - ``break_table``: empty a random cell in the tables structure.
    - ``wrong_field``: swap the values of two random fields.

    Parameters
    ----------
    good:
        A dict with at least the keys ``"entities"``, ``"tables"``, and
        ``"fields"``.  All three should be present; missing keys are treated
        as empty collections so corruptions affecting them become no-ops.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    A deep copy of *good* with 2 corruptions applied.  The returned dict is
    always different from the input (assuming the input has enough content for
    at least one corruption to take effect).
    """
    rng = random.Random(seed)
    corrupted: Dict[str, Any] = copy.deepcopy(good)

    # ------------------------------------------------------------------
    # Corruption helpers (operate in-place on `corrupted`)
    # ------------------------------------------------------------------

    def drop_entity() -> None:
        entities: List[Any] = corrupted.get("entities", [])
        if entities:
            idx = rng.randrange(len(entities))
            entities.pop(idx)

    def hallucinate_value() -> None:
        fields: Dict[str, Any] = corrupted.get("fields", {})
        if fields:
            key = rng.choice(list(fields.keys()))
            fields[key] = "__HALLUCINATED_VALUE__"

    def break_table() -> None:
        tables: List[Any] = corrupted.get("tables", [])
        if tables:
            table = rng.choice(tables)
            # A table may be a list-of-rows or a dict; handle both
            if isinstance(table, list) and table:
                row = rng.choice(table)
                if isinstance(row, list) and row:
                    col_idx = rng.randrange(len(row))
                    row[col_idx] = ""
                elif isinstance(row, dict) and row:
                    key = rng.choice(list(row.keys()))
                    row[key] = ""
            elif isinstance(table, dict):
                # Treat dict values as rows
                rows_key = next(iter(table), None)
                if rows_key is not None:
                    table[rows_key] = ""

    def wrong_field() -> None:
        fields: Dict[str, Any] = corrupted.get("fields", {})
        keys = list(fields.keys())
        if len(keys) >= 2:
            k1, k2 = rng.sample(keys, 2)
            fields[k1], fields[k2] = fields[k2], fields[k1]

    corruption_pool = [drop_entity, hallucinate_value, break_table, wrong_field]
    chosen_corruptions = rng.sample(corruption_pool, 2)
    for corruption_fn in chosen_corruptions:
        corruption_fn()

    return corrupted


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dpo_dataset(data_dir: Path):
    """Load the DPO preference dataset from *data_dir*.

    Expects a ``dpo_pairs.jsonl`` file under *data_dir*.  Each record should
    have ``prompt``, ``chosen``, and ``rejected`` keys (as produced by
    :func:`~src.finetune.v2.dataset_preprocess.format_dpo_pair`).

    Returns ``None`` and logs a warning when the file is absent or unreadable.
    """
    from datasets import load_dataset  # type: ignore

    jsonl_path = data_dir / "dpo_pairs.jsonl"
    if not jsonl_path.exists():
        logger.warning(
            "DPO dataset not found: %s — skipping dataset load.", jsonl_path
        )
        return None

    try:
        ds = load_dataset("json", data_files=str(jsonl_path), split="train")
        logger.info("Loaded %d DPO preference pairs from %s.", len(ds), jsonl_path)
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load DPO dataset (%s) — skipping.", exc)
        return None


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def run_phase2_5(
    config: Optional[DPOPhaseConfig] = None,
    *,
    phase2_dir: Optional[Path] = None,
) -> Path:
    """Execute Phase 2.5 DPO contrastive preference training.

    1. Loads the model from the Phase 2 checkpoint directory.
    2. Loads the DPO preference dataset (chosen vs rejected extraction pairs).
    3. Trains with ``trl.DPOTrainer`` using the config hyperparameters.
    4. Saves the final checkpoint and writes a ``.phase2_5_complete`` marker.

    Parameters
    ----------
    config:
        DPO training configuration.  Uses defaults if ``None``.
    phase2_dir:
        Override path to the Phase 2 output directory.  Falls back to
        ``runs/v2/phase2`` when not provided.

    Returns
    -------
    Path to the output directory containing the DPO-trained checkpoint.
    """
    if config is None:
        config = DPOPhaseConfig()

    p2_dir = phase2_dir or Path("runs/v2/phase2")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 2.5: DPO Contrastive Preference Training ===")
    logger.info(
        "beta=%.3f  LR=%s  epochs=%d  batch=%d  grad_accum=%d",
        config.beta,
        config.learning_rate,
        config.epochs,
        config.per_device_batch_size,
        config.gradient_accumulation_steps,
    )

    # --- Load model from Phase 2 output ---------------------------------------
    from .vision_graft import GraftConfig, VisionGraftedModel

    graft_cfg = GraftConfig(freeze_vision=True, freeze_text=False)
    model = VisionGraftedModel(graft_cfg)
    model.load_vision_encoder()

    proj_ckpt = p2_dir / "projection.pt"
    model.load_projection(checkpoint=proj_ckpt)
    model.load_text_model()

    # --- Load DPO dataset -----------------------------------------------------
    data_dir = p2_dir / "dpo_data"
    dataset = _load_dpo_dataset(data_dir)

    # --- Build training args and run DPOTrainer -------------------------------
    try:
        from trl import DPOConfig, DPOTrainer  # type: ignore
    except ImportError:
        logger.error("trl is not installed. Install with: pip install trl>=0.8")
        raise

    training_args_dict = _build_dpo_training_args(config, config.output_dir)
    dpo_cfg = DPOConfig(**training_args_dict)

    trainer = DPOTrainer(
        model=model._text_model,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=dataset,
        tokenizer=model._tokenizer,
    )

    logger.info("Starting DPO training for %d epochs...", config.epochs)
    trainer.train()

    # --- Save checkpoint and completion marker --------------------------------
    checkpoint_dir = config.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model._text_model.save_pretrained(str(checkpoint_dir))
    if model._tokenizer is not None:
        model._tokenizer.save_pretrained(str(checkpoint_dir))

    marker = config.output_dir / ".phase2_5_complete"
    marker.touch()

    logger.info("Phase 2.5 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
