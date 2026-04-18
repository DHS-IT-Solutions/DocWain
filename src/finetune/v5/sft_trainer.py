"""DocWain V5 — 14B LoRA SFT trainer.

Trains a LoRA adapter on top of the V5 seed (MergeKit output) or the V3
weights when the seed is unavailable. The output is a merged bf16 dense
checkpoint ready for DPO.

V5 is explicitly an **identity-in-weights** model: the training system
turn is always empty. The trainer refuses to inject a DocWain persona
system prompt even when a row carries one — it forces ``system == ""``
before tokenisation. This is the single most important behavioural
difference from the V4 trainer in ``src/finetune/docwain_finetune.py``.

Curriculum ordering follows
``src.finetune.v5.capability_charter.CURRICULUM_ORDER``. Within an
epoch, each capability bucket is shuffled internally, then buckets are
concatenated in CURRICULUM_ORDER. A ``SequentialSampler`` is forced on
the Trainer so this ordering survives the DataLoader.

Checkpointing is wall-clock based — every 6 h of training time (not
steps), because steps on a 14B with grad-accum 16 take minutes not
seconds and a step-based budget drifts badly. See
``WallClockCheckpointCallback``.

Auto-escalation: if the loss's 500-step moving average stops dropping
(delta < 0.01) by the end of epoch 1 and is still above 0.25, the
trainer logs a warning, unloads the r=128 adapter, re-wraps with
r=256, and runs epoch 2. This is documented in the V5 design spec
(section "Risks").

What was reused from V2
-----------------------
* Dataset loading shape (JSONL streamed via ``datasets.load_dataset``)
  — matches ``src/finetune/v2/train_track.py``.
* ``_TrainerWithSampler`` pattern mirrors the curriculum-ordered sampler
  override from ``src/finetune/v2/train_track.py``'s
  ``CurriculumSampler``.
* The merge/save flow (LoRA → ``merge_and_unload`` → ``save_pretrained``)
  is borrowed from ``src/finetune/docwain_finetune.py`` — the V2 variant
  used Unsloth's ``save_pretrained_merged``; V5 uses plain peft/HF
  because the 14B fp16 merge fits on an A100 80GB without Unsloth's
  4-bit gymnastics.

What is *new* in V5
-------------------
* Empty-system tokenisation.
* Capability-aware curriculum sampler.
* Wall-clock-based checkpoint callback.
* Auto-rank-escalation on loss plateau.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Repo root on path for ``python -m src.finetune.v5.sft_trainer``
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# peft's LoRA dispatcher iterates through optional backends (awq, bnb, eetq,
# hqq, ...) and imports them speculatively. On this host ``awq`` is
# installed but incompatible with the pinned transformers version
# (PytorchGELUTanh was renamed). We're not training an AWQ-quantised base,
# so pre-poison the import to keep peft from touching it.
sys.modules.setdefault("awq", None)  # type: ignore[assignment]

from src.finetune.v5.capability_charter import CHARTER, CURRICULUM_ORDER  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _parse_interval(spec: str) -> float:
    """Parse ``6h`` / ``30m`` / ``3600s`` / ``3600`` → seconds (float)."""
    s = str(spec).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hms]?)", s)
    if not m:
        raise ValueError(f"Bad interval spec: {spec!r}")
    n = float(m.group(1))
    unit = m.group(2) or "s"
    mult = {"h": 3600.0, "m": 60.0, "s": 1.0}[unit]
    return n * mult


@dataclass
class V5SFTConfig:
    """Arguments for a V5 SFT run (dry or full)."""

    base_model: str
    corpus_paths: List[str]
    output_dir: str
    lora_rank: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    max_seq_length: int = 4096
    checkpoint_interval_s: float = 6 * 3600.0
    plateau_rank_bump: int = 256
    plateau_loss_floor: float = 0.25
    plateau_delta_threshold: float = 0.01
    plateau_window: int = 500
    seed: int = 42
    dry_run: bool = False
    dry_run_steps: int = 10


# ---------------------------------------------------------------------------
# Dataset loading + tokenisation
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_v5_rows(paths: Sequence[str]) -> List[Dict]:
    """Load all V5 rows across the given JSONL files.

    Each row must carry: ``capability``, ``user``, ``assistant``. Missing
    ``system`` is treated as empty (V5 requirement). Rows without a
    valid capability in the charter are dropped with a warning.
    """
    all_rows: List[Dict] = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            logger.warning("corpus %s does not exist — skipping", pp)
            continue
        count = 0
        for row in _iter_jsonl(pp):
            cap = row.get("capability") or ""
            if cap not in CHARTER:
                continue
            user = (row.get("user") or "").strip()
            assistant = (row.get("assistant") or "").strip()
            if not user or not assistant:
                continue
            all_rows.append({
                "capability": cap,
                "system": "",  # V5 hard invariant
                "user": user,
                "assistant": assistant,
                "difficulty": row.get("difficulty", "medium"),
            })
            count += 1
        logger.info("loaded %d usable rows from %s", count, pp)
    return all_rows


def curriculum_order_rows(rows: List[Dict], seed: int) -> List[Dict]:
    """Re-order rows so curriculum buckets appear in CURRICULUM_ORDER.

    Within each bucket rows are shuffled deterministically off ``seed``.
    Rows tagged with capabilities not in CURRICULUM_ORDER fall to the
    end (shouldn't happen once charter stabilises).
    """
    rng = random.Random(seed)
    by_cap: Dict[str, List[Dict]] = {cap: [] for cap in CURRICULUM_ORDER}
    leftovers: List[Dict] = []
    for r in rows:
        cap = r["capability"]
        if cap in by_cap:
            by_cap[cap].append(r)
        else:
            leftovers.append(r)
    for cap in by_cap:
        rng.shuffle(by_cap[cap])
    rng.shuffle(leftovers)
    ordered: List[Dict] = []
    for cap in CURRICULUM_ORDER:
        bucket = by_cap[cap]
        ordered.extend(bucket)
        if bucket:
            logger.info("curriculum bucket %-25s size=%d", cap, len(bucket))
    if leftovers:
        logger.warning("found %d rows with unknown capability — appending",
                       len(leftovers))
        ordered.extend(leftovers)
    return ordered


def build_tokenised_dataset(rows: List[Dict], tokenizer, max_seq_length: int):
    """Turn rows into a HuggingFace Dataset of ``input_ids``/``labels``.

    Uses the tokenizer's own ``apply_chat_template``. ``system`` is
    always injected as an empty turn — the V5 identity invariant.
    We mask prompt tokens (-100) so loss is measured only on the
    assistant response, which is standard SFT practice and makes
    the 0.25 plateau threshold in the spec comparable with V2 numbers.
    """
    from datasets import Dataset

    def _encode(row):
        # Full conversation with an EMPTY system turn.
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["assistant"]},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_seq_length,
        )

        # Prompt-only (system + user + generation prompt, no assistant).
        prompt_messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": row["user"]},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_seq_length,
        )

        input_ids = list(full_ids)
        labels = list(full_ids)
        n_prompt = min(len(prompt_ids), len(labels))
        for i in range(n_prompt):
            labels[i] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "capability": row["capability"],
        }

    ds = Dataset.from_list(rows)
    ds = ds.map(
        _encode,
        remove_columns=ds.column_names,
        desc="tokenising",
        load_from_cache_file=False,
    )
    return ds


# ---------------------------------------------------------------------------
# Collator (pads to the longest in batch, preserves -100 on labels)
# ---------------------------------------------------------------------------


class _CausalPadCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        input_ids, attn, labels = [], [], []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_id] * pad_n)
            attn.append(list(f["attention_mask"]) + [0] * pad_n)
            labels.append(list(f["labels"]) + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _make_callbacks(cfg: V5SFTConfig, log_path: Path, plateau_tracker):
    """Build the set of TrainerCallbacks used by the trainer.

    Defined inside a function because ``transformers`` is an optional
    import at module load — keeps ``--help`` cheap.
    """
    from transformers import TrainerCallback

    class JsonlLogCallback(TrainerCallback):
        """Write every log entry to ``sft_training.log`` as a JSON line."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            entry = {
                "ts": time.time(),
                "step": state.global_step,
                "epoch": state.epoch,
            }
            for k, v in logs.items():
                if isinstance(v, (int, float, str, bool)):
                    entry[k] = v
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except OSError:
                pass
            # Track rolling loss for plateau detection.
            if "loss" in logs and isinstance(logs["loss"], (int, float)):
                plateau_tracker.add(state.global_step, float(logs["loss"]))

    class WallClockCheckpointCallback(TrainerCallback):
        """Save a checkpoint every ``checkpoint_interval_s`` seconds."""

        def __init__(self, interval_s: float, out_root: Path):
            self.interval_s = interval_s
            self.out_root = out_root
            self.start_ts: Optional[float] = None
            self.last_ckpt_ts: Optional[float] = None

        def on_train_begin(self, args, state, control, **kwargs):
            now = time.time()
            self.start_ts = now
            self.last_ckpt_ts = now

        def on_step_end(self, args, state, control, **kwargs):
            if self.last_ckpt_ts is None:
                return
            if time.time() - self.last_ckpt_ts >= self.interval_s:
                control.should_save = True
                self.last_ckpt_ts = time.time()

    return [
        JsonlLogCallback(),
        WallClockCheckpointCallback(
            cfg.checkpoint_interval_s, Path(cfg.output_dir) / "checkpoints",
        ),
    ]


class PlateauTracker:
    """Tiny moving-average tracker for auto-rank-escalation."""

    def __init__(self, window: int, delta_threshold: float, loss_floor: float):
        self.window = window
        self.delta_threshold = delta_threshold
        self.loss_floor = loss_floor
        self._losses: "deque[float]" = deque(maxlen=window)
        self._prev_avg: Optional[float] = None
        self._first_avg: Optional[float] = None

    def add(self, step: int, loss: float) -> None:
        self._losses.append(loss)
        if len(self._losses) == self.window and self._first_avg is None:
            self._first_avg = sum(self._losses) / self.window

    def avg(self) -> Optional[float]:
        if not self._losses:
            return None
        return sum(self._losses) / len(self._losses)

    def is_plateau(self) -> bool:
        """True if we have a full window and delta < threshold and avg above floor."""
        if len(self._losses) < self.window or self._first_avg is None:
            return False
        current = self.avg() or 0.0
        dropped = self._first_avg - current
        return current > self.loss_floor and dropped < self.delta_threshold


# ---------------------------------------------------------------------------
# Model wiring
# ---------------------------------------------------------------------------


def _linear_target_modules(model) -> List[str]:
    """Return every ``nn.Linear`` leaf name for full-coverage LoRA."""
    import torch.nn as nn

    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            # Skip the language-model head — untying it with LoRA hurts.
            if leaf in {"lm_head", "embed_tokens"}:
                continue
            names.add(leaf)
    return sorted(names)


def load_model_and_tokenizer(cfg: V5SFTConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading base model from %s", cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def wrap_with_lora(model, rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    targets = _linear_target_modules(model)
    logger.info("LoRA target modules (%d): %s", len(targets), targets)
    peft_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    peft_model = get_peft_model(model, peft_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


# ---------------------------------------------------------------------------
# Trainer with forced sequential sampler (preserves curriculum order)
# ---------------------------------------------------------------------------


def _build_trainer(
    cfg: V5SFTConfig,
    model,
    tokenizer,
    train_dataset,
    callbacks,
    *,
    max_steps: Optional[int] = None,
    num_epochs: Optional[float] = None,
):
    import torch
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import SequentialSampler

    ckpt_root = Path(cfg.output_dir) / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args_kwargs = dict(
        output_dir=str(ckpt_root),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=1,
        save_strategy="no",  # we drive saves via WallClockCheckpointCallback
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
        seed=cfg.seed,
        dataloader_drop_last=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=False,  # already enabled on the model
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )
    if max_steps is not None:
        training_args_kwargs["max_steps"] = int(max_steps)
        training_args_kwargs["num_train_epochs"] = 1  # ignored when max_steps > 0
    else:
        training_args_kwargs["num_train_epochs"] = float(num_epochs or cfg.epochs)

    args = TrainingArguments(**training_args_kwargs)

    class _SequentialTrainer(Trainer):
        """Force sequential data order so curriculum bucket order survives."""

        def _get_train_sampler(self, *a, **kw):  # type: ignore[override]
            return SequentialSampler(self.train_dataset)

    collator = _CausalPadCollator(tokenizer)

    trainer = _SequentialTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Route checkpoint save to our custom wall-clock path.
    out_root = ckpt_root

    def _save_timed(*a, **kw):
        ts = int(time.time())
        path = out_root / f"t_{ts}"
        path.mkdir(parents=True, exist_ok=True)
        logger.info("wall-clock checkpoint → %s", path)
        trainer.save_model(str(path))
        tokenizer.save_pretrained(str(path))

    trainer._save_timed_ckpt = _save_timed  # type: ignore[attr-defined]

    # Monkey-patch Trainer._save_checkpoint to our timed saver when
    # should_save was flipped by the wall-clock callback.
    original_save = trainer._save_checkpoint

    def _patched_save(model_, trial, *args_, **kwargs_):
        try:
            trainer._save_timed_ckpt()
        except Exception as exc:
            logger.warning("wall-clock save failed, falling back: %s", exc)
            return original_save(model_, trial, *args_, **kwargs_)

    trainer._save_checkpoint = _patched_save  # type: ignore[assignment]
    return trainer


# ---------------------------------------------------------------------------
# Merge + final save
# ---------------------------------------------------------------------------


def merge_and_save(model, tokenizer, out_dir: Path) -> Path:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("merging LoRA into base weights → %s", out_dir)
    # peft models expose merge_and_unload; plain models don't need merging.
    merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------


def run(cfg: V5SFTConfig) -> Dict:
    import torch

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("finetune_artifacts/v5/sft_training.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist the resolved config for audit.
    with (out_dir / "sft_config.json").open("w", encoding="utf-8") as f:
        json.dump({
            **{k: v for k, v in cfg.__dict__.items() if k != "corpus_paths"},
            "corpus_paths": list(cfg.corpus_paths),
        }, f, indent=2)

    # --- data -------------------------------------------------------------
    rows = load_v5_rows(cfg.corpus_paths)
    if not rows:
        raise RuntimeError("No usable rows in any corpus file.")
    ordered = curriculum_order_rows(rows, seed=cfg.seed)
    logger.info("total usable rows: %d", len(ordered))

    # --- model ------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(cfg)
    tokenised = build_tokenised_dataset(ordered, tokenizer, cfg.max_seq_length)
    logger.info("tokenised dataset: %d examples", len(tokenised))

    # Trim dataset for dry-run so step count × seqlen is bounded.
    if cfg.dry_run:
        keep = min(
            len(tokenised),
            cfg.dry_run_steps * cfg.batch_size * cfg.grad_accum,
        )
        tokenised = tokenised.select(range(keep))
        logger.info("dry-run: trimmed dataset to %d rows", len(tokenised))

    model = wrap_with_lora(
        model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout,
    )

    plateau = PlateauTracker(
        window=cfg.plateau_window,
        delta_threshold=cfg.plateau_delta_threshold,
        loss_floor=cfg.plateau_loss_floor,
    )
    callbacks = _make_callbacks(cfg, log_path, plateau)

    t_start = time.time()
    peak_mem_mb = 0.0
    loss_trajectory: List[Dict] = []

    # --- training ---------------------------------------------------------
    if cfg.dry_run:
        # Single-epoch, max_steps=N. Proves the pipeline without burning
        # hours.
        trainer = _build_trainer(
            cfg, model, tokenizer, tokenised, callbacks,
            max_steps=cfg.dry_run_steps,
        )
        logger.info("dry-run: %d steps", cfg.dry_run_steps)
        result = trainer.train()
        loss_trajectory = _extract_loss_trajectory(log_path, since=t_start)
        _save_dryrun_checkpoint(trainer, tokenizer, out_dir)
    else:
        # Epoch 1.
        trainer = _build_trainer(
            cfg, model, tokenizer, tokenised, callbacks, num_epochs=1,
        )
        logger.info("epoch 1: %d rows, lora_r=%d", len(tokenised), cfg.lora_rank)
        trainer.train()
        loss_trajectory = _extract_loss_trajectory(log_path, since=t_start)

        # Plateau check before epoch 2.
        effective_rank = cfg.lora_rank
        if plateau.is_plateau():
            logger.warning(
                "LOSS PLATEAU DETECTED at end of epoch 1 "
                "(avg=%.3f, floor=%.2f). Re-wrapping with rank=%d for epoch 2.",
                plateau.avg() or 0.0, cfg.plateau_loss_floor, cfg.plateau_rank_bump,
            )
            model = trainer.model
            if hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
            model = wrap_with_lora(
                model, cfg.plateau_rank_bump, cfg.lora_alpha, cfg.lora_dropout,
            )
            effective_rank = cfg.plateau_rank_bump
            trainer = _build_trainer(
                cfg, model, tokenizer, tokenised, callbacks, num_epochs=1,
            )

        # Epoch 2.
        if cfg.epochs >= 2:
            logger.info("epoch 2: effective_rank=%d", effective_rank)
            trainer.train()

    # --- peak memory ------------------------------------------------------
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # --- merge + save (only for full runs or dry-run smoke proof) --------
    if not cfg.dry_run:
        merge_and_save(trainer.model, tokenizer, out_dir)

    wall = time.time() - t_start
    summary = {
        "wall_time_s": wall,
        "steps": len(loss_trajectory),
        "peak_mem_mb": peak_mem_mb,
        "loss_trajectory": loss_trajectory,
        "device": _describe_device(),
        "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
        "lora_rank_initial": cfg.lora_rank,
        "dry_run": cfg.dry_run,
    }
    with (out_dir / "sft_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("done. wall=%.1fs peak_mem=%.1fMB", wall, peak_mem_mb)
    return summary


def _save_dryrun_checkpoint(trainer, tokenizer, out_dir: Path) -> None:
    """Snapshot a usable checkpoint from a dry-run so we can prove the pipeline."""
    ckpt = out_dir / "checkpoints" / f"dryrun_t_{int(time.time())}"
    ckpt.mkdir(parents=True, exist_ok=True)
    try:
        trainer.save_model(str(ckpt))
        tokenizer.save_pretrained(str(ckpt))
        logger.info("dry-run checkpoint saved → %s", ckpt)
    except Exception as exc:
        logger.warning("dry-run checkpoint save failed: %s", exc)


def _extract_loss_trajectory(log_path: Path, since: float) -> List[Dict]:
    out: List[Dict] = []
    if not log_path.exists():
        return out
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("ts", 0) < since:
                continue
            if "loss" not in entry:
                continue
            out.append({
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "loss": entry.get("loss"),
            })
    return out


def _describe_device() -> Dict:
    try:
        import torch
    except Exception:
        return {"device": "cpu", "reason": "torch unavailable"}
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    idx = torch.cuda.current_device()
    return {
        "device": f"cuda:{idx}",
        "name": torch.cuda.get_device_name(idx),
        "total_mem_mb": torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2),
        "bf16": torch.cuda.is_bf16_supported(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> V5SFTConfig:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="Base model path (seed or V3)")
    ap.add_argument(
        "--corpus", required=True,
        help="Comma-separated JSONL paths (concatenated as one dataset)",
    )
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--checkpoint-interval", default="6h",
                    help="Wall-clock checkpoint cadence (e.g. 6h, 30m)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Run ~10 steps to prove the pipeline, then save a "
                         "checkpoint and a run summary. Does not merge.")
    ap.add_argument("--dry-run-steps", type=int, default=10)
    args = ap.parse_args()

    return V5SFTConfig(
        base_model=args.base,
        corpus_paths=[p.strip() for p in args.corpus.split(",") if p.strip()],
        output_dir=args.output,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        checkpoint_interval_s=_parse_interval(args.checkpoint_interval),
        seed=args.seed,
        dry_run=args.dry_run,
        dry_run_steps=args.dry_run_steps,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    cfg = _parse_args()
    logger.info("V5 SFT trainer starting — dry_run=%s", cfg.dry_run)
    summary = run(cfg)

    if cfg.dry_run:
        report_path = Path("finetune_artifacts/v5/sft_dryrun_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump({
                "base_model": cfg.base_model,
                "output_dir": cfg.output_dir,
                "dry_run_steps": cfg.dry_run_steps,
                **summary,
            }, f, indent=2)
        logger.info("dry-run report → %s", report_path)


if __name__ == "__main__":
    main()
