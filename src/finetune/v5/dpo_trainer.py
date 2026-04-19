"""V5 DPO trainer — preferences over the SFT-completed 14B checkpoint.

Consumes V5-format DPO pairs (see ``transform_v4_to_v5.transform_dpo``):

    {"capability": str, "source": str, "system": "",
     "user": str, "chosen": str, "rejected": str, "difficulty": str}

The ``system`` field is always empty — V5 bakes identity into weights,
not prompts. Existing ``src/finetune/dpo_trainer.py`` hardcoded a
"You are DocWain..." system prompt and can't be used as-is without
regressing that property.

Training config per the V5 design spec (Layer 4):
    * LoRA r=128, α=32 (matching SFT so the adapters stack cleanly)
    * β=0.1
    * 2 epochs
    * AdamW, LR 5e-6 (lower than SFT — DPO is more sensitive)
    * warmup_ratio 0.03
    * bf16 on A100 80GB

Output: ``models/DocWain-14B-v5/`` — the final 14B V5 checkpoint.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _load_dpo_pairs(paths: List[str]) -> List[Dict[str, str]]:
    """Read V5-format DPO JSONL across multiple files. Empty-system required."""
    rows: List[Dict[str, str]] = []
    for path in paths:
        if not Path(path).exists():
            logger.warning("DPO corpus missing: %s", path)
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not (r.get("user") and r.get("chosen") and r.get("rejected")):
                    continue
                # Enforce empty system — training signal must not carry identity
                rows.append({
                    "prompt": r["user"],
                    "chosen": r["chosen"],
                    "rejected": r["rejected"],
                    "capability": r.get("capability", "unknown"),
                })
    return rows


def _apply_chat_template(tokenizer, user: str, assistant: str) -> str:
    """Format a (user, assistant) turn with the tokenizer's chat template.

    No system turn — identity in weights.
    """
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )


def train(
    *,
    base_model: str,
    corpus_paths: List[str],
    output_dir: str,
    lora_rank: int = 128,
    lora_alpha: int = 32,
    beta: float = 0.1,
    epochs: int = 2,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 5e-6,
    warmup_ratio: float = 0.03,
    dry_run: bool = False,
) -> None:
    """Run DPO on V5 preference pairs.

    ``dry_run`` loads models + runs ~5 optimizer steps to prove the
    pipeline without committing to the full 12h run.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    logger.info("v5 DPO start: base=%s corpus=%s", base_model, corpus_paths)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("loading base model (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    rows = _load_dpo_pairs(corpus_paths)
    if not rows:
        raise ValueError(f"No DPO pairs loaded from {corpus_paths}")
    if dry_run:
        rows = rows[:32]
    logger.info("loaded %d DPO pairs", len(rows))

    # Format for DPOTrainer: {prompt, chosen, rejected}
    # We wrap each side with the chat template so the tokenizer sees the
    # Qwen3 turn markers. DPOTrainer computes loss on the assistant turn.
    formatted: List[Dict[str, str]] = []
    for r in rows:
        prompt_text = _apply_chat_template(tokenizer, r["prompt"], "")
        # strip the trailing assistant-start so DPOTrainer can append chosen/rejected
        prompt_text = prompt_text.rsplit("<|im_start|>assistant", 1)[0]
        formatted.append({
            "prompt": prompt_text,
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        })
    ds = Dataset.from_list(formatted)

    peft_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=(0.05 if dry_run else float(epochs)),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        bf16=True,
        logging_steps=5,
        save_strategy=("no" if dry_run else "steps"),
        save_steps=500,
        beta=beta,
        max_length=4096,
        max_prompt_length=3072,
        remove_unused_columns=False,
    )
    # TRL 0.24+ renamed `tokenizer` to `processing_class`. Use the new kw.
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    t0 = time.monotonic()
    trainer.train()
    elapsed = time.monotonic() - t0

    if dry_run:
        logger.info("dry-run DPO completed in %.1fs (%d rows)", elapsed, len(rows))
        return

    logger.info("merging LoRA → dense weights, saving to %s", output_dir)
    model = trainer.model
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    logger.info("v5 DPO complete — final weights at %s (%.0fs)", output_dir, elapsed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True,
                    help="SFT-completed 14B checkpoint (e.g. models/DocWain-14B-v5-sft)")
    ap.add_argument("--pairs", required=True,
                    help="Comma-separated DPO JSONL files")
    ap.add_argument("--output", default="models/DocWain-14B-v5")
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(
        base_model=args.base,
        corpus_paths=args.pairs.split(","),
        output_dir=args.output,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
