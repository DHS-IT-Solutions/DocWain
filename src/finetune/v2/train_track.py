"""Unified per-track training script for DocWain V2.

Handles SFT (+ optional DPO) for any track, merges to FP16,
exports GGUF, and updates the Ollama model.

Usage::

    python -m src.finetune.v2.train_track --track excel_csv \\
        --data-path finetune_data/v2/excel_csv_sft.jsonl \\
        --output-dir finetune_artifacts/v2_upgrade/excel_csv

    # Or import and call directly:
    from src.finetune.v2.train_track import train_track, TrackTrainingConfig
    cfg = TrackTrainingConfig(track_name="excel_csv", data_path="...", output_dir="...")
    checkpoint = train_track(cfg)
"""

from __future__ import annotations

# Patch llm_blender compatibility with latest transformers (TRANSFORMERS_CACHE removed)
import transformers.utils.hub as _hub
if not hasattr(_hub, "TRANSFORMERS_CACHE"):
    _hub.TRANSFORMERS_CACHE = __import__("os").path.join(
        __import__("os").path.expanduser("~"), ".cache", "huggingface", "hub"
    )

import glob
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modelfile reference path
# ---------------------------------------------------------------------------

_V1_MODELFILE = Path("models/v1_backup/Modelfile.v1")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrackTrainingConfig:
    """Configuration for training a single track."""

    track_name: str = ""
    base_model: str = "unsloth/Qwen3-14B-bnb-4bit"
    base_checkpoint: Optional[str] = None  # path to previous track's merged checkpoint
    data_path: str = ""  # path to SFT JSONL
    dpo_path: Optional[str] = None  # optional DPO JSONL
    output_dir: str = ""

    # LoRA config
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    warmup_ratio: float = 0.10

    # DPO
    dpo_epochs: int = 2
    dpo_lr: float = 5e-6
    dpo_beta: float = 0.1

    # Ollama
    ollama_model_name: str = "DHS/DocWain"
    ollama_tag: str = "v2-wip"


# ---------------------------------------------------------------------------
# Modelfile builder
# ---------------------------------------------------------------------------


def _read_v1_modelfile_blocks() -> Dict[str, str]:
    """Read SYSTEM, TEMPLATE, and PARAMETER blocks from the V1 Modelfile.

    Returns dict with keys: system, template, parameters (each a string).
    """
    result = {"system": "", "template": "", "parameters": ""}

    modelfile = _V1_MODELFILE
    if not modelfile.exists():
        logger.warning("V1 Modelfile not found at %s", modelfile)
        return result

    content = modelfile.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Parse TEMPLATE block
    in_template = False
    template_lines: List[str] = []
    in_system = False
    system_lines: List[str] = []
    param_lines: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith('TEMPLATE """'):
            in_template = True
            template_lines.append(line)
            i += 1
            continue
        if in_template:
            template_lines.append(line)
            if '"""' in line and not line.startswith('TEMPLATE'):
                in_template = False
            i += 1
            continue

        if line.startswith('SYSTEM """'):
            in_system = True
            system_lines.append(line)
            i += 1
            continue
        if in_system:
            system_lines.append(line)
            if line.strip() == '"""' or (line.endswith('"""') and len(system_lines) > 1):
                in_system = False
            i += 1
            continue

        if line.startswith("PARAMETER "):
            param_lines.append(line)

        i += 1

    result["template"] = "\n".join(template_lines)
    result["system"] = "\n".join(system_lines)
    result["parameters"] = "\n".join(param_lines)
    return result


def _build_modelfile(gguf_path: str) -> str:
    """Build a complete Modelfile for the new GGUF.

    Re-uses the SYSTEM prompt, TEMPLATE, and PARAMETERs from V1.
    """
    blocks = _read_v1_modelfile_blocks()

    # Ollama requires absolute path for the GGUF file
    abs_gguf = str(Path(gguf_path).resolve())
    parts = [f"FROM {abs_gguf}"]

    if blocks["template"]:
        parts.append(blocks["template"])
    else:
        # Fallback Qwen3 template
        parts.append("""TEMPLATE \"\"\"{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
{{- range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}
<|im_start|>assistant
\"\"\"""")

    if blocks["system"]:
        parts.append(blocks["system"])

    if blocks["parameters"]:
        parts.append(blocks["parameters"])
    else:
        parts.append("PARAMETER temperature 0.3")
        parts.append("PARAMETER top_p 0.85")
        parts.append("PARAMETER top_k 40")
        parts.append("PARAMETER repeat_penalty 1.1")
        parts.append("PARAMETER num_ctx 4096")
        parts.append("PARAMETER stop <|im_end|>")

    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# SFT training
# ---------------------------------------------------------------------------


def _run_sft(config: TrackTrainingConfig) -> Dict:
    """Run SFT training using Unsloth FastLanguageModel.

    Returns dict with training metrics.
    """
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # Determine base: previous track checkpoint or base model
    model_name = config.base_checkpoint or config.base_model
    load_4bit = config.base_checkpoint is None  # only 4bit for HF base model
    logger.info(
        "Loading model for SFT: %s (4bit=%s, track=%s)",
        model_name, load_4bit, config.track_name,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=load_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load and prepare dataset
    dataset = load_dataset("json", data_files=config.data_path, split="train")
    logger.info("Loaded %d SFT examples from %s", len(dataset), config.data_path)

    # The data already has a "text" field formatted with Qwen3 chat template
    # Verify and handle if messages format is used instead
    sample = dataset[0]
    if "text" not in sample and "messages" in sample:
        logger.info("Converting messages format to text format")

        def _format_messages(example):
            messages = example["messages"]
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            return {"text": text}

        dataset = dataset.map(_format_messages)

    # Training output
    sft_output = Path(config.output_dir) / "sft_checkpoints"
    sft_output.mkdir(parents=True, exist_ok=True)

    # Detect precision
    try:
        import torch
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False
    use_fp16 = not use_bf16

    training_args = SFTConfig(
        output_dir=str(sft_output),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        warmup_ratio=config.warmup_ratio,
        logging_steps=1,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        seed=42,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info(
        "Starting SFT training: %d examples, %d epochs, lr=%.2e",
        len(dataset), config.epochs, config.learning_rate,
    )
    train_result = trainer.train()
    final_loss = train_result.metrics.get("train_loss", 0.0)
    logger.info("SFT complete (loss=%.4f)", final_loss)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_loss": final_loss,
        "epochs": config.epochs,
        "examples": len(dataset),
    }


# ---------------------------------------------------------------------------
# DPO training
# ---------------------------------------------------------------------------


def _run_dpo(
    model,
    tokenizer,
    config: TrackTrainingConfig,
) -> Dict:
    """Run DPO training on top of the SFT-trained model.

    Returns dict with DPO metrics.
    """
    from datasets import load_dataset
    try:
        from trl import DPOTrainer, DPOConfig
    except (ImportError, RuntimeError) as exc:
        logger.warning("DPO trainer not available (%s), skipping DPO", exc)
        return {"skipped": True, "reason": str(exc)}

    if not config.dpo_path or not Path(config.dpo_path).exists():
        logger.info("No DPO data at %s, skipping DPO", config.dpo_path)
        return {"skipped": True}

    dataset = load_dataset("json", data_files=config.dpo_path, split="train")
    logger.info("Loaded %d DPO pairs from %s", len(dataset), config.dpo_path)

    if len(dataset) == 0:
        logger.warning("DPO dataset is empty, skipping")
        return {"skipped": True}

    dpo_output = Path(config.output_dir) / "dpo_checkpoints"
    dpo_output.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False
    use_fp16 = not use_bf16

    dpo_config = DPOConfig(
        output_dir=str(dpo_output),
        per_device_train_batch_size=max(1, config.batch_size // 2),
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.dpo_lr,
        num_train_epochs=config.dpo_epochs,
        warmup_ratio=config.warmup_ratio,
        beta=config.dpo_beta,
        logging_steps=1,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        seed=42,
        max_length=config.max_seq_length,
        max_prompt_length=config.max_seq_length // 2,
    )

    from unsloth import FastLanguageModel
    FastLanguageModel.for_training(model)

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info(
        "Starting DPO training: %d pairs, %d epochs, beta=%.2f",
        len(dataset), config.dpo_epochs, config.dpo_beta,
    )
    train_result = trainer.train()
    dpo_loss = train_result.metrics.get("train_loss", 0.0)
    logger.info("DPO complete (loss=%.4f)", dpo_loss)

    return {
        "skipped": False,
        "train_loss": dpo_loss,
        "epochs": config.dpo_epochs,
        "examples": len(dataset),
    }


# ---------------------------------------------------------------------------
# Merge, GGUF export, Ollama update
# ---------------------------------------------------------------------------


def _merge_and_export(model, tokenizer, config: TrackTrainingConfig) -> str:
    """Merge LoRA into base, save FP16, export GGUF, update Ollama.

    Returns path to the merged checkpoint directory.
    """
    out = Path(config.output_dir)
    merged_dir = out / "merged_16bit"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Save merged FP16
    logger.info("Saving merged FP16 to %s", merged_dir)
    try:
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit",
        )
    except AttributeError:
        logger.warning("save_pretrained_merged not available, using save_pretrained")
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    # Export GGUF Q4_K_M
    gguf_dir = out / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting GGUF Q4_K_M to %s", gguf_dir)
    try:
        model.save_pretrained_gguf(
            str(gguf_dir), tokenizer, quantization_method="q4_k_m",
        )
        logger.info("GGUF export successful")
    except Exception as exc:
        logger.warning("GGUF export failed: %s", exc)

    # Find the GGUF file (Unsloth may save in a subdirectory like gguf_gguf/)
    gguf_files = sorted(glob.glob(str(gguf_dir / "*.gguf")))
    if not gguf_files:
        gguf_files = sorted(glob.glob(str(gguf_dir / "**" / "*.gguf"), recursive=True))
    if not gguf_files:
        # Also check parent directory variations
        gguf_files = sorted(glob.glob(str(gguf_dir.parent / "**" / "*.gguf"), recursive=True))
    if not gguf_files:
        logger.warning("No GGUF file found after export in %s or subdirectories", gguf_dir)
        return str(merged_dir)

    gguf_path = gguf_files[-1]
    logger.info("Using GGUF: %s", gguf_path)

    # Build Modelfile and update Ollama
    modelfile_content = _build_modelfile(gguf_path)
    modelfile_path = out / "Modelfile"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    logger.info("Modelfile written to %s", modelfile_path)

    full_model_name = f"{config.ollama_model_name}:{config.ollama_tag}"
    _update_ollama(full_model_name, str(modelfile_path))

    return str(merged_dir)


def _update_ollama(model_name: str, modelfile_path: str) -> bool:
    """Create/update an Ollama model from a Modelfile.

    Returns True on success.
    """
    logger.info("Updating Ollama model: %s", model_name)
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode == 0:
            logger.info("Ollama model %s updated successfully", model_name)
            return True
        logger.error("Ollama update failed (code %d): %s", result.returncode, result.stderr)
        return False
    except FileNotFoundError:
        logger.warning("ollama CLI not found, skipping model update")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Ollama model creation timed out (>900s)")
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_track(config: TrackTrainingConfig) -> str:
    """Train a single track with SFT (+ optional DPO).

    Steps:
    1. Load base model or previous checkpoint via Unsloth FastLanguageModel
    2. Apply LoRA with config parameters
    3. Load JSONL dataset, format for Qwen3 chat template
    4. Train SFT with TRL SFTTrainer
    5. If dpo_path provided: train DPO on preference pairs
    6. Save merged checkpoint (merged_16bit)
    7. Export to GGUF Q4_K_M
    8. Update Ollama model with new GGUF

    Parameters
    ----------
    config:
        Training configuration for this track.

    Returns
    -------
    Path to the merged checkpoint directory, suitable as base_checkpoint
    for the next track.
    """
    logger.info("=== Training track: %s ===", config.track_name)
    logger.info("  base_model: %s", config.base_model)
    logger.info("  base_checkpoint: %s", config.base_checkpoint)
    logger.info("  data_path: %s", config.data_path)
    logger.info("  dpo_path: %s", config.dpo_path)
    logger.info("  output_dir: %s", config.output_dir)
    logger.info("  lora_r=%d alpha=%d lr=%.2e epochs=%d",
                config.lora_r, config.lora_alpha, config.learning_rate, config.epochs)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config_path = out / "track_config.json"
    config_dict = {
        "track_name": config.track_name,
        "base_model": config.base_model,
        "base_checkpoint": config.base_checkpoint,
        "data_path": config.data_path,
        "dpo_path": config.dpo_path,
        "output_dir": config.output_dir,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": config.target_modules,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_seq_length": config.max_seq_length,
        "warmup_ratio": config.warmup_ratio,
        "dpo_epochs": config.dpo_epochs,
        "dpo_lr": config.dpo_lr,
        "dpo_beta": config.dpo_beta,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Step 1-4: SFT training
    sft_result = _run_sft(config)
    model = sft_result["model"]
    tokenizer = sft_result["tokenizer"]

    # Step 5: Optional DPO
    dpo_result = _run_dpo(model, tokenizer, config)

    # Steps 6-8: Merge, GGUF, Ollama
    merged_dir = _merge_and_export(model, tokenizer, config)

    # Save training summary
    summary = {
        "track_name": config.track_name,
        "sft": {
            "train_loss": sft_result["train_loss"],
            "epochs": sft_result["epochs"],
            "examples": sft_result["examples"],
        },
        "dpo": {
            "skipped": dpo_result.get("skipped", True),
            "train_loss": dpo_result.get("train_loss", 0.0),
            "epochs": dpo_result.get("epochs", 0),
            "examples": dpo_result.get("examples", 0),
        },
        "merged_dir": merged_dir,
    }
    summary_path = out / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Track %s training complete. Merged checkpoint: %s",
                config.track_name, merged_dir)
    return merged_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for training a single track."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a single DocWain V2 track (SFT + optional DPO)",
    )
    parser.add_argument("--track", required=True, help="Track name (e.g. excel_csv)")
    parser.add_argument("--data-path", required=True, help="Path to SFT JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--base-model", default="unsloth/Qwen3-14B-bnb-4bit")
    parser.add_argument("--base-checkpoint", default=None, help="Previous track checkpoint")
    parser.add_argument("--dpo-path", default=None, help="Path to DPO JSONL")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dpo-epochs", type=int, default=2)
    parser.add_argument("--dpo-lr", type=float, default=5e-6)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config = TrackTrainingConfig(
        track_name=args.track,
        base_model=args.base_model,
        base_checkpoint=args.base_checkpoint,
        data_path=args.data_path,
        dpo_path=args.dpo_path,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dpo_epochs=args.dpo_epochs,
        dpo_lr=args.dpo_lr,
        dpo_beta=args.dpo_beta,
    )

    checkpoint = train_track(config)
    print(f"\nTraining complete. Merged checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
