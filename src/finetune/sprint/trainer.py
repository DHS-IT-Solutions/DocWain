"""Sprint Training Engine — SFT + DPO with curriculum sorting and LoRA merge.

Uses Unsloth + TRL when available; falls back to dry-run mode (saves metadata
JSON and logs a warning) so the module remains importable in environments
without GPU / Unsloth installed.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Difficulty ordering for curriculum learning
_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}

# Default difficulty when key is absent (treated as medium)
_DEFAULT_DIFFICULTY = "medium"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file and return a list of dicts."""
    examples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def curriculum_sort(examples: List[Dict]) -> List[Dict]:
    """Sort examples easy → medium → hard for curriculum learning.

    Examples without a ``difficulty`` key are treated as ``medium``.
    """
    return sorted(
        examples,
        key=lambda ex: _DIFFICULTY_ORDER.get(
            ex.get("difficulty", _DEFAULT_DIFFICULTY), 1
        ),
    )


def split_sft_dpo(examples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split a mixed dataset into SFT examples and DPO preference pairs.

    - SFT example: has a ``"text"`` key (plain completion training).
    - DPO example: has ``"prompt"`` **and** ``"chosen"`` keys (preference pair).

    Examples that match neither pattern are dropped with a warning.
    """
    sft: List[Dict] = []
    dpo: List[Dict] = []
    for ex in examples:
        if "prompt" in ex and "chosen" in ex:
            dpo.append(ex)
        elif "text" in ex:
            sft.append(ex)
        else:
            logger.warning("Skipping unrecognised example (no 'text' or 'prompt'+'chosen'): %s", ex)
    return sft, dpo


# ---------------------------------------------------------------------------
# SprintTrainer
# ---------------------------------------------------------------------------

class SprintTrainer:
    """Wraps Unsloth + TRL for LoRA SFT and DPO training.

    When Unsloth is not importable the trainer operates in *dry-run* mode:
    it skips actual GPU training, writes a metadata JSON to the output
    directory, logs a warning, and returns the output path so callers can
    proceed normally.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : SprintConfig
            Training hyper-parameters and path settings.
        """
        self.config = config
        self._unsloth_available = self._check_unsloth()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_unsloth() -> bool:
        try:
            import unsloth  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "Unsloth is not installed. SprintTrainer will operate in "
                "dry-run mode (no actual GPU training)."
            )
            return False

    def _save_dry_run_metadata(self, output_dir: Path, mode: str, extra: dict) -> Path:
        """Write a metadata JSON so the caller can detect dry-run completion."""
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "mode": mode,
            "dry_run": True,
            "config": {
                "base_model": self.config.base_model,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
            },
            **extra,
        }
        meta_path = output_dir / "dry_run_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.warning("Dry-run: metadata saved to %s", meta_path)
        return output_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_sft(
        self,
        data_path: Path,
        output_dir: Path,
        epochs: Optional[int] = None,
    ) -> Path:
        """Run supervised fine-tuning (SFT) with LoRA.

        Parameters
        ----------
        data_path : Path
            Path to a JSONL file containing SFT examples (``"text"`` field).
        output_dir : Path
            Directory where the merged 16-bit model will be saved.
        epochs : int, optional
            Overrides ``config.sft_epochs`` when provided.

        Returns
        -------
        Path
            Path to the merged model directory.
        """
        output_dir = Path(output_dir)
        epochs = epochs if epochs is not None else self.config.sft_epochs

        if not self._unsloth_available:
            return self._save_dry_run_metadata(
                output_dir,
                mode="sft",
                extra={"data_path": str(data_path), "epochs": epochs},
            )

        # --- real training path ---
        from unsloth import FastLanguageModel
        from trl import SFTTrainer as TRLSFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        raw = load_jsonl(data_path)
        raw = curriculum_sort(raw)
        dataset = Dataset.from_list(raw)

        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=self.config.sft_batch_size,
            gradient_accumulation_steps=self.config.sft_grad_accum,
            learning_rate=self.config.sft_lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            bf16=True,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
        )

        trainer = TRLSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=training_args,
        )
        trainer.train()

        merged_dir = output_dir / "merged_16bit"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        logger.info("SFT complete. Merged model saved to %s", merged_dir)
        return merged_dir

    def train_dpo(
        self,
        data_path: Path,
        base_model_path: Path,
        output_dir: Path,
    ) -> Path:
        """Run DPO (Direct Preference Optimisation) fine-tuning with LoRA.

        Parameters
        ----------
        data_path : Path
            Path to a JSONL file containing DPO preference pairs
            (``"prompt"``, ``"chosen"``, ``"rejected"`` fields).
        base_model_path : Path
            Path to the SFT-merged model to use as the starting checkpoint.
        output_dir : Path
            Directory where the merged 16-bit DPO model will be saved.

        Returns
        -------
        Path
            Path to the merged model directory.
        """
        output_dir = Path(output_dir)

        if not self._unsloth_available:
            return self._save_dry_run_metadata(
                output_dir,
                mode="dpo",
                extra={
                    "data_path": str(data_path),
                    "base_model_path": str(base_model_path),
                },
            )

        # --- real training path ---
        from unsloth import FastLanguageModel
        from trl import DPOTrainer as TRLDPOTrainer, DPOConfig
        from datasets import Dataset

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(base_model_path),
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        raw = load_jsonl(data_path)
        dataset = Dataset.from_list(raw)

        dpo_config = DPOConfig(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=self.config.dpo_epochs,
            per_device_train_batch_size=self.config.dpo_batch_size,
            gradient_accumulation_steps=self.config.dpo_grad_accum,
            learning_rate=self.config.dpo_lr,
            beta=self.config.dpo_beta,
            bf16=True,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
        )

        trainer = TRLDPOTrainer(
            model=model,
            ref_model=None,  # use implicit ref (PEFT adapter disabled)
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

        merged_dir = output_dir / "merged_16bit"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        logger.info("DPO complete. Merged model saved to %s", merged_dir)
        return merged_dir
