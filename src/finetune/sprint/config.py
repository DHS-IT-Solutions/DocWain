import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class SprintConfig:
    # Base model
    base_model: str = "unsloth/Qwen3-14B-bnb-4bit"
    model_name: str = "DocWain-14B-v2"
    max_seq_length: int = 4096

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # SFT training
    sft_epochs: int = 3
    sft_lr: float = 2e-5
    sft_batch_size: int = 4
    sft_grad_accum: int = 8

    # DPO training
    dpo_epochs: int = 1
    dpo_lr: float = 5e-6
    dpo_beta: float = 0.1
    dpo_batch_size: int = 2
    dpo_grad_accum: int = 8

    # Phase 1 data targets
    phase1_sft_target: int = 13000
    phase1_dpo_target: int = 5000

    # Phase 2 data targets
    phase2_sft_target: int = 27000
    phase2_dpo_target: int = 4000

    # Distillation batch size (for eval gating)
    distill_batch_size: int = 1000

    # Paths
    artifacts_dir: str = "finetune_artifacts/sprint"
    eval_bank_path: str = "finetune_artifacts/sprint/eval_bank.jsonl"

    # Final targets
    hallucination_target: float = 0.05
    completeness_target: float = 0.90
    intent_target: float = 0.90
    judge_score_target: float = 4.0

    @property
    def final_targets(self) -> dict:
        return {
            "hallucination_rate": self.hallucination_target,
            "extraction_completeness": self.completeness_target,
            "intent_understanding": self.intent_target,
            "excel_csv_score": self.judge_score_target,
            "ocr_accuracy": 0.95,
            "reasoning_depth": self.judge_score_target,
            "cross_doc_score": self.judge_score_target,
            "content_generation": self.judge_score_target,
            "domain_awareness": self.judge_score_target,
        }

    @property
    def phase1_gate(self) -> dict:
        return {
            "hallucination_rate": 0.08,
            "completeness": 0.82,
            "intent_accuracy": 0.85,
        }


@dataclass
class SprintState:
    base_dir: Path = field(default_factory=lambda: Path("finetune_artifacts/sprint"))
    phase: str = "init"
    phase1_passed: bool = False
    final_passed: bool = False
    scores: dict = field(default_factory=dict)
    eval_history: list = field(default_factory=list)
    best_checkpoint: Optional[str] = None
    best_score: float = 0.0
    sft_count: int = 0
    dpo_count: int = 0

    def save(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.base_dir / "state.json"
        data = asdict(self)
        data["base_dir"] = str(data["base_dir"])
        state_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, base_dir: Path) -> "SprintState":
        state_path = base_dir / "state.json"
        if not state_path.exists():
            return cls(base_dir=base_dir)
        data = json.loads(state_path.read_text())
        data["base_dir"] = Path(data["base_dir"])
        return cls(**data)
