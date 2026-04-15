import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import json


def test_trainer_loads_config():
    from src.finetune.sprint.trainer import SprintTrainer
    from src.finetune.sprint.config import SprintConfig

    cfg = SprintConfig()
    trainer = SprintTrainer(cfg)
    assert trainer.config.lora_r == 64
    assert trainer.config.sft_epochs == 3


def test_load_jsonl_dataset():
    from src.finetune.sprint.trainer import load_jsonl

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(5):
            f.write(json.dumps({"text": f"example {i}", "category": "test"}) + "\n")
        path = f.name

    examples = load_jsonl(Path(path))
    assert len(examples) == 5
    assert examples[0]["text"] == "example 0"


def test_curriculum_sort():
    from src.finetune.sprint.trainer import curriculum_sort

    examples = [
        {"text": "hard", "difficulty": "hard"},
        {"text": "easy", "difficulty": "easy"},
        {"text": "medium", "difficulty": "medium"},
    ]
    sorted_ex = curriculum_sort(examples)
    assert sorted_ex[0]["difficulty"] == "easy"
    assert sorted_ex[1]["difficulty"] == "medium"
    assert sorted_ex[2]["difficulty"] == "hard"


def test_split_sft_dpo():
    from src.finetune.sprint.trainer import split_sft_dpo

    examples = [
        {"text": "sft1", "category": "extraction"},
        {"prompt": "q1", "chosen": "good", "rejected": "bad", "category": "anti_hallucination"},
        {"text": "sft2", "category": "intent"},
    ]
    sft, dpo = split_sft_dpo(examples)
    assert len(sft) == 2
    assert len(dpo) == 1
    assert "text" in sft[0]
    assert "prompt" in dpo[0]
