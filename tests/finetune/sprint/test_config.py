import json
import tempfile
from pathlib import Path


def test_sprint_config_defaults():
    from src.finetune.sprint.config import SprintConfig

    cfg = SprintConfig()
    assert cfg.phase1_sft_target == 13000
    assert cfg.phase1_dpo_target == 5000
    assert cfg.phase2_sft_target == 27000
    assert cfg.phase2_dpo_target == 4000
    assert cfg.hallucination_target <= 0.05
    assert cfg.completeness_target >= 0.90
    assert cfg.base_model == "unsloth/Qwen3-14B-bnb-4bit"
    assert cfg.lora_r == 64


def test_sprint_config_targets():
    from src.finetune.sprint.config import SprintConfig

    cfg = SprintConfig()
    targets = cfg.final_targets
    assert targets["hallucination_rate"] <= 0.05
    assert targets["extraction_completeness"] >= 0.90
    assert targets["intent_understanding"] >= 0.90
    assert targets["excel_csv_score"] >= 4.0
    assert targets["ocr_accuracy"] >= 0.95
    assert targets["reasoning_depth"] >= 4.0
    assert targets["cross_doc_score"] >= 4.0
    assert targets["content_generation"] >= 4.0
    assert targets["domain_awareness"] >= 4.0


def test_sprint_config_phase1_gate():
    from src.finetune.sprint.config import SprintConfig

    cfg = SprintConfig()
    gate = cfg.phase1_gate
    assert gate["hallucination_rate"] <= 0.08
    assert gate["completeness"] >= 0.82
    assert gate["intent_accuracy"] >= 0.85


def test_sprint_state_save_load():
    from src.finetune.sprint.config import SprintState

    with tempfile.TemporaryDirectory() as tmpdir:
        state = SprintState(base_dir=Path(tmpdir))
        state.phase = "phase1_sft"
        state.scores = {"accuracy": 4.2, "completeness": 3.8}
        state.save()

        loaded = SprintState.load(Path(tmpdir))
        assert loaded.phase == "phase1_sft"
        assert loaded.scores["accuracy"] == 4.2


def test_sprint_state_defaults():
    from src.finetune.sprint.config import SprintState

    with tempfile.TemporaryDirectory() as tmpdir:
        state = SprintState(base_dir=Path(tmpdir))
        assert state.phase == "init"
        assert state.phase1_passed is False
        assert state.final_passed is False
        assert state.best_checkpoint is None
