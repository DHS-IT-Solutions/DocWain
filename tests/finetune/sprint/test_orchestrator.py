import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_orchestrator_initializes():
    from src.finetune.sprint.orchestrator import SprintOrchestrator
    from src.finetune.sprint.config import SprintConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SprintConfig(artifacts_dir=tmpdir)
        orch = SprintOrchestrator(cfg)
        assert orch.state.phase == "init"


def test_orchestrator_phase_sequence():
    from src.finetune.sprint.orchestrator import PHASE_SEQUENCE

    assert PHASE_SEQUENCE == [
        "generate_eval_bank",
        "phase1_generate",
        "phase1_sft",
        "phase1_dpo",
        "phase1_gate",
        "phase2_generate",
        "phase2_sft",
        "phase2_dpo",
        "final_gate",
        "convert",
        "done",
    ]


def test_orchestrator_advance_phase():
    from src.finetune.sprint.orchestrator import SprintOrchestrator
    from src.finetune.sprint.config import SprintConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SprintConfig(artifacts_dir=tmpdir)
        orch = SprintOrchestrator(cfg)
        assert orch.state.phase == "init"
        orch._advance_phase()
        assert orch.state.phase == "generate_eval_bank"
        orch._advance_phase()
        assert orch.state.phase == "phase1_generate"
