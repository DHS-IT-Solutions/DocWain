"""Tests for curriculum trainer orchestrator."""

import json
import tempfile
from pathlib import Path

from src.finetune.v2.curriculum_trainer import (
    PipelineState, load_state, save_state, PHASES,
)


class TestPipelineState:
    def test_initial_state(self):
        state = PipelineState()
        assert state.iteration == 0
        assert state.phase == "generate"
        assert state.basics_passed is False
        assert state.production_passed is False
        assert state.dataset_sizes == {}
        assert state.eval_history == []

    def test_state_serialization(self):
        state = PipelineState(
            iteration=3, phase="eval", basics_passed=True,
            dataset_sizes={"iter_1_base": 5000, "iter_2_augment": 800},
            eval_history=[{"iteration": 1, "overall_avg": 3.2}],
        )
        d = state.to_dict()
        assert d["iteration"] == 3
        assert d["phase"] == "eval"
        assert d["basics_passed"] is True

    def test_state_deserialization(self):
        d = {
            "iteration": 2, "phase": "train",
            "basics_passed": False, "production_passed": False,
            "dataset_sizes": {"iter_1_base": 5000},
            "eval_history": [], "failure_analyses": [],
        }
        state = PipelineState.from_dict(d)
        assert state.iteration == 2
        assert state.phase == "train"

    def test_failure_analyses_default(self):
        state = PipelineState()
        assert state.failure_analyses == []

    def test_best_checkpoint_default(self):
        state = PipelineState()
        assert state.best_checkpoint is None
        assert state.best_score == 0.0

    def test_to_dict_contains_all_fields(self):
        state = PipelineState(iteration=1, best_score=3.7)
        d = state.to_dict()
        assert "iteration" in d
        assert "phase" in d
        assert "basics_passed" in d
        assert "production_passed" in d
        assert "dataset_sizes" in d
        assert "eval_history" in d
        assert "failure_analyses" in d
        assert "best_checkpoint" in d
        assert "best_score" in d

    def test_from_dict_handles_missing_optional_fields(self):
        d = {
            "iteration": 1, "phase": "train",
            "basics_passed": False, "production_passed": False,
        }
        state = PipelineState.from_dict(d)
        assert state.iteration == 1
        assert state.dataset_sizes == {}
        assert state.eval_history == []
        assert state.failure_analyses == []


class TestStatePersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state = PipelineState(iteration=5, phase="analyze")
            save_state(state, state_path)
            loaded = load_state(state_path)
            assert loaded.iteration == 5
            assert loaded.phase == "analyze"

    def test_load_missing_file_returns_initial(self):
        state = load_state(Path("/nonexistent/state.json"))
        assert state.iteration == 0
        assert state.phase == "generate"

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "subdir" / "deep" / "state.json"
            state = PipelineState(iteration=2)
            save_state(state, state_path)
            assert state_path.exists()

    def test_save_produces_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state = PipelineState(
                iteration=3, phase="eval", basics_passed=True,
                best_score=3.8, best_checkpoint="/some/path",
            )
            save_state(state, state_path)
            raw = json.loads(state_path.read_text())
            assert raw["iteration"] == 3
            assert raw["best_score"] == 3.8

    def test_round_trip_preserves_complex_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state = PipelineState(
                iteration=4,
                dataset_sizes={"iter_1_base": 5000, "iter_2_augment": 800},
                eval_history=[{"iteration": 1, "overall_avg": 3.2},
                               {"iteration": 2, "overall_avg": 3.7}],
                failure_analyses=[{"weak_areas": [], "total_augmentation_count": 0}],
            )
            save_state(state, state_path)
            loaded = load_state(state_path)
            assert loaded.dataset_sizes == {"iter_1_base": 5000, "iter_2_augment": 800}
            assert len(loaded.eval_history) == 2
            assert loaded.eval_history[1]["overall_avg"] == 3.7


class TestPhases:
    def test_phase_order(self):
        assert PHASES == ["generate", "train", "eval", "analyze"]

    def test_next_phase(self):
        assert PHASES[(PHASES.index("generate") + 1) % len(PHASES)] == "train"
        assert PHASES[(PHASES.index("train") + 1) % len(PHASES)] == "eval"
        assert PHASES[(PHASES.index("eval") + 1) % len(PHASES)] == "analyze"
        assert PHASES[(PHASES.index("analyze") + 1) % len(PHASES)] == "generate"

    def test_phases_length(self):
        assert len(PHASES) == 4

    def test_phases_are_strings(self):
        for phase in PHASES:
            assert isinstance(phase, str)
