"""Integration test for curriculum pipeline — validates wiring without GPU."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.finetune.v2.curriculum_trainer import (
    PipelineState, phase_generate, save_state, load_state,
)
from src.finetune.v2.curriculum_generator import (
    build_initial_briefs, validate_example, merge_datasets,
)
from src.finetune.v2.curriculum_evaluator import (
    aggregate_scores, check_gates, build_failure_analysis,
)


class TestPipelineWiring:
    def test_initial_briefs_generate_valid_prompts(self):
        """All initial briefs produce non-empty prompts with required elements."""
        briefs = build_initial_briefs()
        assert len(briefs) == 6
        for brief in briefs:
            prompt = brief.to_prompt()
            assert len(prompt) > 500
            assert "<|im_start|>" in prompt
            assert brief.area in prompt
            assert str(brief.count) in prompt

    def test_full_gate_check_flow(self):
        """Scores -> aggregate -> gate check works end-to-end."""
        scores = []
        for track in ["excel_csv", "layout", "ocr_vision", "reasoning", "kg", "visualization"]:
            for _ in range(10):
                scores.append({
                    "track": track,
                    "scores": {
                        "factual_correctness": 4.2,
                        "reasoning_quality": 3.9,
                        "completeness": 4.1,
                        "grounding": 3.8,
                    },
                })
        agg = aggregate_scores(scores)
        gates = check_gates(agg)
        assert gates.basics_passed is True
        assert gates.overall_avg >= 3.9

    def test_failure_analysis_produces_augmentation_briefs(self):
        """Low scores -> failure analysis -> augmentation briefs."""
        scores = [
            {"track": "excel_csv", "category": "aggregation", "difficulty": "hard",
             "scores": {"factual_correctness": 2.5, "reasoning_quality": 2.0,
                        "completeness": 3.0, "grounding": 2.5},
             "prompt": "test", "response": "bad"},
        ] * 10

        analysis = build_failure_analysis(scores, threshold=3.5)
        assert len(analysis["weak_areas"]) > 0

        from src.finetune.v2.curriculum_generator import build_augmentation_briefs
        briefs = build_augmentation_briefs(analysis, iteration=2)
        assert len(briefs) > 0
        for brief in briefs:
            assert brief.iteration == 2
            assert len(brief.focus_instructions) > 0

    def test_state_survives_full_cycle(self):
        """State persists correctly through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            state = PipelineState(
                iteration=3, phase="eval", basics_passed=True,
                dataset_sizes={"iter_1_base": 5100, "iter_2_augment": 800},
                eval_history=[
                    {"iteration": 1, "overall_avg": 3.2},
                    {"iteration": 2, "overall_avg": 3.6},
                ],
                failure_analyses=[
                    {"weak_areas": [{"area": "excel_csv", "dimension": "aggregation_accuracy",
                                     "avg_score": 2.8}]},
                ],
            )
            save_state(state, path)
            loaded = load_state(path)
            assert loaded.iteration == 3
            assert loaded.basics_passed is True
            assert len(loaded.eval_history) == 2
            assert len(loaded.failure_analyses) == 1

    def test_validate_example_matches_brief_format(self):
        """Examples generated per the brief format should pass validation."""
        from src.finetune.v2.data_generator.base import DOCWAIN_SYSTEM_PROMPT
        example = {
            "text": (
                f"<|im_start|>system\n{DOCWAIN_SYSTEM_PROMPT}<|im_end|>\n"
                "<|im_start|>user\n[SPREADSHEET: report.xlsx / Sheet1]\n"
                "| Name | Sales |\n| Alice | 50000 |\n| Bob | 30000 |\n\n"
                "What are the total sales?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
                "Step 1: I need to sum the Sales column.\n"
                "Step 2: Alice has 50000, Bob has 30000.\n"
                "Step 3: Total = 50000 + 30000 = 80000.\n"
                "</think>\n\n"
                "The total sales are **$80,000** across 2 employees.\n"
                "Confidence: High — direct column summation.<|im_end|>"
            ),
            "area": "excel_csv",
            "difficulty": "easy",
            "category": "single_sheet_tabular_qa",
        }
        assert validate_example(example) is True
