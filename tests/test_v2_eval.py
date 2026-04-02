"""Tests for the DocWain V2+ evaluation infrastructure."""

import pytest


class TestRubrics:
    def test_get_rubric(self):
        from src.finetune.v2.eval.rubrics import get_rubric

        text = get_rubric("synthesis_coherence")
        assert "5:" in text
        assert "1:" in text

    def test_all_rubrics_exist(self):
        from src.finetune.v2.eval.rubrics import RUBRIC_NAMES, get_rubric

        assert len(RUBRIC_NAMES) >= 5
        for name in RUBRIC_NAMES:
            rubric = get_rubric(name)
            assert len(rubric) > 50, f"Rubric {name} is too short"

    def test_get_rubric_unknown_raises(self):
        from src.finetune.v2.eval.rubrics import get_rubric

        with pytest.raises(ValueError, match="Unknown rubric"):
            get_rubric("nonexistent_rubric")

    def test_score_with_rubric(self):
        from src.finetune.v2.eval.rubrics import score_with_rubric

        result = score_with_rubric(
            rubric_name="synthesis_coherence",
            model_output="<think>Reasoning here</think> The answer with [source 1].",
            reference="A reference answer of moderate length for comparison.",
            context="Some context.",
        )
        assert 1 <= result["score"] <= 5
        assert isinstance(result["reasoning"], str)
        assert result["rubric_name"] == "synthesis_coherence"

    def test_score_penalises_short_output(self):
        from src.finetune.v2.eval.rubrics import score_with_rubric

        result = score_with_rubric(
            rubric_name="intent_alignment",
            model_output="Ok",
            reference="A much longer reference that the output should approximate.",
        )
        # Short output should get penalised, score should be lower
        assert result["score"] <= 3


class TestGateChecker:
    def test_phase2_gates(self):
        from src.finetune.v2.eval.gate_checker import check_gates

        metrics = {"docvqa_accuracy": 0.80, "table_f1": 0.85, "layout_map": 0.75}
        result = check_gates("phase2", metrics)
        assert result["passed"] is True
        assert result["phase"] == "phase2"
        assert len(result["failures"]) == 0

    def test_phase2_gates_fail(self):
        from src.finetune.v2.eval.gate_checker import check_gates

        metrics = {"docvqa_accuracy": 0.50, "table_f1": 0.60, "layout_map": 0.40}
        result = check_gates("phase2", metrics)
        assert result["passed"] is False
        assert len(result["failures"]) > 0

    def test_phase37_gates(self):
        from src.finetune.v2.eval.gate_checker import check_gates

        metrics = {
            "synthesis_coherence": 0.85,
            "intent_alignment": 0.90,
            "depth_calibration": 0.80,
            "domain_accuracy": 0.85,
        }
        result = check_gates("phase3_7", metrics)
        assert result["passed"] is True

    def test_all_phases_have_gates(self):
        from src.finetune.v2.eval.gate_checker import PHASE_GATES

        required = [
            "phase1", "phase2", "phase2_5", "phase3", "phase3_5",
            "phase3_7", "phase4", "round1", "round2", "round3",
        ]
        for phase in required:
            assert phase in PHASE_GATES, f"Missing gate for {phase}"

    def test_upper_bound_metrics(self):
        from src.finetune.v2.eval.gate_checker import check_gates

        # hallucination_rate is upper-bound: 0.03 <= 0.05 should pass
        metrics = {"hallucination_rate": 0.03, "extraction_f1_improvement": 0.10}
        result = check_gates("phase2_5", metrics)
        assert result["passed"] is True

        # hallucination_rate 0.10 > 0.05 should fail
        metrics_fail = {"hallucination_rate": 0.10, "extraction_f1_improvement": 0.10}
        result_fail = check_gates("phase2_5", metrics_fail)
        assert result_fail["passed"] is False

    def test_unknown_phase_raises(self):
        from src.finetune.v2.eval.gate_checker import check_gates

        with pytest.raises(ValueError, match="Unknown phase"):
            check_gates("nonexistent_phase", {})


class TestRunner:
    def test_run_eval_returns_scores(self):
        from src.finetune.v2.eval.runner import run_eval_on_examples

        examples = [
            {
                "prompt": "What is in this document?",
                "reference": "The document contains financial data.",
                "benchmark": "synthesis",
            }
        ]

        def mock_model(prompt: str) -> str:
            return "<think>Analysing</think> The document contains financial data with [source 1]."

        results = run_eval_on_examples(examples, mock_model)
        assert len(results) == 1
        assert "score" in results[0]
        assert 1 <= results[0]["score"] <= 5
        assert "reasoning" in results[0]
        assert "model_output" in results[0]

    def test_compute_phase_metrics(self):
        from src.finetune.v2.eval.runner import compute_phase_metrics

        results = [
            {"benchmark": "synthesis", "score": 4},
            {"benchmark": "synthesis", "score": 5},
            {"benchmark": "tool_calling", "score": 3},
        ]
        metrics = compute_phase_metrics(results)
        assert "synthesis" in metrics
        assert "tool_calling" in metrics
        # synthesis: mean=4.5, normalised=(4.5-1)/4=0.875
        assert abs(metrics["synthesis"] - 0.875) < 0.01
