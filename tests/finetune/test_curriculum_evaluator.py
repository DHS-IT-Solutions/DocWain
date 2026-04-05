"""Tests for curriculum evaluation and subagent judging."""

import json
from src.finetune.v2.curriculum_evaluator import (
    JudgingBrief, parse_judge_scores, aggregate_scores,
    check_gates, GateResult, build_failure_analysis,
)


class TestJudgingBrief:
    def test_brief_construction(self):
        examples = [{
            "prompt": "What is Carol's salary?",
            "response": "Carol's salary is $102,000.",
            "reference": {"expected_answer": "102000"},
            "track": "excel_csv", "category": "single_sheet_lookup", "difficulty": "easy",
        }]
        brief = JudgingBrief(examples=examples, batch_index=0)
        prompt = brief.to_prompt()
        assert "Carol" in prompt
        assert "102000" in prompt

    def test_brief_prompt_includes_scoring_dimensions(self):
        brief = JudgingBrief(examples=[{
            "prompt": "test", "response": "test", "reference": {},
            "track": "excel_csv", "category": "test", "difficulty": "easy",
        }], batch_index=0)
        prompt = brief.to_prompt()
        assert "1.0" in prompt and "5.0" in prompt


class TestParseJudgeScores:
    def test_parses_valid_json_scores(self):
        raw = json.dumps([{"example_index": 0, "scores": {
            "factual_correctness": 4.5, "reasoning_quality": 3.8,
            "completeness": 4.0, "grounding": 4.2,
        }}])
        scores = parse_judge_scores(raw)
        assert len(scores) == 1
        assert scores[0]["scores"]["factual_correctness"] == 4.5

    def test_handles_markdown_fenced_json(self):
        raw = "```json\n" + json.dumps([{"example_index": 0, "scores": {
            "factual_correctness": 4.0, "reasoning_quality": 3.5,
            "completeness": 4.0, "grounding": 3.5,
        }}]) + "\n```"
        scores = parse_judge_scores(raw)
        assert len(scores) == 1


class TestAggregateScores:
    def test_aggregates_across_tracks(self):
        all_scores = [
            {"track": "excel_csv", "scores": {
                "factual_correctness": 4.0, "reasoning_quality": 3.5,
                "completeness": 4.0, "grounding": 3.5,
            }},
            {"track": "excel_csv", "scores": {
                "factual_correctness": 5.0, "reasoning_quality": 4.5,
                "completeness": 5.0, "grounding": 4.5,
            }},
        ]
        agg = aggregate_scores(all_scores)
        assert "excel_csv" in agg
        assert agg["excel_csv"]["factual_correctness"] == 4.5
        assert "overall_avg" in agg


class TestGateChecks:
    def test_basics_gate_passes(self):
        agg = {"overall_avg": 3.6, "min_dimension": 3.1,
               "excel_csv": {"factual_correctness": 3.6, "reasoning_quality": 3.5,
                             "completeness": 3.8, "grounding": 3.5}}
        result = check_gates(agg)
        assert result.basics_passed is True
        assert result.production_passed is False

    def test_basics_gate_fails_on_low_dimension(self):
        agg = {"overall_avg": 3.6, "min_dimension": 2.8,
               "excel_csv": {"factual_correctness": 3.6, "reasoning_quality": 2.8,
                             "completeness": 3.8, "grounding": 3.5}}
        result = check_gates(agg)
        assert result.basics_passed is False

    def test_production_gate_passes(self):
        agg = {"overall_avg": 4.2, "min_dimension": 3.8,
               "excel_csv": {"factual_correctness": 4.2, "reasoning_quality": 3.8,
                             "completeness": 4.5, "grounding": 4.0}}
        result = check_gates(agg)
        assert result.basics_passed is True
        assert result.production_passed is True


class TestFailureAnalysis:
    def test_identifies_weak_dimensions(self):
        all_scores = [{
            "track": "excel_csv", "category": "aggregation", "difficulty": "hard",
            "scores": {"factual_correctness": 4.0, "reasoning_quality": 2.5,
                       "completeness": 3.8, "grounding": 3.5},
            "prompt": "Aggregate question", "response": "Bad response",
        }] * 10
        analysis = build_failure_analysis(all_scores, threshold=3.5)
        assert len(analysis["weak_areas"]) > 0
        weak_dims = [wa["dimension"] for wa in analysis["weak_areas"]]
        assert "reasoning_quality" in weak_dims
