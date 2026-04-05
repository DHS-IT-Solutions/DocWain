"""Tests for curriculum data generation."""

import json
import tempfile
from pathlib import Path

from src.finetune.v2.curriculum_generator import (
    GenerationBrief,
    build_initial_briefs,
    build_augmentation_briefs,
    parse_generated_examples,
    validate_example,
    merge_datasets,
    AREA_CONFIGS,
)


class TestAreaConfigs:
    def test_all_six_areas_defined(self):
        expected = {"excel_csv", "layout", "ocr_vision", "reasoning", "kg", "visualization"}
        assert set(AREA_CONFIGS.keys()) == expected

    def test_initial_counts_sum_to_5100(self):
        total = sum(cfg["initial_count"] for cfg in AREA_CONFIGS.values())
        assert total == 5100

    def test_each_area_has_required_keys(self):
        for area, cfg in AREA_CONFIGS.items():
            assert "initial_count" in cfg, f"{area} missing initial_count"
            assert "categories" in cfg, f"{area} missing categories"
            assert "difficulty_split" in cfg, f"{area} missing difficulty_split"


class TestGenerationBrief:
    def test_brief_construction(self):
        brief = GenerationBrief(
            area="excel_csv", count=50,
            difficulty_split={"easy": 0.2, "medium": 0.5, "hard": 0.3},
            categories=["tabular_qa", "multi_sheet"],
            focus_instructions="Focus on aggregation with >5 columns.",
            iteration=1,
        )
        assert brief.area == "excel_csv"
        assert brief.count == 50

    def test_brief_to_prompt(self):
        brief = GenerationBrief(
            area="excel_csv", count=10,
            difficulty_split={"easy": 0.2, "medium": 0.5, "hard": 0.3},
            categories=["tabular_qa"],
            focus_instructions="", iteration=1,
        )
        prompt = brief.to_prompt()
        assert "excel_csv" in prompt
        assert "10" in prompt
        assert "<|im_start|>" in prompt


class TestBuildBriefs:
    def test_initial_briefs_cover_all_areas(self):
        briefs = build_initial_briefs()
        areas = {b.area for b in briefs}
        assert areas == set(AREA_CONFIGS.keys())

    def test_augmentation_briefs_from_failure_analysis(self):
        failure_analysis = {
            "weak_areas": [
                {"area": "excel_csv", "dimension": "aggregation_accuracy", "avg_score": 2.5,
                 "failure_patterns": ["fails on >5 column spreadsheets"]},
                {"area": "reasoning", "dimension": "evidence_grounding", "avg_score": 2.8,
                 "failure_patterns": ["omits source citations"]},
            ],
            "total_augmentation_count": 1000,
        }
        briefs = build_augmentation_briefs(failure_analysis, iteration=3)
        assert len(briefs) == 2
        assert briefs[0].area == "excel_csv"
        assert "aggregation" in briefs[0].focus_instructions.lower()
        assert briefs[0].iteration == 3


class TestValidation:
    def test_valid_example_passes(self):
        example = {
            "text": (
                "<|im_start|>system\nYou are DocWain<|im_end|>\n"
                "<|im_start|>user\nWhat is the total?\n<|im_end|>\n"
                "<|im_start|>assistant\n<think>\nStep 1: sum values\n</think>\n\n"
                "The total is 500.<|im_end|>"
            ),
            "area": "excel_csv", "difficulty": "medium", "category": "tabular_qa",
        }
        assert validate_example(example) is True

    def test_missing_think_block_fails(self):
        example = {
            "text": (
                "<|im_start|>system\nYou are DocWain<|im_end|>\n"
                "<|im_start|>user\nWhat is the total?\n<|im_end|>\n"
                "<|im_start|>assistant\nThe total is 500.<|im_end|>"
            ),
            "area": "excel_csv", "difficulty": "medium", "category": "tabular_qa",
        }
        assert validate_example(example) is False

    def test_missing_area_fails(self):
        example = {"text": "<|im_start|>system\ntest<|im_end|>"}
        assert validate_example(example) is False


class TestMergeDatasets:
    def test_merge_combines_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "iter_1_base.jsonl"
            f2 = Path(tmpdir) / "iter_2_augment.jsonl"
            f1.write_text(
                json.dumps({"text": "example1", "area": "excel_csv", "difficulty": "easy"}) + "\n"
                + json.dumps({"text": "example2", "area": "layout", "difficulty": "medium"}) + "\n"
            )
            f2.write_text(
                json.dumps({"text": "example3", "area": "excel_csv", "difficulty": "hard"}) + "\n"
            )
            combined = Path(tmpdir) / "combined.jsonl"
            count = merge_datasets([f1, f2], combined)
            assert count == 3

    def test_merge_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "a.jsonl"
            f1.write_text(
                json.dumps({"text": "same", "area": "excel_csv", "difficulty": "easy"}) + "\n"
                + json.dumps({"text": "same", "area": "excel_csv", "difficulty": "easy"}) + "\n"
            )
            combined = Path(tmpdir) / "combined.jsonl"
            count = merge_datasets([f1], combined)
            assert count == 1
