"""Tests for curriculum training modifications to train_track."""

import json
import tempfile
from pathlib import Path

from src.finetune.v2.train_track import TrackTrainingConfig, CurriculumSampler


class TestTrackTrainingConfig:
    def test_new_defaults(self):
        cfg = TrackTrainingConfig()
        assert cfg.lora_dropout == 0.05
        assert cfg.curriculum_sampling is True
        assert cfg.checkpoint_save_pct == [25, 50, 75, 100]
        assert cfg.skip_ollama_export is False

    def test_skip_ollama_export(self):
        cfg = TrackTrainingConfig(skip_ollama_export=True)
        assert cfg.skip_ollama_export is True


class TestCurriculumSampler:
    def test_ordering_by_difficulty(self):
        examples = [
            {"text": "hard_1", "difficulty": "hard"},
            {"text": "easy_1", "difficulty": "easy"},
            {"text": "medium_1", "difficulty": "medium"},
            {"text": "easy_2", "difficulty": "easy"},
            {"text": "hard_2", "difficulty": "hard"},
            {"text": "medium_2", "difficulty": "medium"},
        ]
        sampler = CurriculumSampler(examples)
        indices = list(sampler)
        texts = [examples[i]["text"] for i in indices]
        easy = [t for t in texts if t.startswith("easy")]
        medium = [t for t in texts if t.startswith("medium")]
        hard = [t for t in texts if t.startswith("hard")]
        assert texts.index(easy[-1]) < texts.index(medium[0])
        assert texts.index(medium[-1]) < texts.index(hard[0])

    def test_length_matches_dataset(self):
        examples = [{"text": f"ex_{i}", "difficulty": "easy"} for i in range(10)]
        sampler = CurriculumSampler(examples)
        assert len(sampler) == 10

    def test_no_difficulty_field_falls_back_to_medium(self):
        examples = [{"text": "no_diff"}]
        sampler = CurriculumSampler(examples)
        assert len(list(sampler)) == 1
