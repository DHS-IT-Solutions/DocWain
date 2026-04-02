"""Tests for Phase 3.7 holistic reasoning SFT training loop."""

import pytest
from pathlib import Path


class TestPhase37Config:
    def test_config_defaults(self):
        from src.finetune.v2.train_phase3_7_holistic import HolisticConfig

        cfg = HolisticConfig()
        assert cfg.lora_r == 64
        assert cfg.lora_alpha == 128
        assert cfg.learning_rate == 8e-6
        assert cfg.epochs == 3
        assert cfg.max_seq_length == 8192
        assert cfg.bf16 is True

    def test_config_quality_gates(self):
        from src.finetune.v2.train_phase3_7_holistic import HolisticConfig

        cfg = HolisticConfig()
        assert cfg.gate_synthesis_coherence == 0.80
        assert cfg.gate_intent_alignment == 0.85
        assert cfg.gate_depth_calibration == 0.75
        assert cfg.gate_domain_accuracy == 0.80

    def test_build_training_args(self):
        from src.finetune.v2.train_phase3_7_holistic import (
            HolisticConfig,
            _build_training_args,
        )

        cfg = HolisticConfig()
        out = Path("/tmp/test_phase37")
        args = _build_training_args(cfg, out)

        assert args["output_dir"] == str(out)
        assert args["num_train_epochs"] == 3
        assert args["learning_rate"] == 8e-6
        assert args["max_seq_length"] == 8192
        assert args["bf16"] is True
        assert args["fp16"] is False
        assert args["weight_decay"] == 0.01
        assert args["max_grad_norm"] == 1.0

    def test_load_holistic_dataset_missing_file(self):
        from src.finetune.v2.train_phase3_7_holistic import _load_holistic_dataset

        result = _load_holistic_dataset(Path("/tmp/nonexistent_dir_phase37"))
        assert result is None

    def test_config_lora_target_modules(self):
        from src.finetune.v2.train_phase3_7_holistic import HolisticConfig

        cfg = HolisticConfig()
        expected = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert cfg.lora_target_modules == expected
