"""Tests for post-training refinement rounds (DPO, confidence SFT, distillation)."""

import pytest
from pathlib import Path


class TestRound1Config:
    def test_defaults(self):
        from src.finetune.v2.post_training.round1_conversational_dpo import Round1Config

        cfg = Round1Config()
        assert cfg.learning_rate == 1e-6
        assert cfg.beta == 0.05
        assert cfg.epochs == 2
        assert cfg.use_lora is False
        assert cfg.bf16 is True
        assert cfg.use_gradient_checkpointing is True
        assert cfg.gate_conversation_quality == 0.80

    def test_build_training_args(self):
        from src.finetune.v2.post_training.round1_conversational_dpo import (
            Round1Config,
            _build_training_args,
        )

        cfg = Round1Config()
        out = Path("/tmp/test_round1")
        args = _build_training_args(cfg, out)

        assert args["beta"] == 0.05
        assert args["learning_rate"] == 1e-6
        assert args["num_train_epochs"] == 2
        assert args["output_dir"] == str(out)
        assert args["bf16"] is True


class TestRound2Config:
    def test_defaults(self):
        from src.finetune.v2.post_training.round2_confidence_sft import Round2Config

        cfg = Round2Config()
        assert cfg.learning_rate == 1e-6
        assert cfg.epochs == 2
        assert cfg.gate_ece <= 0.10
        assert cfg.bf16 is True
        assert cfg.max_seq_length == 4096

    def test_build_training_args(self):
        from src.finetune.v2.post_training.round2_confidence_sft import (
            Round2Config,
            _build_training_args,
        )

        cfg = Round2Config()
        out = Path("/tmp/test_round2")
        args = _build_training_args(cfg, out)

        assert args["learning_rate"] == 1e-6
        assert args["num_train_epochs"] == 2
        assert args["output_dir"] == str(out)


class TestRound3Config:
    def test_defaults(self):
        from src.finetune.v2.post_training.round3_distillation import Round3Config

        cfg = Round3Config()
        assert cfg.learning_rate == 5e-7
        assert cfg.epochs == 1
        assert cfg.gate_max_quality_drop <= 0.03
        assert cfg.gate_min_speed_toks >= 25.0
        assert cfg.bf16 is True

    def test_build_training_args(self):
        from src.finetune.v2.post_training.round3_distillation import (
            Round3Config,
            _build_training_args,
        )

        cfg = Round3Config()
        out = Path("/tmp/test_round3")
        args = _build_training_args(cfg, out)

        assert args["num_train_epochs"] == 1
        assert args["learning_rate"] == 5e-7
        assert args["output_dir"] == str(out)
