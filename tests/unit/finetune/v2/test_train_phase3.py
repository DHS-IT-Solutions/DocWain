# tests/unit/finetune/v2/test_train_phase3.py
"""Unit tests for train_phase3 — Phase3Config, _build_training_args, and run_phase3."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Phase3Config — defaults
# ---------------------------------------------------------------------------


class TestPhase3ConfigDefaults:
    def test_lora_r(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.lora_r == 64

    def test_lora_alpha(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.lora_alpha == 128

    def test_learning_rate(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.learning_rate == pytest.approx(1e-5)

    def test_epochs(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.epochs == 5

    def test_max_seq_length(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.max_seq_length == 4096

    def test_per_device_batch_size(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.per_device_batch_size == 4

    def test_gradient_accumulation_steps(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.gradient_accumulation_steps == 8

    def test_bf16_true(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.bf16 is True

    def test_lr_scheduler_type(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.lr_scheduler_type == "cosine"

    def test_warmup_ratio(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.warmup_ratio == pytest.approx(0.10)

    def test_freeze_projection_true(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.freeze_projection is True

    def test_lora_target_modules(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        expected = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert cfg.lora_target_modules == expected

    def test_output_dir_is_path(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert isinstance(cfg.output_dir, Path)

    def test_data_sources_list(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert set(cfg.data_sources) == {"synthetic", "toolbench", "gorilla", "nexusraven"}

    def test_gate_tool_accuracy(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.gate_tool_accuracy == pytest.approx(0.85)

    def test_gate_arg_correctness(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.gate_arg_correctness == pytest.approx(0.90)

    def test_gate_false_positive_rate(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.gate_false_positive_rate == pytest.approx(0.10)

    def test_custom_values_accepted(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config(lora_r=32, epochs=3, learning_rate=2e-5)
        assert cfg.lora_r == 32
        assert cfg.epochs == 3
        assert cfg.learning_rate == pytest.approx(2e-5)

    def test_source_weights_independent_per_instance(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg1 = Phase3Config()
        cfg2 = Phase3Config()
        cfg1.source_weights["new_source"] = 0.99
        assert "new_source" not in cfg2.source_weights

    def test_lora_target_modules_independent_per_instance(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg1 = Phase3Config()
        cfg2 = Phase3Config()
        cfg1.lora_target_modules.append("extra_proj")
        assert "extra_proj" not in cfg2.lora_target_modules


# ---------------------------------------------------------------------------
# Source weights sum to 1.0
# ---------------------------------------------------------------------------


class TestSourceWeightsSum:
    def test_default_source_weights_sum_to_one(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert sum(cfg.source_weights.values()) == pytest.approx(1.0)

    def test_source_weights_keys(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert set(cfg.source_weights.keys()) == {"synthetic", "toolbench", "gorilla", "nexusraven"}

    def test_source_weights_individual_values(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.source_weights["synthetic"] == pytest.approx(0.40)
        assert cfg.source_weights["toolbench"] == pytest.approx(0.25)
        assert cfg.source_weights["gorilla"] == pytest.approx(0.20)
        assert cfg.source_weights["nexusraven"] == pytest.approx(0.15)

    def test_source_weights_all_positive(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        for source, weight in cfg.source_weights.items():
            assert weight > 0, f"Weight for '{source}' must be positive"

    def test_source_weights_count_matches_data_sources(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert set(cfg.source_weights.keys()) == set(cfg.data_sources)


# ---------------------------------------------------------------------------
# freeze_projection=True
# ---------------------------------------------------------------------------


class TestFreezeProjection:
    def test_freeze_projection_default_true(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert cfg.freeze_projection is True

    def test_freeze_projection_can_be_disabled(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config(freeze_projection=False)
        assert cfg.freeze_projection is False

    def test_freeze_projection_is_bool(self):
        from src.finetune.v2.train_phase3 import Phase3Config

        cfg = Phase3Config()
        assert isinstance(cfg.freeze_projection, bool)


# ---------------------------------------------------------------------------
# _build_training_args
# ---------------------------------------------------------------------------


class TestBuildTrainingArgs:
    def _build(self, **overrides):
        from src.finetune.v2.train_phase3 import Phase3Config, _build_training_args

        cfg = Phase3Config(**overrides)
        return _build_training_args(cfg, Path("/tmp/phase3_test"))

    def test_returns_dict(self):
        assert isinstance(self._build(), dict)

    def test_output_dir_is_string(self):
        args = self._build()
        assert isinstance(args["output_dir"], str)
        assert args["output_dir"] == "/tmp/phase3_test"

    def test_num_train_epochs_matches_config(self):
        args = self._build(epochs=5)
        assert args["num_train_epochs"] == 5

    def test_per_device_batch_size(self):
        args = self._build(per_device_batch_size=4)
        assert args["per_device_train_batch_size"] == 4

    def test_gradient_accumulation_steps(self):
        args = self._build(gradient_accumulation_steps=8)
        assert args["gradient_accumulation_steps"] == 8

    def test_learning_rate(self):
        args = self._build(learning_rate=1e-5)
        assert args["learning_rate"] == pytest.approx(1e-5)

    def test_lr_scheduler_type(self):
        args = self._build(lr_scheduler_type="cosine")
        assert args["lr_scheduler_type"] == "cosine"

    def test_warmup_ratio(self):
        args = self._build(warmup_ratio=0.10)
        assert args["warmup_ratio"] == pytest.approx(0.10)

    def test_bf16_true(self):
        args = self._build(bf16=True)
        assert args["bf16"] is True

    def test_fp16_false_when_bf16_true(self):
        args = self._build(bf16=True)
        assert args["fp16"] is False

    def test_max_seq_length(self):
        args = self._build(max_seq_length=4096)
        assert args["max_seq_length"] == 4096

    def test_save_steps_matches_checkpoint_steps(self):
        args = self._build(checkpoint_steps=200)
        assert args["save_steps"] == 200

    def test_dataset_text_field(self):
        args = self._build()
        assert args["dataset_text_field"] == "text"

    def test_report_to_none(self):
        args = self._build()
        assert args["report_to"] == "none"

    def test_seed_set(self):
        args = self._build()
        assert args["seed"] == 42

    def test_required_keys_present(self):
        args = self._build()
        required = [
            "output_dir", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
            "warmup_ratio", "bf16", "fp16", "logging_steps", "save_steps",
            "eval_steps", "max_seq_length",
        ]
        for key in required:
            assert key in args, f"Missing key: {key}"

    def test_output_dir_override(self):
        from src.finetune.v2.train_phase3 import Phase3Config, _build_training_args

        cfg = Phase3Config()
        custom_dir = Path("/tmp/custom_phase3_output")
        args = _build_training_args(cfg, custom_dir)
        assert args["output_dir"] == str(custom_dir)

    def test_custom_epochs_reflected(self):
        args = self._build(epochs=10)
        assert args["num_train_epochs"] == 10


# ---------------------------------------------------------------------------
# run_phase3 (integration-level, heavily mocked)
# ---------------------------------------------------------------------------


class TestRunPhase3:
    """Smoke-tests for run_phase3 using mocked heavy dependencies."""

    def _make_mock_model(self):
        model = MagicMock()
        model._text_model = MagicMock()
        model._tokenizer = MagicMock()
        model._projection = MagicMock()
        model._projection.parameters.return_value = [MagicMock(requires_grad=True)]
        return model

    def _vision_graft_mock(self, mock_model):
        vg = MagicMock()
        vg.VisionGraftedModel.return_value = mock_model
        vg.GraftConfig = MagicMock()
        return vg

    def test_returns_output_dir(self, tmp_path):
        from src.finetune.v2.train_phase3 import Phase3Config, run_phase3

        cfg = Phase3Config(
            output_dir=tmp_path / "out",
            data_dir=tmp_path / "data",
            phase2_dir=tmp_path / "phase2_5",
        )
        mock_model = self._make_mock_model()
        vg_mock = self._vision_graft_mock(mock_model)

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
            "datasets": MagicMock(load_dataset=MagicMock(return_value=mock_dataset)),
        }):
            with patch("src.finetune.v2.train_phase3.build_tool_calling_dataset"):
                result = run_phase3(cfg)

        assert result == cfg.output_dir

    def test_phase3_complete_marker_created(self, tmp_path):
        from src.finetune.v2.train_phase3 import Phase3Config, run_phase3

        cfg = Phase3Config(
            output_dir=tmp_path / "out",
            data_dir=tmp_path / "data",
            phase2_dir=tmp_path / "phase2_5",
        )
        mock_model = self._make_mock_model()
        vg_mock = self._vision_graft_mock(mock_model)

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=50)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
            "datasets": MagicMock(load_dataset=MagicMock(return_value=mock_dataset)),
        }):
            with patch("src.finetune.v2.train_phase3.build_tool_calling_dataset"):
                run_phase3(cfg)

        assert (cfg.output_dir / ".phase3_complete").exists()

    def test_uses_default_config_when_none(self, tmp_path):
        from src.finetune.v2.train_phase3 import run_phase3

        mock_model = self._make_mock_model()
        vg_mock = self._vision_graft_mock(mock_model)

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
            "datasets": MagicMock(load_dataset=MagicMock(return_value=mock_dataset)),
        }):
            with patch("src.finetune.v2.train_phase3.Phase3Config") as mock_cfg_cls:
                mock_cfg = MagicMock()
                mock_cfg.output_dir = MagicMock()
                mock_cfg.phase2_dir = tmp_path / "phase2_5"
                mock_cfg.lora_r = 64
                mock_cfg.lora_alpha = 128
                mock_cfg.learning_rate = 1e-5
                mock_cfg.epochs = 5
                mock_cfg.per_device_batch_size = 4
                mock_cfg.freeze_projection = True
                mock_cfg.data_sources = ["synthetic"]
                mock_cfg.source_weights = {"synthetic": 1.0}
                mock_cfg.data_dir = tmp_path
                mock_cfg_cls.return_value = mock_cfg

                with patch("src.finetune.v2.train_phase3.build_tool_calling_dataset"):
                    run_phase3(None)

                mock_cfg_cls.assert_called_once_with()

    def test_phase2_dir_override(self, tmp_path):
        from src.finetune.v2.train_phase3 import Phase3Config, run_phase3

        custom_p2_dir = tmp_path / "custom_phase2_5"
        cfg = Phase3Config(
            output_dir=tmp_path / "out",
            data_dir=tmp_path / "data",
            phase2_dir=tmp_path / "default_phase2_5",
        )
        mock_model = self._make_mock_model()
        vg_mock = self._vision_graft_mock(mock_model)

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
            "datasets": MagicMock(load_dataset=MagicMock(return_value=mock_dataset)),
        }):
            with patch("src.finetune.v2.train_phase3.build_tool_calling_dataset"):
                run_phase3(cfg, phase2_dir=custom_p2_dir)

        mock_model.load_projection.assert_called_once_with(
            checkpoint=custom_p2_dir / "projection.pt"
        )

    def test_model_lora_called_with_config_values(self, tmp_path):
        from src.finetune.v2.train_phase3 import Phase3Config, run_phase3

        cfg = Phase3Config(
            output_dir=tmp_path / "out",
            data_dir=tmp_path / "data",
            phase2_dir=tmp_path / "phase2_5",
            lora_r=64,
            lora_alpha=128,
        )
        mock_model = self._make_mock_model()
        vg_mock = self._vision_graft_mock(mock_model)

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
            "datasets": MagicMock(load_dataset=MagicMock(return_value=mock_dataset)),
        }):
            with patch("src.finetune.v2.train_phase3.build_tool_calling_dataset"):
                run_phase3(cfg)

        mock_model.add_lora.assert_called_once_with(r=64, lora_alpha=128)

    def test_projection_frozen_when_freeze_projection_true(self, tmp_path):
        from src.finetune.v2.train_phase3 import Phase3Config, run_phase3

        cfg = Phase3Config(
            output_dir=tmp_path / "out",
            data_dir=tmp_path / "data",
            phase2_dir=tmp_path / "phase2_5",
            freeze_projection=True,
        )
        mock_model = self._make_mock_model()
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_model._projection.parameters.return_value = [mock_param]
        vg_mock = self._vision_graft_mock(mock_model)

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
            "datasets": MagicMock(load_dataset=MagicMock(return_value=mock_dataset)),
        }):
            with patch("src.finetune.v2.train_phase3.build_tool_calling_dataset"):
                run_phase3(cfg)

        assert mock_param.requires_grad is False

    def test_projection_not_frozen_when_freeze_projection_false(self, tmp_path):
        from src.finetune.v2.train_phase3 import Phase3Config, run_phase3

        cfg = Phase3Config(
            output_dir=tmp_path / "out",
            data_dir=tmp_path / "data",
            phase2_dir=tmp_path / "phase2_5",
            freeze_projection=False,
        )
        mock_model = self._make_mock_model()
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_model._projection.parameters.return_value = [mock_param]
        vg_mock = self._vision_graft_mock(mock_model)

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
            "datasets": MagicMock(load_dataset=MagicMock(return_value=mock_dataset)),
        }):
            with patch("src.finetune.v2.train_phase3.build_tool_calling_dataset"):
                run_phase3(cfg)

        # requires_grad should remain True (not set to False)
        assert mock_param.requires_grad is True

    def test_save_all_called(self, tmp_path):
        from src.finetune.v2.train_phase3 import Phase3Config, run_phase3

        cfg = Phase3Config(
            output_dir=tmp_path / "out",
            data_dir=tmp_path / "data",
            phase2_dir=tmp_path / "phase2_5",
        )
        mock_model = self._make_mock_model()
        vg_mock = self._vision_graft_mock(mock_model)

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
            "datasets": MagicMock(load_dataset=MagicMock(return_value=mock_dataset)),
        }):
            with patch("src.finetune.v2.train_phase3.build_tool_calling_dataset"):
                run_phase3(cfg)

        mock_model.save_all.assert_called_once_with(cfg.output_dir)
