# tests/unit/finetune/v2/test_train_phase3_5_insights.py
"""Unit tests for train_phase3_5_insights — INSIGHT_CATEGORIES, InsightPhaseConfig,
_build_training_args, and run_phase3_5."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# INSIGHT_CATEGORIES
# ---------------------------------------------------------------------------


class TestInsightCategories:
    def _categories(self):
        from src.finetune.v2.train_phase3_5_insights import INSIGHT_CATEGORIES
        return INSIGHT_CATEGORIES

    def test_is_list(self):
        assert isinstance(self._categories(), list)

    def test_has_five_entries(self):
        assert len(self._categories()) == 5

    def test_contains_pattern_recognition(self):
        assert "pattern_recognition" in self._categories()

    def test_contains_anomaly_detection(self):
        assert "anomaly_detection" in self._categories()

    def test_contains_trend_analysis(self):
        assert "trend_analysis" in self._categories()

    def test_contains_comparative_analysis(self):
        assert "comparative_analysis" in self._categories()

    def test_contains_gap_analysis(self):
        assert "gap_analysis" in self._categories()

    def test_all_entries_are_strings(self):
        for entry in self._categories():
            assert isinstance(entry, str), f"Non-string entry: {entry!r}"

    def test_no_duplicate_entries(self):
        cats = self._categories()
        assert len(cats) == len(set(cats))

    def test_exact_order(self):
        expected = [
            "pattern_recognition",
            "anomaly_detection",
            "trend_analysis",
            "comparative_analysis",
            "gap_analysis",
        ]
        assert self._categories() == expected


# ---------------------------------------------------------------------------
# InsightPhaseConfig — default values
# ---------------------------------------------------------------------------


class TestInsightPhaseConfigDefaults:
    def _cfg(self):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig
        return InsightPhaseConfig()

    def test_default_lora_r(self):
        assert self._cfg().lora_r == 64

    def test_default_lora_alpha(self):
        assert self._cfg().lora_alpha == 128

    def test_default_learning_rate(self):
        assert self._cfg().learning_rate == pytest.approx(1e-5)

    def test_default_epochs(self):
        assert self._cfg().epochs == 4

    def test_default_per_device_batch_size(self):
        assert self._cfg().per_device_batch_size == 4

    def test_default_gradient_accumulation_steps(self):
        assert self._cfg().gradient_accumulation_steps == 8

    def test_default_max_seq_length(self):
        assert self._cfg().max_seq_length == 4096

    def test_default_bf16(self):
        assert self._cfg().bf16 is True

    def test_default_lr_scheduler_type(self):
        assert self._cfg().lr_scheduler_type == "cosine"

    def test_default_warmup_ratio(self):
        assert self._cfg().warmup_ratio == pytest.approx(0.10)

    def test_default_output_dir(self):
        assert self._cfg().output_dir == Path("runs/v2/phase3_5_insights")

    def test_output_dir_is_path(self):
        assert isinstance(self._cfg().output_dir, Path)

    def test_default_gate_insight_precision(self):
        assert self._cfg().gate_insight_precision == pytest.approx(0.80)

    def test_default_gate_insight_recall(self):
        assert self._cfg().gate_insight_recall == pytest.approx(0.60)

    def test_effective_batch_size(self):
        """Effective batch size must equal per_device_batch_size * gradient_accumulation_steps."""
        cfg = self._cfg()
        effective = cfg.per_device_batch_size * cfg.gradient_accumulation_steps
        assert effective == 32

    def test_custom_values_accepted(self):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig
        cfg = InsightPhaseConfig(lora_r=32, epochs=2, learning_rate=5e-6)
        assert cfg.lora_r == 32
        assert cfg.epochs == 2
        assert cfg.learning_rate == pytest.approx(5e-6)


# ---------------------------------------------------------------------------
# _build_training_args
# ---------------------------------------------------------------------------


class TestBuildTrainingArgs:
    def _build(self, **overrides):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig, _build_training_args
        cfg = InsightPhaseConfig(**overrides)
        return _build_training_args(cfg, Path("/tmp/phase3_5_test"))

    def test_returns_dict(self):
        assert isinstance(self._build(), dict)

    def test_output_dir_is_string(self):
        args = self._build()
        assert isinstance(args["output_dir"], str)
        assert args["output_dir"] == "/tmp/phase3_5_test"

    def test_num_train_epochs(self):
        args = self._build(epochs=4)
        assert args["num_train_epochs"] == 4

    def test_per_device_train_batch_size(self):
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

    def test_max_seq_length(self):
        args = self._build(max_seq_length=4096)
        assert args["max_seq_length"] == 4096

    def test_bf16_true(self):
        args = self._build(bf16=True)
        assert args["bf16"] is True

    def test_fp16_false_when_bf16(self):
        args = self._build(bf16=True)
        assert args["fp16"] is False

    def test_report_to_none(self):
        args = self._build()
        assert args["report_to"] == "none"

    def test_seed_present(self):
        args = self._build()
        assert "seed" in args

    def test_logging_steps_present(self):
        args = self._build()
        assert "logging_steps" in args

    def test_save_steps_present(self):
        args = self._build()
        assert "save_steps" in args

    def test_required_keys_present(self):
        args = self._build()
        required = [
            "output_dir", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
            "warmup_ratio", "max_seq_length", "bf16", "logging_steps", "save_steps",
            "report_to", "seed",
        ]
        for key in required:
            assert key in args, f"Missing key: {key}"

    def test_output_dir_uses_passed_path_not_config(self):
        """output_dir in result must reflect the *output_dir* argument, not config.output_dir."""
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig, _build_training_args
        cfg = InsightPhaseConfig(output_dir=Path("/ignored/path"))
        args = _build_training_args(cfg, Path("/actual/path"))
        assert args["output_dir"] == "/actual/path"

    def test_custom_epochs_reflected(self):
        args = self._build(epochs=2)
        assert args["num_train_epochs"] == 2

    def test_custom_batch_size_reflected(self):
        args = self._build(per_device_batch_size=8)
        assert args["per_device_train_batch_size"] == 8

    def test_custom_learning_rate_reflected(self):
        args = self._build(learning_rate=2e-5)
        assert args["learning_rate"] == pytest.approx(2e-5)


# ---------------------------------------------------------------------------
# run_phase3_5 (smoke tests, heavily mocked)
# ---------------------------------------------------------------------------


class TestRunPhase35:
    def _make_mock_model(self):
        model = MagicMock()
        model._text_model = MagicMock()
        model._tokenizer = MagicMock()
        model._projection = MagicMock()
        model._projection.parameters.return_value = [MagicMock(requires_grad=False)]
        return model

    def _vg_mock(self, mock_model):
        vg = MagicMock()
        vg.VisionGraftedModel.return_value = mock_model
        vg.GraftConfig = MagicMock()
        return vg

    def test_returns_output_dir(self, tmp_path):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig, run_phase3_5

        cfg = InsightPhaseConfig(output_dir=tmp_path / "out")
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase3_5_insights._load_insight_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                result = run_phase3_5(cfg, phase3_dir=tmp_path / "phase3")

        assert result == cfg.output_dir

    def test_completion_marker_written(self, tmp_path):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig, run_phase3_5

        cfg = InsightPhaseConfig(output_dir=tmp_path / "out")
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase3_5_insights._load_insight_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                out = run_phase3_5(cfg, phase3_dir=tmp_path / "phase3")

        assert (out / ".phase3_5_complete").exists()

    def test_uses_default_config_when_none(self, tmp_path):
        from src.finetune.v2.train_phase3_5_insights import run_phase3_5

        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase3_5_insights._load_insight_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                with patch("src.finetune.v2.train_phase3_5_insights.InsightPhaseConfig") as mock_cfg_cls:
                    mock_cfg = MagicMock()
                    mock_cfg.output_dir = tmp_path / "out"
                    mock_cfg.lora_r = 64
                    mock_cfg.lora_alpha = 128
                    mock_cfg.learning_rate = 1e-5
                    mock_cfg.epochs = 4
                    mock_cfg.per_device_batch_size = 4
                    mock_cfg.gradient_accumulation_steps = 8
                    mock_cfg.max_seq_length = 4096
                    mock_cfg.bf16 = True
                    mock_cfg_cls.return_value = mock_cfg

                    run_phase3_5(None, phase3_dir=tmp_path / "phase3")
                    mock_cfg_cls.assert_called_once_with()

    def test_phase3_dir_override(self, tmp_path):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig, run_phase3_5

        custom_phase3 = tmp_path / "custom_phase3"
        cfg = InsightPhaseConfig(output_dir=tmp_path / "out")
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase3_5_insights._load_insight_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                run_phase3_5(cfg, phase3_dir=custom_phase3)

        mock_model.load_projection.assert_called_once_with(
            checkpoint=custom_phase3 / "projection.pt"
        )

    def test_lora_applied_with_config_values(self, tmp_path):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig, run_phase3_5

        cfg = InsightPhaseConfig(output_dir=tmp_path / "out", lora_r=64, lora_alpha=128)
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase3_5_insights._load_insight_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                run_phase3_5(cfg, phase3_dir=tmp_path / "phase3")

        mock_model.add_lora.assert_called_once_with(r=64, lora_alpha=128)

    def test_output_dir_created(self, tmp_path):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig, run_phase3_5

        out_dir = tmp_path / "nested" / "out"
        cfg = InsightPhaseConfig(output_dir=out_dir)
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase3_5_insights._load_insight_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                run_phase3_5(cfg, phase3_dir=tmp_path / "phase3")

        assert out_dir.exists()

    def test_checkpoint_final_dir_created(self, tmp_path):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig, run_phase3_5

        cfg = InsightPhaseConfig(output_dir=tmp_path / "out")
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase3_5_insights._load_insight_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                out = run_phase3_5(cfg, phase3_dir=tmp_path / "phase3")

        assert (out / "checkpoint_final").exists()

    def test_vision_encoder_loaded(self, tmp_path):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig, run_phase3_5

        cfg = InsightPhaseConfig(output_dir=tmp_path / "out")
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase3_5_insights._load_insight_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                run_phase3_5(cfg, phase3_dir=tmp_path / "phase3")

        mock_model.load_vision_encoder.assert_called_once()

    def test_text_model_loaded(self, tmp_path):
        from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig, run_phase3_5

        cfg = InsightPhaseConfig(output_dir=tmp_path / "out")
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase3_5_insights._load_insight_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                run_phase3_5(cfg, phase3_dir=tmp_path / "phase3")

        mock_model.load_text_model.assert_called_once()
