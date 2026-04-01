# tests/unit/finetune/v2/test_train_phase2.py
"""Unit tests for train_phase2 — Phase2Config, helper functions, and training loop."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Phase2Config
# ---------------------------------------------------------------------------


class TestPhase2Config:
    def test_default_lora_r(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.lora_r == 64

    def test_default_lora_alpha(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.lora_alpha == 128

    def test_default_learning_rate(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.learning_rate == pytest.approx(2e-5)

    def test_default_epochs(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.epochs == 8

    def test_default_curriculum_stages(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.curriculum_stages == 4

    def test_default_per_device_batch_size(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.per_device_batch_size == 4

    def test_default_gradient_accumulation_steps(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.gradient_accumulation_steps == 8

    def test_default_max_seq_length(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.max_seq_length == 4096

    def test_default_bf16_true(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.bf16 is True

    def test_default_lr_scheduler_type(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.lr_scheduler_type == "cosine"

    def test_default_warmup_ratio(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.warmup_ratio == pytest.approx(0.10)

    def test_default_checkpoint_steps(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.checkpoint_steps == 500

    def test_default_lora_target_modules(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        expected = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert cfg.lora_target_modules == expected

    def test_default_dataset_mix_keys(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert set(cfg.dataset_mix.keys()) == {"table", "layout", "ocr", "cross_ref"}

    def test_default_dataset_mix_sum(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert sum(cfg.dataset_mix.values()) == pytest.approx(1.0)

    def test_default_gate_docvqa_accuracy(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.gate_docvqa_accuracy == pytest.approx(0.75)

    def test_default_gate_table_f1(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.gate_table_f1 == pytest.approx(0.80)

    def test_default_gate_layout_map(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert cfg.gate_layout_map == pytest.approx(0.70)

    def test_output_dir_is_path(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config()
        assert isinstance(cfg.output_dir, Path)

    def test_dataset_mix_is_independent_per_instance(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg1 = Phase2Config()
        cfg2 = Phase2Config()
        cfg1.dataset_mix["new_key"] = 0.1
        assert "new_key" not in cfg2.dataset_mix

    def test_custom_values_accepted(self):
        from src.finetune.v2.train_phase2 import Phase2Config

        cfg = Phase2Config(lora_r=32, epochs=4, learning_rate=1e-4)
        assert cfg.lora_r == 32
        assert cfg.epochs == 4
        assert cfg.learning_rate == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# _get_curriculum_epochs
# ---------------------------------------------------------------------------


class TestGetCurriculumEpochs:
    def test_even_split(self):
        from src.finetune.v2.train_phase2 import Phase2Config, _get_curriculum_epochs

        cfg = Phase2Config(epochs=8, curriculum_stages=4)
        result = _get_curriculum_epochs(cfg)
        assert result == [(0, 2), (2, 4), (4, 6), (6, 8)]

    def test_single_stage(self):
        from src.finetune.v2.train_phase2 import Phase2Config, _get_curriculum_epochs

        cfg = Phase2Config(epochs=5, curriculum_stages=1)
        result = _get_curriculum_epochs(cfg)
        assert len(result) == 1
        assert result[0] == (0, 5)

    def test_stages_cover_all_epochs(self):
        from src.finetune.v2.train_phase2 import Phase2Config, _get_curriculum_epochs

        cfg = Phase2Config(epochs=10, curriculum_stages=3)
        result = _get_curriculum_epochs(cfg)
        # First epoch of first stage is 0
        assert result[0][0] == 0
        # Last epoch of last stage is total epochs
        assert result[-1][1] == 10

    def test_no_gaps_between_stages(self):
        from src.finetune.v2.train_phase2 import Phase2Config, _get_curriculum_epochs

        cfg = Phase2Config(epochs=7, curriculum_stages=3)
        result = _get_curriculum_epochs(cfg)
        for i in range(len(result) - 1):
            assert result[i][1] == result[i + 1][0], f"Gap between stage {i} and {i+1}"

    def test_total_epochs_preserved(self):
        for epochs, stages in [(8, 4), (10, 3), (7, 4), (5, 2), (1, 1)]:
            from src.finetune.v2.train_phase2 import Phase2Config, _get_curriculum_epochs

            cfg = Phase2Config(epochs=epochs, curriculum_stages=stages)
            result = _get_curriculum_epochs(cfg)
            total = sum(e - s for s, e in result)
            assert total == epochs, f"epochs={epochs} stages={stages}: got {total}"

    def test_each_stage_has_positive_width(self):
        from src.finetune.v2.train_phase2 import Phase2Config, _get_curriculum_epochs

        cfg = Phase2Config(epochs=4, curriculum_stages=4)
        result = _get_curriculum_epochs(cfg)
        for start, end in result:
            assert end > start

    def test_returns_list_of_tuples(self):
        from src.finetune.v2.train_phase2 import Phase2Config, _get_curriculum_epochs

        cfg = Phase2Config()
        result = _get_curriculum_epochs(cfg)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_stages_capped_at_epochs(self):
        """When stages > epochs, no zero-width entries should appear."""
        from src.finetune.v2.train_phase2 import Phase2Config, _get_curriculum_epochs

        cfg = Phase2Config(epochs=2, curriculum_stages=8)
        result = _get_curriculum_epochs(cfg)
        for start, end in result:
            assert end > start


# ---------------------------------------------------------------------------
# _build_training_args
# ---------------------------------------------------------------------------


class TestBuildTrainingArgs:
    def _build(self, **overrides):
        from src.finetune.v2.train_phase2 import Phase2Config, _build_training_args

        cfg = Phase2Config(**overrides)
        return _build_training_args(cfg, Path("/tmp/phase2_test"))

    def test_returns_dict(self):
        assert isinstance(self._build(), dict)

    def test_output_dir_is_string(self):
        args = self._build()
        assert isinstance(args["output_dir"], str)
        assert args["output_dir"] == "/tmp/phase2_test"

    def test_num_train_epochs_matches_config(self):
        args = self._build(epochs=8)
        assert args["num_train_epochs"] == 8

    def test_per_device_batch_size(self):
        args = self._build(per_device_batch_size=4)
        assert args["per_device_train_batch_size"] == 4

    def test_gradient_accumulation_steps(self):
        args = self._build(gradient_accumulation_steps=8)
        assert args["gradient_accumulation_steps"] == 8

    def test_learning_rate(self):
        args = self._build(learning_rate=2e-5)
        assert args["learning_rate"] == pytest.approx(2e-5)

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
        args = self._build(checkpoint_steps=500)
        assert args["save_steps"] == 500

    def test_required_keys_present(self):
        args = self._build()
        required = [
            "output_dir", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
            "warmup_ratio", "bf16", "logging_steps", "save_steps", "max_seq_length",
        ]
        for key in required:
            assert key in args, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# _load_stage_dataset
# ---------------------------------------------------------------------------


class TestLoadStageDataset:
    def test_returns_none_for_missing_file(self, tmp_path):
        from src.finetune.v2.train_phase2 import _load_stage_dataset

        result = _load_stage_dataset("nonexistent_stage", tmp_path)
        assert result is None

    def test_returns_dataset_for_existing_file(self, tmp_path):
        import json as _json

        jsonl_path = tmp_path / "table.jsonl"
        rows = [{"text": f"row {i}", "label": i} for i in range(5)]
        jsonl_path.write_text("\n".join(_json.dumps(r) for r in rows))

        from src.finetune.v2.train_phase2 import _load_stage_dataset

        ds = _load_stage_dataset("table", tmp_path)
        assert ds is not None
        assert len(ds) == 5

    def test_dataset_has_expected_fields(self, tmp_path):
        import json as _json

        jsonl_path = tmp_path / "ocr.jsonl"
        rows = [{"text": "hello", "label": 1}]
        jsonl_path.write_text(_json.dumps(rows[0]))

        from src.finetune.v2.train_phase2 import _load_stage_dataset

        ds = _load_stage_dataset("ocr", tmp_path)
        assert ds is not None
        assert "text" in ds.column_names


# ---------------------------------------------------------------------------
# run_phase2 (integration-level, heavily mocked)
# ---------------------------------------------------------------------------


class TestRunPhase2:
    """Smoke-tests for run_phase2 using mocked heavy dependencies.

    GraftConfig and VisionGraftedModel are imported lazily inside run_phase2
    via ``from .vision_graft import ...``, so we patch the vision_graft module
    directly rather than attributes on train_phase2.
    """

    def _make_mock_model(self):
        model = MagicMock()
        model._text_model = MagicMock()
        model._tokenizer = MagicMock()
        model._projection = MagicMock()
        model._projection.parameters.return_value = [MagicMock(requires_grad=False)]
        return model

    def _vision_graft_mock(self, mock_model):
        """Return a mock vision_graft module whose VisionGraftedModel returns mock_model."""
        vg = MagicMock()
        vg.VisionGraftedModel.return_value = mock_model
        vg.GraftConfig = MagicMock()
        return vg

    @patch("src.finetune.v2.train_phase2._load_stage_dataset", return_value=None)
    def test_returns_output_dir(self, mock_load, tmp_path):
        from src.finetune.v2.train_phase2 import Phase2Config, run_phase2

        cfg = Phase2Config(
            output_dir=tmp_path / "out",
            data_dir=tmp_path / "data",
            phase1_checkpoint=tmp_path / "proj.pt",
            epochs=4,
            curriculum_stages=2,
        )
        mock_model = self._make_mock_model()
        vg_mock = self._vision_graft_mock(mock_model)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
        }):
            result = run_phase2(cfg)

        assert result == cfg.output_dir

    @patch("src.finetune.v2.train_phase2._load_stage_dataset", return_value=None)
    def test_uses_default_config_when_none(self, mock_load, tmp_path):
        """run_phase2(None) should not raise — it creates a default config."""
        from src.finetune.v2.train_phase2 import run_phase2

        mock_model = self._make_mock_model()
        vg_mock = self._vision_graft_mock(mock_model)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
        }):
            with patch("src.finetune.v2.train_phase2.Phase2Config") as mock_cfg_cls:
                mock_cfg = MagicMock()
                # Use a MagicMock for output_dir so mkdir can be freely mocked
                mock_output_dir = MagicMock()
                mock_cfg.output_dir = mock_output_dir
                mock_cfg.phase1_checkpoint = tmp_path / "proj.pt"
                mock_cfg.lora_r = 64
                mock_cfg.lora_alpha = 128
                mock_cfg.learning_rate = 2e-5
                mock_cfg.epochs = 8
                mock_cfg.per_device_batch_size = 4
                mock_cfg.curriculum_stages = 4
                mock_cfg.dataset_mix = {"table": 1.0}
                mock_cfg.data_dir = tmp_path
                mock_cfg_cls.return_value = mock_cfg

                run_phase2(None)
                mock_cfg_cls.assert_called_once_with()

    def test_phase1_checkpoint_override(self, tmp_path):
        from src.finetune.v2.train_phase2 import Phase2Config, run_phase2

        custom_ckpt = tmp_path / "custom_proj.pt"
        cfg = Phase2Config(
            output_dir=tmp_path / "out",
            data_dir=tmp_path / "data",
            phase1_checkpoint=tmp_path / "default_proj.pt",
            epochs=2,
            curriculum_stages=1,
        )
        mock_model = self._make_mock_model()
        vg_mock = self._vision_graft_mock(mock_model)

        with patch("src.finetune.v2.train_phase2._load_stage_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                run_phase2(cfg, phase1_checkpoint=custom_ckpt)

        mock_model.load_projection.assert_called_once_with(checkpoint=custom_ckpt)

    @patch("src.finetune.v2.train_phase2._load_stage_dataset", return_value=None)
    def test_model_lora_called_with_config_values(self, mock_load, tmp_path):
        from src.finetune.v2.train_phase2 import Phase2Config, run_phase2

        cfg = Phase2Config(
            output_dir=tmp_path / "out",
            data_dir=tmp_path / "data",
            phase1_checkpoint=tmp_path / "proj.pt",
            lora_r=64,
            lora_alpha=128,
            epochs=2,
            curriculum_stages=1,
        )
        mock_model = self._make_mock_model()
        vg_mock = self._vision_graft_mock(mock_model)

        with patch.dict("sys.modules", {
            "trl": MagicMock(),
            "src.finetune.v2.vision_graft": vg_mock,
        }):
            run_phase2(cfg)

        mock_model.add_lora.assert_called_once_with(r=64, lora_alpha=128)
