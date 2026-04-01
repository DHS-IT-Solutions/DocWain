# tests/unit/finetune/v2/test_train_phase2_5_dpo.py
"""Unit tests for train_phase2_5_dpo — DPOPhaseConfig, helper functions, and corrupt_extraction."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# DPOPhaseConfig — default values
# ---------------------------------------------------------------------------


class TestDPOPhaseConfigDefaults:
    def _cfg(self):
        from src.finetune.v2.train_phase2_5_dpo import DPOPhaseConfig
        return DPOPhaseConfig()

    def test_default_beta(self):
        assert self._cfg().beta == pytest.approx(0.1)

    def test_default_learning_rate(self):
        assert self._cfg().learning_rate == pytest.approx(5e-6)

    def test_default_lr_scheduler_type(self):
        assert self._cfg().lr_scheduler_type == "cosine"

    def test_default_warmup_ratio(self):
        assert self._cfg().warmup_ratio == pytest.approx(0.10)

    def test_default_epochs(self):
        assert self._cfg().epochs == 3

    def test_default_per_device_batch_size(self):
        assert self._cfg().per_device_batch_size == 2

    def test_default_gradient_accumulation_steps(self):
        assert self._cfg().gradient_accumulation_steps == 16

    def test_default_max_prompt_length(self):
        assert self._cfg().max_prompt_length == 2048

    def test_default_max_response_length(self):
        assert self._cfg().max_response_length == 2048

    def test_default_bf16(self):
        assert self._cfg().bf16 is True

    def test_default_output_dir(self):
        assert self._cfg().output_dir == Path("runs/v2/phase2_5_dpo")

    def test_output_dir_is_path(self):
        assert isinstance(self._cfg().output_dir, Path)

    def test_default_gate_hallucination_rate(self):
        assert self._cfg().gate_hallucination_rate == pytest.approx(0.05)

    def test_default_gate_extraction_f1_improvement(self):
        assert self._cfg().gate_extraction_f1_improvement == pytest.approx(0.05)

    def test_effective_batch_size(self):
        """Effective batch size must equal per_device_batch_size * gradient_accumulation_steps."""
        cfg = self._cfg()
        effective = cfg.per_device_batch_size * cfg.gradient_accumulation_steps
        assert effective == 32

    def test_custom_values_accepted(self):
        from src.finetune.v2.train_phase2_5_dpo import DPOPhaseConfig
        cfg = DPOPhaseConfig(beta=0.2, epochs=5, learning_rate=1e-5)
        assert cfg.beta == pytest.approx(0.2)
        assert cfg.epochs == 5
        assert cfg.learning_rate == pytest.approx(1e-5)


# ---------------------------------------------------------------------------
# _build_dpo_training_args
# ---------------------------------------------------------------------------


class TestBuildDPOTrainingArgs:
    def _build(self, **overrides):
        from src.finetune.v2.train_phase2_5_dpo import DPOPhaseConfig, _build_dpo_training_args
        cfg = DPOPhaseConfig(**overrides)
        return _build_dpo_training_args(cfg, Path("/tmp/phase2_5_test"))

    def test_returns_dict(self):
        assert isinstance(self._build(), dict)

    def test_output_dir_is_string(self):
        args = self._build()
        assert isinstance(args["output_dir"], str)
        assert args["output_dir"] == "/tmp/phase2_5_test"

    def test_beta_present(self):
        args = self._build(beta=0.1)
        assert args["beta"] == pytest.approx(0.1)

    def test_learning_rate(self):
        args = self._build(learning_rate=5e-6)
        assert args["learning_rate"] == pytest.approx(5e-6)

    def test_num_train_epochs(self):
        args = self._build(epochs=3)
        assert args["num_train_epochs"] == 3

    def test_per_device_train_batch_size(self):
        args = self._build(per_device_batch_size=2)
        assert args["per_device_train_batch_size"] == 2

    def test_gradient_accumulation_steps(self):
        args = self._build(gradient_accumulation_steps=16)
        assert args["gradient_accumulation_steps"] == 16

    def test_lr_scheduler_type(self):
        args = self._build(lr_scheduler_type="cosine")
        assert args["lr_scheduler_type"] == "cosine"

    def test_warmup_ratio(self):
        args = self._build(warmup_ratio=0.10)
        assert args["warmup_ratio"] == pytest.approx(0.10)

    def test_max_prompt_length(self):
        args = self._build(max_prompt_length=2048)
        assert args["max_prompt_length"] == 2048

    def test_max_length_is_prompt_plus_response(self):
        args = self._build(max_prompt_length=2048, max_response_length=2048)
        assert args["max_length"] == 4096

    def test_max_length_custom(self):
        args = self._build(max_prompt_length=1024, max_response_length=512)
        assert args["max_length"] == 1536

    def test_bf16_true(self):
        args = self._build(bf16=True)
        assert args["bf16"] is True

    def test_fp16_false_when_bf16(self):
        args = self._build(bf16=True)
        assert args["fp16"] is False

    def test_required_keys_present(self):
        args = self._build()
        required = [
            "output_dir", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
            "warmup_ratio", "beta", "max_prompt_length", "max_length",
            "bf16", "logging_steps", "save_steps",
        ]
        for key in required:
            assert key in args, f"Missing key: {key}"

    def test_report_to_none(self):
        args = self._build()
        assert args["report_to"] == "none"

    def test_seed_present(self):
        args = self._build()
        assert "seed" in args


# ---------------------------------------------------------------------------
# corrupt_extraction
# ---------------------------------------------------------------------------


class TestCorruptExtraction:
    """Tests that corrupt_extraction returns a mutated deep copy."""

    def _good(self):
        return {
            "entities": ["Alice", "Bob", "Charlie"],
            "tables": [
                [["cell_00", "cell_01"], ["cell_10", "cell_11"]],
            ],
            "fields": {
                "invoice_number": "INV-001",
                "total": "1500.00",
                "date": "2026-01-01",
                "vendor": "Acme Corp",
            },
        }

    def test_returns_dict(self):
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        result = corrupt_extraction(self._good())
        assert isinstance(result, dict)

    def test_result_differs_from_input(self):
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        good = self._good()
        corrupted = corrupt_extraction(good, seed=0)
        assert corrupted != good

    def test_does_not_mutate_original(self):
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        good = self._good()
        import copy
        original_snapshot = copy.deepcopy(good)
        corrupt_extraction(good, seed=0)
        assert good == original_snapshot

    def test_reproducible_with_same_seed(self):
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        good = self._good()
        r1 = corrupt_extraction(good, seed=7)
        r2 = corrupt_extraction(good, seed=7)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        good = self._good()
        r0 = corrupt_extraction(good, seed=0)
        r1 = corrupt_extraction(good, seed=1)
        # Not guaranteed for every input, but with rich content they should differ
        # We check that at least one of the many seeds changes something
        assert r0 != good or r1 != good

    def test_preserves_top_level_keys(self):
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        good = self._good()
        corrupted = corrupt_extraction(good, seed=0)
        assert set(corrupted.keys()) >= {"entities", "tables", "fields"}

    def test_entities_list_shrinks_or_values_change(self):
        """After corruption the entities list may have shrunk or another field changed."""
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        good = self._good()
        corrupted = corrupt_extraction(good, seed=0)
        # The overall structure must differ — some key must have changed
        assert (
            corrupted["entities"] != good["entities"]
            or corrupted["fields"] != good["fields"]
            or corrupted["tables"] != good["tables"]
        )

    def test_hallucinate_value_injects_marker(self):
        """At least one seed should trigger hallucinate_value, injecting the marker string."""
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        found_marker = False
        good = self._good()
        for s in range(20):
            corrupted = corrupt_extraction(good, seed=s)
            for v in corrupted["fields"].values():
                if v == "__HALLUCINATED_VALUE__":
                    found_marker = True
                    break
            if found_marker:
                break
        assert found_marker, "No seed in 0-19 triggered hallucinate_value"

    def test_drop_entity_reduces_count(self):
        """At least one seed should trigger drop_entity, reducing entity count."""
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        good = self._good()
        original_count = len(good["entities"])
        found = False
        for s in range(20):
            corrupted = corrupt_extraction(good, seed=s)
            if len(corrupted["entities"]) < original_count:
                found = True
                break
        assert found, "No seed in 0-19 triggered drop_entity"

    def test_wrong_field_swaps_values(self):
        """At least one seed should trigger wrong_field, swapping two field values."""
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        good = self._good()
        good_values = set(good["fields"].values())
        found = False
        for s in range(20):
            corrupted = corrupt_extraction(good, seed=s)
            # If a swap happened, the set of values stays the same but mapping differs
            if (
                set(corrupted["fields"].values()) == good_values
                and corrupted["fields"] != good["fields"]
            ):
                found = True
                break
        assert found, "No seed in 0-19 triggered wrong_field"

    def test_empty_extraction_does_not_raise(self):
        """corrupt_extraction must not crash on sparse/empty input."""
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        empty = {"entities": [], "tables": [], "fields": {}}
        result = corrupt_extraction(empty, seed=0)
        assert isinstance(result, dict)

    def test_partial_extraction_does_not_raise(self):
        """Handles missing keys gracefully."""
        from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
        partial = {"entities": ["X"], "tables": [], "fields": {"a": "1"}}
        result = corrupt_extraction(partial, seed=3)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# run_phase2_5 (smoke tests, heavily mocked)
# ---------------------------------------------------------------------------


class TestRunPhase25:
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
        from src.finetune.v2.train_phase2_5_dpo import DPOPhaseConfig, run_phase2_5

        cfg = DPOPhaseConfig(output_dir=tmp_path / "out")
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase2_5_dpo._load_dpo_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                result = run_phase2_5(cfg, phase2_dir=tmp_path / "phase2")

        assert result == cfg.output_dir

    def test_completion_marker_written(self, tmp_path):
        from src.finetune.v2.train_phase2_5_dpo import DPOPhaseConfig, run_phase2_5

        cfg = DPOPhaseConfig(output_dir=tmp_path / "out")
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase2_5_dpo._load_dpo_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                out = run_phase2_5(cfg, phase2_dir=tmp_path / "phase2")

        assert (out / ".phase2_5_complete").exists()

    def test_uses_default_config_when_none(self, tmp_path):
        from src.finetune.v2.train_phase2_5_dpo import run_phase2_5

        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase2_5_dpo._load_dpo_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                with patch("src.finetune.v2.train_phase2_5_dpo.DPOPhaseConfig") as mock_cfg_cls:
                    mock_cfg = MagicMock()
                    mock_cfg.output_dir = tmp_path / "out"
                    mock_cfg.beta = 0.1
                    mock_cfg.learning_rate = 5e-6
                    mock_cfg.epochs = 3
                    mock_cfg.per_device_batch_size = 2
                    mock_cfg.gradient_accumulation_steps = 16
                    mock_cfg.max_prompt_length = 2048
                    mock_cfg.max_response_length = 2048
                    mock_cfg.bf16 = True
                    mock_cfg_cls.return_value = mock_cfg

                    run_phase2_5(None, phase2_dir=tmp_path / "phase2")
                    mock_cfg_cls.assert_called_once_with()

    def test_phase2_dir_override(self, tmp_path):
        from src.finetune.v2.train_phase2_5_dpo import DPOPhaseConfig, run_phase2_5

        custom_phase2 = tmp_path / "custom_phase2"
        cfg = DPOPhaseConfig(output_dir=tmp_path / "out")
        mock_model = self._make_mock_model()
        vg_mock = self._vg_mock(mock_model)

        with patch("src.finetune.v2.train_phase2_5_dpo._load_dpo_dataset", return_value=None):
            with patch.dict("sys.modules", {
                "trl": MagicMock(),
                "src.finetune.v2.vision_graft": vg_mock,
            }):
                run_phase2_5(cfg, phase2_dir=custom_phase2)

        mock_model.load_projection.assert_called_once_with(
            checkpoint=custom_phase2 / "projection.pt"
        )
