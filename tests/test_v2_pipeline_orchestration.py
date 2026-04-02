"""Tests for V2+ pipeline orchestrator — Phase 3.7 and post-training rounds."""

import tempfile
from pathlib import Path

import pytest

from src.finetune.v2.merge_promote import Phase4Config, get_new_capability_criteria
from src.finetune.v2.pipeline import PHASE_MARKERS, V2Pipeline


class TestV2PlusPipeline:
    """Verify the updated pipeline includes Phase 3.7 and post-training rounds."""

    def test_phases_include_3_7(self):
        pipeline = V2Pipeline()
        assert "phase3_7" in pipeline.phases

    def test_phase_ordering(self):
        pipeline = V2Pipeline()
        idx_35 = pipeline.phases.index("phase3_5")
        idx_37 = pipeline.phases.index("phase3_7")
        idx_4 = pipeline.phases.index("phase4")
        assert idx_35 < idx_37 < idx_4

    def test_post_training_phases(self):
        pipeline = V2Pipeline()
        for phase in ("round1", "round2", "round3"):
            assert phase in pipeline.phases

    def test_phase_markers_include_3_7(self):
        for key in ("phase3_7", "round1", "round2", "round3"):
            assert key in PHASE_MARKERS

    def test_next_phase_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            pipeline = V2Pipeline(base_dir=base)

            # Mark phase1 complete
            (base / "phase1").mkdir(parents=True)
            (base / "phase1" / PHASE_MARKERS["phase1"]).touch()

            # Mark phase2 complete
            (base / "phase2").mkdir(parents=True)
            (base / "phase2" / PHASE_MARKERS["phase2"]).touch()

            assert pipeline.next_phase() == "phase2_5"

    def test_merge_promote_config_includes_phase37(self):
        config = Phase4Config()
        assert "phase3_7" in str(config.phase3_dir)

    def test_new_capability_criteria_has_v2plus_metrics(self):
        criteria = get_new_capability_criteria()
        assert "synthesis_coherence" in criteria
        assert "intent_alignment" in criteria
        assert "depth_calibration" in criteria
        assert criteria["synthesis_coherence"] == 0.80
        assert criteria["intent_alignment"] == 0.85
        assert criteria["depth_calibration"] == 0.75
