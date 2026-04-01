"""Unit tests for src/finetune/v2/pipeline.py and merge_promote.py."""

from __future__ import annotations

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# PHASE_MARKERS & V2Pipeline structure tests
# ---------------------------------------------------------------------------


def test_phase_markers_includes_phase2_5():
    from src.finetune.v2.pipeline import PHASE_MARKERS

    assert "phase2_5" in PHASE_MARKERS
    assert PHASE_MARKERS["phase2_5"] == ".phase2_5_complete"


def test_phase_markers_includes_phase3_5():
    from src.finetune.v2.pipeline import PHASE_MARKERS

    assert "phase3_5" in PHASE_MARKERS
    assert PHASE_MARKERS["phase3_5"] == ".phase3_5_complete"


def test_pipeline_has_six_phases():
    from src.finetune.v2.pipeline import V2Pipeline

    assert len(V2Pipeline.phases) == 6


def test_pipeline_phase_order():
    from src.finetune.v2.pipeline import V2Pipeline

    assert V2Pipeline.phases == [
        "phase1",
        "phase2",
        "phase2_5",
        "phase3",
        "phase3_5",
        "phase4",
    ]


# ---------------------------------------------------------------------------
# Pipeline status: phase2_5 is next after phase1 + phase2 complete
# ---------------------------------------------------------------------------


def test_next_phase_is_phase2_5_after_phase1_and_phase2(tmp_path):
    from src.finetune.v2.pipeline import V2Pipeline, PHASE_MARKERS

    # Create marker artifacts for phase1 and phase2 only
    for phase in ("phase1", "phase2"):
        phase_dir = tmp_path / phase
        phase_dir.mkdir()
        (phase_dir / PHASE_MARKERS[phase]).write_text("done")

    pipeline = V2Pipeline(base_dir=tmp_path)

    assert pipeline.next_phase() == "phase2_5"


def test_status_next_phase_is_phase2_5_after_phase1_and_phase2(tmp_path):
    from src.finetune.v2.pipeline import V2Pipeline, PHASE_MARKERS

    for phase in ("phase1", "phase2"):
        phase_dir = tmp_path / phase
        phase_dir.mkdir()
        (phase_dir / PHASE_MARKERS[phase]).write_text("done")

    pipeline = V2Pipeline(base_dir=tmp_path)
    status = pipeline.status()

    assert status["next_phase"] == "phase2_5"
    assert "phase1" in status["completed_phases"]
    assert "phase2" in status["completed_phases"]
    assert "phase2_5" not in status["completed_phases"]


# ---------------------------------------------------------------------------
# merge_promote: new capability criteria include new metrics
# ---------------------------------------------------------------------------


def test_merge_regression_includes_insight_precision():
    from src.finetune.v2.merge_promote import get_new_capability_criteria

    criteria = get_new_capability_criteria()
    assert "insight_precision" in criteria
    assert criteria["insight_precision"] == pytest.approx(0.80)


def test_merge_regression_includes_confidence_calibration_ece():
    from src.finetune.v2.merge_promote import get_new_capability_criteria

    criteria = get_new_capability_criteria()
    assert "confidence_calibration_ece" in criteria
    assert criteria["confidence_calibration_ece"] == pytest.approx(0.10)


def test_merge_regression_existing_criteria_unchanged():
    from src.finetune.v2.merge_promote import get_new_capability_criteria

    criteria = get_new_capability_criteria()
    assert criteria["vision_accuracy"] == pytest.approx(70.0)
    assert criteria["table_extraction_f1"] == pytest.approx(75.0)
    assert criteria["tool_call_accuracy"] == pytest.approx(80.0)
    assert criteria["tool_arg_correctness"] == pytest.approx(85.0)
    assert criteria["layout_detection_map"] == pytest.approx(65.0)
