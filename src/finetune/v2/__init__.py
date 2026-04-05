"""DocWain V2 — Curriculum-trained unified model.

Core modules:
- curriculum_trainer: Orchestrator for generate → train → eval → analyze loop
- curriculum_generator: Subagent-driven training data generation
- curriculum_evaluator: LoRA inference evaluation with subagent judges
- train_track: Unified SFT training with curriculum sampling
"""

from .curriculum_trainer import PipelineState, run_pipeline
from .curriculum_generator import (
    GenerationBrief,
    build_initial_briefs,
    build_augmentation_briefs,
    merge_datasets,
    AREA_CONFIGS,
)
from .curriculum_evaluator import (
    run_lora_inference,
    aggregate_scores,
    check_gates,
    GateResult,
)
from .train_track import TrackTrainingConfig, CurriculumSampler

__all__ = [
    "PipelineState",
    "run_pipeline",
    "GenerationBrief",
    "build_initial_briefs",
    "build_augmentation_briefs",
    "merge_datasets",
    "AREA_CONFIGS",
    "run_lora_inference",
    "aggregate_scores",
    "check_gates",
    "GateResult",
    "TrackTrainingConfig",
    "CurriculumSampler",
]
