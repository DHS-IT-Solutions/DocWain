"""Strategy evolver for the DocWain V2 iterative training loop.

Analyses evaluation results across iterations and evolves the training
strategy to target persistent weaknesses. No fixed iteration cap — the
strategy keeps adapting until quality gates pass.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingStrategy:
    """Mutable training configuration for a single track."""

    track: str
    learning_rate: float = 2e-5
    lora_r: int = 64
    lora_alpha: int = 128
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    warmup_ratio: float = 0.10

    # Data augmentation
    extra_sft_categories: List[str] = field(default_factory=list)
    extra_sft_count: int = 0
    add_dpo_pairs: bool = False
    dpo_count: int = 0

    # DPO config
    dpo_epochs: int = 2
    dpo_lr: float = 5e-6
    dpo_beta: float = 0.1

    # Curriculum
    curriculum_order: List[str] = field(default_factory=list)

    # Metadata
    strategy_version: int = 1
    changes_log: List[str] = field(default_factory=list)


@dataclass
class IterationRecord:
    """Record of a single training iteration's results."""

    iteration: int
    track: str
    scores: Dict[str, float]  # dimension -> score
    overall_avg: float
    passed: bool
    weak_categories: List[str]
    strategy_version: int
    duration_seconds: float = 0.0


class StrategyEvolver:
    """Analyses iteration history and evolves training strategy.

    Rules applied in order:
    1. Rounds 1-5: standard adjustments (more data for weak categories,
       DPO from failures, minor LR tweaks).
    2. After 5 rounds with same approach: major strategy shift —
       double data, restructure curriculum, increase LoRA rank.
    3. If score regresses: revert last change and try alternative.
    4. If a dimension is stuck below 3.0 for 3+ rounds:
       aggressive targeting with 2x data + lower LR.
    """

    def __init__(self) -> None:
        self.history: List[IterationRecord] = []

    def record(self, record: IterationRecord) -> None:
        """Store an iteration result."""
        self.history.append(record)

    def evolve(
        self,
        track: str,
        iteration: int,
        eval_result: Dict[str, Any],
        current_strategy: TrainingStrategy,
    ) -> TrainingStrategy:
        """Analyse eval results and return an evolved strategy."""

        new_strategy = deepcopy(current_strategy)
        new_strategy.strategy_version += 1
        new_strategy.changes_log = []

        scores = eval_result.get("dimensions", {})
        weak_cats = eval_result.get("weak_categories", [])
        overall = eval_result.get("overall_avg", 0.0)

        # Record this iteration
        self.record(IterationRecord(
            iteration=iteration,
            track=track,
            scores=scores,
            overall_avg=overall,
            passed=eval_result.get("passed", False),
            weak_categories=weak_cats,
            strategy_version=current_strategy.strategy_version,
        ))

        track_history = [r for r in self.history if r.track == track]

        # --- Rule 1: Target weak categories with more data ---
        if weak_cats:
            new_strategy.extra_sft_categories = weak_cats
            new_strategy.extra_sft_count = 200 * len(weak_cats)
            new_strategy.changes_log.append(
                f"Adding {new_strategy.extra_sft_count} targeted examples "
                f"for weak categories: {weak_cats}"
            )

        # --- Rule 2: Add DPO pairs from model failures ---
        if iteration >= 2:
            new_strategy.add_dpo_pairs = True
            new_strategy.dpo_count = max(200, 100 * len(weak_cats))
            new_strategy.changes_log.append(
                f"Adding {new_strategy.dpo_count} DPO pairs from model failures"
            )

        # --- Rule 3: Check for plateau ---
        if len(track_history) >= 2:
            prev_score = track_history[-2].overall_avg
            delta = overall - prev_score
            if abs(delta) < 0.1:
                # Plateau — adjust hyperparameters
                new_strategy.learning_rate *= 0.7
                new_strategy.epochs += 1
                new_strategy.changes_log.append(
                    f"Score plateau detected (delta={delta:.3f}). "
                    f"Reducing LR to {new_strategy.learning_rate:.2e}, "
                    f"increasing epochs to {new_strategy.epochs}"
                )

        # --- Rule 4: Check for regression ---
        if len(track_history) >= 2:
            prev_score = track_history[-2].overall_avg
            if overall < prev_score - 0.2:
                # Regression — revert LR and try different approach
                new_strategy.learning_rate = current_strategy.learning_rate * 1.5
                new_strategy.add_dpo_pairs = True
                new_strategy.dpo_count = 400
                new_strategy.changes_log.append(
                    f"Score REGRESSED ({prev_score:.2f} -> {overall:.2f}). "
                    f"Increasing LR to {new_strategy.learning_rate:.2e} and "
                    f"adding 400 DPO pairs"
                )

        # --- Rule 5: Stuck dimensions (below 3.0 for 3+ rounds) ---
        stuck_dims = self._find_stuck_dimensions(track, threshold=3.0, min_rounds=3)
        if stuck_dims:
            new_strategy.extra_sft_count += 300 * len(stuck_dims)
            new_strategy.learning_rate *= 0.5
            new_strategy.changes_log.append(
                f"Dimensions stuck below 3.0 for 3+ rounds: {stuck_dims}. "
                f"Halving LR, adding {300 * len(stuck_dims)} extra examples"
            )

        # --- Rule 6: Major strategy shift after 5 rounds ---
        if iteration >= 5 and iteration % 5 == 0:
            new_strategy = self._major_shift(new_strategy, track_history)

        # Clamp learning rate to reasonable bounds
        new_strategy.learning_rate = max(1e-6, min(1e-3, new_strategy.learning_rate))

        for change in new_strategy.changes_log:
            logger.info("Strategy evolution [%s, iter %d]: %s", track, iteration, change)

        return new_strategy

    def _find_stuck_dimensions(
        self, track: str, threshold: float, min_rounds: int
    ) -> List[str]:
        """Find dimensions that have been below threshold for min_rounds."""
        track_history = [r for r in self.history if r.track == track]
        if len(track_history) < min_rounds:
            return []

        recent = track_history[-min_rounds:]
        all_dims = set()
        for r in recent:
            all_dims.update(r.scores.keys())

        stuck = []
        for dim in all_dims:
            dim_scores = [r.scores.get(dim, 0.0) for r in recent]
            if all(s < threshold for s in dim_scores):
                stuck.append(dim)
        return stuck

    def _major_shift(
        self, strategy: TrainingStrategy, history: List[IterationRecord]
    ) -> TrainingStrategy:
        """Apply a major strategy shift after 5+ rounds."""
        strategy.lora_r = min(128, strategy.lora_r * 2)
        strategy.lora_alpha = strategy.lora_r * 2
        strategy.extra_sft_count *= 2
        strategy.epochs = min(8, strategy.epochs + 2)
        strategy.gradient_accumulation_steps = min(16, strategy.gradient_accumulation_steps + 4)
        strategy.add_dpo_pairs = True
        strategy.dpo_count = max(strategy.dpo_count, 500)

        strategy.changes_log.append(
            f"MAJOR SHIFT: LoRA rank → {strategy.lora_r}, "
            f"epochs → {strategy.epochs}, "
            f"doubled extra data, "
            f"DPO pairs → {strategy.dpo_count}"
        )
        return strategy

    def get_summary(self, track: str) -> Dict[str, Any]:
        """Return a summary of the evolution history for a track."""
        track_history = [r for r in self.history if r.track == track]
        if not track_history:
            return {"track": track, "iterations": 0}

        return {
            "track": track,
            "iterations": len(track_history),
            "scores": [
                {"iteration": r.iteration, "overall": r.overall_avg, "dims": r.scores}
                for r in track_history
            ],
            "best_score": max(r.overall_avg for r in track_history),
            "best_iteration": max(track_history, key=lambda r: r.overall_avg).iteration,
            "strategy_versions": max(r.strategy_version for r in track_history),
        }
