"""Strategy evolver for the DocWain V2 iterative training loop.

Analyses evaluation results across iterations and evolves the training
strategy to target persistent weaknesses.  No fixed iteration cap -- the
strategy keeps adapting until quality gates pass.

Usage::

    from src.finetune.v2.strategy_evolver import StrategyEvolver, TrainingStrategy

    strategy = TrainingStrategy(track="excel_csv")
    evolver = StrategyEvolver()

    # After each training + eval round:
    new_strategy = evolver.evolve(
        track="excel_csv",
        iteration=1,
        eval_result=evaluator.evaluate_track("excel_csv"),
        current_strategy=strategy,
    )
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate thresholds (must stay in sync with evaluator)
# ---------------------------------------------------------------------------

_GATE_THRESHOLDS: Dict[str, float] = {
    "excel_csv": 4.0,
    "layout": 4.0,
    "ocr_vision": 4.0,
    "reasoning": 4.0,
    "kg": 3.8,
    "visualization": 4.0,
}

# ---------------------------------------------------------------------------
# Dimension -> category mapping (used for targeted data generation)
# ---------------------------------------------------------------------------

_DIM_TO_CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "excel_csv": {
        "tabular_qa_accuracy": ["single_sheet_lookup", "conditional_filtering"],
        "cross_sheet_reasoning": ["multi_sheet_reasoning"],
        "data_type_correctness": ["data_type_handling"],
        "aggregation_accuracy": ["aggregation", "formula_interpretation"],
    },
    "layout": {
        "structure_accuracy": ["field_extraction", "nested_structure"],
        "relationship_extraction": ["multi_column_layout", "nested_structure"],
        "noise_robustness": ["noisy_document"],
        "completeness_score": ["field_extraction", "header_footer", "table_in_layout"],
    },
    "ocr_vision": {
        "printed_accuracy": ["printed_text_extraction"],
        "handwriting_accuracy": ["handwriting_recognition"],
        "diagram_understanding": ["diagram_understanding"],
        "image_table_reconstruction": ["table_from_image"],
        "overlay_handling": ["overlay_text", "mixed_content"],
    },
    "reasoning": {
        "reasoning_depth": ["causal_reasoning", "multi_document_synthesis"],
        "evidence_grounding": ["evidence_based_conclusion", "multi_document_synthesis"],
        "synthesis_coherence": ["comparative_analysis", "contradiction_detection"],
    },
    "kg": {
        "entity_usage": ["entity_lookup", "entity_disambiguation"],
        "relationship_reasoning": ["relationship_reasoning", "multi_hop_query"],
        "citation_accuracy": ["cross_document_kg", "multi_hop_query"],
    },
    "visualization": {
        "trigger_judgment": ["no_chart", "flow_analysis"],
        "spec_correctness": ["bar_chart", "line_chart", "specialized_chart"],
        "data_accuracy": ["bar_chart", "line_chart", "pie_donut"],
        "type_selection": ["specialized_chart", "bar_chart"],
    },
}

_ALL_CATEGORIES: Dict[str, List[str]] = {
    "excel_csv": ["single_sheet_lookup", "aggregation", "multi_sheet_reasoning",
                  "data_type_handling", "formula_interpretation", "conditional_filtering"],
    "layout": ["field_extraction", "multi_column_layout", "nested_structure",
               "noisy_document", "header_footer", "table_in_layout"],
    "ocr_vision": ["printed_text_extraction", "handwriting_recognition",
                   "diagram_understanding", "table_from_image",
                   "overlay_text", "mixed_content"],
    "reasoning": ["multi_document_synthesis", "causal_reasoning",
                  "comparative_analysis", "contradiction_detection",
                  "evidence_based_conclusion"],
    "kg": ["entity_lookup", "relationship_reasoning", "multi_hop_query",
           "entity_disambiguation", "cross_document_kg"],
    "visualization": ["bar_chart", "line_chart", "pie_donut",
                      "specialized_chart", "no_chart", "flow_analysis"],
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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
    weight_decay: float = 0.01

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

    def summary(self) -> str:
        """Return a compact human-readable summary."""
        parts = [
            f"v{self.strategy_version}",
            f"lr={self.learning_rate:.1e}",
            f"lora_r={self.lora_r}",
            f"alpha={self.lora_alpha}",
            f"epochs={self.epochs}",
            f"bs={self.batch_size}",
        ]
        if self.extra_sft_count > 0:
            parts.append(f"+sft={self.extra_sft_count}")
        if self.add_dpo_pairs:
            parts.append(f"+dpo={self.dpo_count}")
        if self.curriculum_order:
            parts.append(f"curriculum={len(self.curriculum_order)} stages")
        return " | ".join(parts)


@dataclass
class IterationRecord:
    """Record of a single training iteration's results."""

    iteration: int
    track: str
    scores: Dict[str, float]       # dimension -> score (1.0-5.0)
    overall_avg: float
    passed: bool
    weak_categories: List[str]
    strategy_version: int
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Strategy Evolver
# ---------------------------------------------------------------------------


class StrategyEvolver:
    """Analyses iteration history and evolves training strategy.

    Rules (applied in priority order):

    1. **Already passing** -- If the track passes the gate, apply only
       conservative adjustments for remaining weak categories.
    2. **Regression guard** -- If score dropped vs previous iteration,
       revert LR and add DPO pairs from model failures.
    3. **Plateau breaker** -- If score delta < 0.1 for 2 consecutive
       rounds, increase LoRA rank, shift curriculum, and bump LR.
    4. **Persistent weakness** -- If a dimension is below 3.0 for 3+
       rounds, aggressively target those categories with extra data.
    5. **Major strategy shift** -- After 5 rounds with minimal overall
       improvement, double data volume, restructure curriculum, bump
       LoRA rank, and enable DPO.
    6. **Steady improvement** -- Default path: targeted data for weak
       categories, minor parameter adjustments.
    """

    def __init__(self) -> None:
        self.history: List[IterationRecord] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def evolve(
        self,
        track: str,
        iteration: int,
        eval_result: Dict[str, Any],
        current_strategy: TrainingStrategy,
    ) -> TrainingStrategy:
        """Analyse eval results and return an evolved strategy.

        Parameters
        ----------
        track:
            Track name (e.g. ``"excel_csv"``).
        iteration:
            Current iteration number (1-based).
        eval_result:
            Result dict from :meth:`TrackEvaluator.evaluate_track`.
        current_strategy:
            The strategy used for this iteration.

        Returns
        -------
        A new :class:`TrainingStrategy` with evolved parameters.
        """
        new = deepcopy(current_strategy)
        new.strategy_version += 1
        new.changes_log = []

        scores = eval_result.get("dimensions", {})
        weak_cats = eval_result.get("weak_categories", [])
        overall = eval_result.get("overall_avg", 0.0)
        passed = eval_result.get("passed", False)
        gate = _GATE_THRESHOLDS.get(track, 4.0)

        # Record this iteration
        self.history.append(IterationRecord(
            iteration=iteration,
            track=track,
            scores=dict(scores),
            overall_avg=overall,
            passed=passed,
            weak_categories=list(weak_cats),
            strategy_version=current_strategy.strategy_version,
        ))

        track_history = [r for r in self.history if r.track == track]

        logger.info(
            "Evolving strategy for %s iter=%d avg=%.2f gate=%.1f passed=%s",
            track, iteration, overall, gate, passed,
        )

        # --- Rule 1: Already passing ---
        if passed and iteration >= 2:
            _log_change(new, "Track passing gate; conservative adjustments only")
            if weak_cats:
                new.extra_sft_categories = weak_cats
                new.extra_sft_count = max(50, new.extra_sft_count)
                _log_change(new, f"Minor augmentation for remaining weak: {weak_cats}")
            _clamp_lr(new)
            return new

        # --- Rule 2: Regression guard ---
        if len(track_history) >= 2:
            prev = track_history[-2]
            delta = overall - prev.overall_avg
            if delta < -0.05:
                _log_change(
                    new,
                    f"REGRESSION ({prev.overall_avg:.2f} -> {overall:.2f}). "
                    f"Reverting LR, adding DPO pairs.",
                )
                new.learning_rate = current_strategy.learning_rate
                new.add_dpo_pairs = True
                new.dpo_count = max(new.dpo_count + 100, 200)
                if new.epochs != prev.strategy_version:
                    pass  # don't touch epochs on regression
                _clamp_lr(new)
                return new

        # --- Rule 3: Plateau breaker ---
        if len(track_history) >= 3:
            recent_deltas = [
                track_history[i].overall_avg - track_history[i - 1].overall_avg
                for i in range(len(track_history) - 2, len(track_history))
            ]
            if all(abs(d) < 0.1 for d in recent_deltas):
                _log_change(
                    new,
                    f"PLATEAU (deltas: {[f'{d:.3f}' for d in recent_deltas]}). "
                    f"Increasing LoRA rank, shifting curriculum, bumping LR.",
                )
                new.lora_r = min(new.lora_r * 2, 256)
                new.lora_alpha = new.lora_r * 2
                new.learning_rate = min(new.learning_rate * 1.5, 1e-4)
                _reorder_curriculum(new, scores)
                _clamp_lr(new)
                return new

        # --- Rule 4: Persistent weakness ---
        stuck_dims = self._find_stuck_dimensions(track, threshold=3.0, min_rounds=3)
        if stuck_dims:
            _log_change(
                new,
                f"PERSISTENT WEAKNESS in {stuck_dims}. "
                f"Increasing targeted data, adjusting LR.",
            )
            new.learning_rate = min(new.learning_rate * 1.2, 5e-5)
            cats = _dims_to_categories(track, stuck_dims)
            new.extra_sft_categories = list(set(new.extra_sft_categories) | set(cats) | set(weak_cats))
            new.extra_sft_count = max(new.extra_sft_count + 200, 500)
            _clamp_lr(new)
            return new

        # --- Rule 5: Major strategy shift ---
        if len(track_history) >= 5:
            first_avg = track_history[-5].overall_avg
            if overall - first_avg < 0.2:
                new = self._major_shift(new, track_history, scores, weak_cats, track)
                _clamp_lr(new)
                return new

        # --- Rule 6: Steady improvement (default) ---
        _log_change(new, "Steady improvement path; targeted adjustments.")

        if weak_cats:
            new.extra_sft_categories = weak_cats
            new.extra_sft_count = max(new.extra_sft_count + 100, 200)
            _log_change(new, f"Adding {new.extra_sft_count} SFT examples for {weak_cats}")

        # Add DPO pairs starting from iteration 2
        if iteration >= 2:
            new.add_dpo_pairs = True
            new.dpo_count = max(200, 100 * len(weak_cats))
            _log_change(new, f"Adding {new.dpo_count} DPO pairs from model failures")

        # Target the weakest dimension
        if scores:
            weakest_dim = min(scores, key=scores.get)  # type: ignore[arg-type]
            weakest_score = scores[weakest_dim]
            if weakest_score < 2.5:
                new.extra_sft_count = max(new.extra_sft_count + 200, 400)
                new.add_dpo_pairs = True
                new.dpo_count = max(new.dpo_count + 50, 100)
                _log_change(new, f"Very weak dim {weakest_dim}={weakest_score:.2f}: aggressive augmentation")
            elif weakest_score < 3.5:
                new.extra_sft_count = max(new.extra_sft_count + 100, 200)
                _log_change(new, f"Moderate weakness {weakest_dim}={weakest_score:.2f}: adding data")

        # Decay LR in later iterations for fine-grained learning
        if iteration >= 3 and new.learning_rate > 1e-5:
            new.learning_rate = max(new.learning_rate * 0.8, 1e-5)
            _log_change(new, f"LR decay for iter {iteration}: {new.learning_rate:.1e}")

        _clamp_lr(new)

        for change in new.changes_log:
            logger.info("  [%s iter=%d] %s", track, iteration, change)

        return new

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_summary(self, track: str) -> Dict[str, Any]:
        """Return a summary of the evolution history for a track."""
        track_history = [r for r in self.history if r.track == track]
        if not track_history:
            return {"track": track, "iterations": 0}

        return {
            "track": track,
            "iterations": len(track_history),
            "scores": [
                {
                    "iteration": r.iteration,
                    "overall": r.overall_avg,
                    "dims": r.scores,
                    "passed": r.passed,
                }
                for r in track_history
            ],
            "best_score": max(r.overall_avg for r in track_history),
            "best_iteration": max(track_history, key=lambda r: r.overall_avg).iteration,
            "latest_version": max(r.strategy_version for r in track_history),
        }

    def get_best_iteration(self, track: str) -> Optional[IterationRecord]:
        """Return the iteration with the highest overall average for a track."""
        track_history = [r for r in self.history if r.track == track]
        if not track_history:
            return None
        return max(track_history, key=lambda r: r.overall_avg)

    def get_trend(self, track: str) -> str:
        """Return a trend label for the track.

        Returns one of: ``'improving'``, ``'plateaued'``, ``'regressing'``,
        or ``'insufficient_data'``.
        """
        track_history = [r for r in self.history if r.track == track]
        if len(track_history) < 2:
            return "insufficient_data"

        recent = [r.overall_avg for r in track_history[-3:]]
        deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

        if all(d > 0.05 for d in deltas):
            return "improving"
        elif all(d < -0.05 for d in deltas):
            return "regressing"
        elif all(abs(d) <= 0.1 for d in deltas):
            return "plateaued"
        else:
            return "improving" if sum(deltas) > 0 else "plateaued"

    def get_history_for_track(self, track: str) -> List[IterationRecord]:
        """Return all iteration records for a track."""
        return [r for r in self.history if r.track == track]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_stuck_dimensions(
        self, track: str, threshold: float, min_rounds: int
    ) -> List[str]:
        """Find dimensions below threshold for min_rounds consecutive iterations."""
        track_history = [r for r in self.history if r.track == track]
        if len(track_history) < min_rounds:
            return []

        recent = track_history[-min_rounds:]
        all_dims: set[str] = set()
        for r in recent:
            all_dims.update(r.scores.keys())

        stuck: List[str] = []
        for dim in sorted(all_dims):
            dim_scores = [r.scores.get(dim, 5.0) for r in recent]
            if all(s < threshold for s in dim_scores):
                stuck.append(dim)
        return stuck

    def _major_shift(
        self,
        strategy: TrainingStrategy,
        history: List[IterationRecord],
        scores: Dict[str, float],
        weak_cats: List[str],
        track: str,
    ) -> TrainingStrategy:
        """Apply a major strategy shift after 5+ rounds with minimal improvement."""
        strategy.lora_r = min(strategy.lora_r * 2, 256)
        strategy.lora_alpha = strategy.lora_r * 2
        strategy.extra_sft_count = max(strategy.extra_sft_count * 2, 1000)
        strategy.extra_sft_categories = (
            weak_cats if weak_cats else _ALL_CATEGORIES.get(track, [])
        )
        strategy.epochs = min(strategy.epochs + 2, 8)
        strategy.gradient_accumulation_steps = min(
            strategy.gradient_accumulation_steps + 4, 16
        )
        strategy.add_dpo_pairs = True
        strategy.dpo_count = max(strategy.dpo_count, 500)

        _reorder_curriculum(strategy, scores)

        _log_change(
            strategy,
            f"MAJOR SHIFT: lora_r={strategy.lora_r}, "
            f"epochs={strategy.epochs}, "
            f"sft={strategy.extra_sft_count}, "
            f"dpo={strategy.dpo_count}, "
            f"curriculum reordered",
        )
        return strategy


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _log_change(strategy: TrainingStrategy, msg: str) -> None:
    """Append a change message to the strategy's log."""
    strategy.changes_log.append(msg)


def _clamp_lr(strategy: TrainingStrategy) -> None:
    """Clamp learning rate to [1e-6, 1e-3]."""
    strategy.learning_rate = max(1e-6, min(1e-3, strategy.learning_rate))


def _reorder_curriculum(
    strategy: TrainingStrategy, dims: Dict[str, float]
) -> None:
    """Reorder curriculum to teach weakest-scoring categories first.

    The idea is that the model gets more gradient updates on weak areas
    before learning rate decays.
    """
    if not dims:
        return

    track = strategy.track
    cat_map = _DIM_TO_CATEGORIES.get(track, {})

    # Sort dimensions by score ascending (weakest first)
    ordered_dims = sorted(dims.keys(), key=lambda d: dims.get(d, 5.0))

    new_order: List[str] = []
    seen: set[str] = set()
    for dim in ordered_dims:
        for cat in cat_map.get(dim, []):
            if cat not in seen:
                new_order.append(cat)
                seen.add(cat)

    # Append remaining categories not tied to any weak dimension
    all_cats = _ALL_CATEGORIES.get(track, [])
    for cat in all_cats:
        if cat not in seen:
            new_order.append(cat)
            seen.add(cat)

    strategy.curriculum_order = new_order


def _dims_to_categories(track: str, dims: List[str]) -> List[str]:
    """Map dimension names to training data categories."""
    cat_map = _DIM_TO_CATEGORIES.get(track, {})
    cats: List[str] = []
    for dim in dims:
        cats.extend(cat_map.get(dim, []))
    return cats
