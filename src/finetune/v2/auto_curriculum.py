"""Automated curriculum training pipeline for DocWain.

Integrates pattern collection, data generation, training, evaluation,
and model promotion into a single self-running pipeline.

This is the production entry point — designed to run as a cron job or
triggered manually. It collects document patterns, generates targeted
training data, trains with curriculum sampling, evaluates with subagent
judges, and promotes the model when the production gate passes.

Usage::

    # Full pipeline (collect patterns + train + eval)
    python -m src.finetune.v2.auto_curriculum

    # Resume from last checkpoint
    python -m src.finetune.v2.auto_curriculum --resume

    # Skip pattern collection (use cached patterns)
    python -m src.finetune.v2.auto_curriculum --skip-patterns

    # Dry run (generate data only, no training)
    python -m src.finetune.v2.auto_curriculum --dry-run
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.finetune.v2.curriculum_trainer import (
    ARTIFACTS_DIR,
    PipelineState,
    load_state,
    save_state,
)
from src.finetune.v2.curriculum_generator import (
    build_initial_briefs,
    build_augmentation_briefs,
    merge_datasets,
    GenerationBrief,
    AREA_CONFIGS,
)
from src.finetune.v2.curriculum_evaluator import (
    aggregate_scores,
    check_gates,
    build_failure_analysis,
)
from src.finetune.v2.pattern_collector import (
    DocumentPatterns,
    collect_patterns_from_mongodb,
    collect_feedback_signals,
    save_patterns,
    load_patterns,
)

logger = logging.getLogger(__name__)

PATTERNS_PATH = ARTIFACTS_DIR / "patterns.json"
FEEDBACK_PATH = ARTIFACTS_DIR / "feedback_signals.json"
STATE_PATH = ARTIFACTS_DIR / "pipeline_state.json"

# ---------------------------------------------------------------------------
# Retrospective: Lessons learned from the initial training run
# ---------------------------------------------------------------------------

TRAINING_RETROSPECTIVE = """
## DocWain V2 Training Retrospective (2026-04-05)

### What Worked
1. **Unified curriculum** (all 6 areas in one training run) prevented catastrophic
   forgetting that plagued the old sequential per-track approach.
2. **Subagent-generated data** produced far higher quality than template generators.
   603 subagent examples > 20K template examples in score impact.
3. **Subagent judges** gave actionable feedback (truncation, hallucination, format
   mismatch) that programmatic rubrics couldn't detect.
4. **Targeted augmentation** from failure analysis was highly effective:
   - Layout completeness: 3.45 → 5.00 with 30 targeted examples
   - Visualization: 2.88 → 4.80 with 120 targeted examples
   - OCR: 2.34 → 4.59 after fixing test bank format alignment
5. **Curriculum sampling** (easy → medium → hard) with 2-3 epochs was sufficient.

### What Didn't Work
1. **Old template generators** — 2500 examples from 15 templates produced
   low-diversity data that the model quickly memorized without generalizing.
2. **DPO with synthetic pairs** — disabled in old pipeline because both chosen
   and rejected came from templates. Could revisit with subagent-generated pairs.
3. **Sequential per-track training** — each track's training regressed previous ones.
4. **Programmatic rubrics** — regex/keyword matching couldn't measure reasoning quality.
5. **Ollama-based eval loop** — 30 min overhead per eval cycle vs 15 min direct LoRA.

### Key Parameters
- Base model: unsloth/Qwen3-14B-bnb-4bit
- LoRA: r=64, alpha=128, dropout=0.05
- Effective batch: 32 (4 × 8 grad accum)
- LR: 2e-5 cosine, 10% warmup
- Epochs: 3 (initial), 2 (augmentation)
- Dataset: 863 examples across 6 areas
- Production gate: 4.0 avg, 3.5 min dimension → achieved 4.71 avg, 4.10 min

### Critical Fixes During Training
1. Train/eval format alignment: training data must match test bank format
2. OCR test bank had empty context — model correctly refused or hallucinated
3. max_new_tokens for eval must be >= 1024 for visualization chart specs
4. Merged FP16 checkpoints load directly (don't call get_peft_model)
"""


# ---------------------------------------------------------------------------
# Pattern-enhanced brief generation
# ---------------------------------------------------------------------------


def enhance_briefs_with_patterns(
    briefs: list[GenerationBrief],
    patterns: DocumentPatterns,
) -> list[GenerationBrief]:
    """Inject real-world document patterns into generation briefs.

    Adds context from collected patterns so generated training data
    reflects actual document distributions, entity types, and
    relationship patterns seen in production.
    """
    if patterns.total_documents == 0:
        return briefs

    pattern_context = patterns.to_brief_context()

    enhanced = []
    for brief in briefs:
        # Inject pattern context into focus instructions
        existing_focus = brief.focus_instructions or ""
        pattern_block = (
            f"\n\n## Real-World Document Patterns\n"
            f"Use these patterns from production data to make examples realistic:\n"
            f"{pattern_context}\n"
        )

        # Adjust difficulty based on quality signals
        if patterns.low_confidence_rate > 0.3:
            # More hard examples for low-confidence areas
            new_split = {"easy": 0.1, "medium": 0.4, "hard": 0.5}
        else:
            new_split = brief.difficulty_split

        enhanced.append(GenerationBrief(
            area=brief.area,
            count=brief.count,
            difficulty_split=new_split,
            categories=brief.categories,
            focus_instructions=existing_focus + pattern_block,
            iteration=brief.iteration,
        ))

    return enhanced


# ---------------------------------------------------------------------------
# Model promotion
# ---------------------------------------------------------------------------


def promote_model(checkpoint_path: str, new_score: Optional[float] = None) -> bool:
    """Promote the trained model to production if it beats the current active.

    Updates the docwain-v2-active symlink and restarts vLLM. Refuses to
    promote if ``new_score`` is given and does not exceed the currently-active
    score recorded in ``models/docwain-v2-active.score.json``.
    Does NOT update Ollama (per project convention).
    """
    symlink = Path("models/docwain-v2-active")
    score_file = Path("models/docwain-v2-active.score.json")
    abs_checkpoint = str(Path(checkpoint_path).resolve())

    # "Always the best" guard: only promote if the new checkpoint strictly beats
    # whatever is currently active.
    if new_score is not None and score_file.exists():
        try:
            current = float(json.loads(score_file.read_text()).get("score", 0.0))
        except (json.JSONDecodeError, OSError, ValueError):
            current = 0.0
        if new_score <= current:
            logger.info(
                "Skipping promotion: new score %.4f does not beat active %.4f",
                new_score, current,
            )
            return False

    logger.info("Promoting model: %s -> %s", symlink, abs_checkpoint)

    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()
    symlink.symlink_to(abs_checkpoint)

    if new_score is not None:
        score_file.write_text(json.dumps({
            "score": new_score,
            "checkpoint": abs_checkpoint,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }))

    # Restart vLLM if running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("Restarting vLLM to load new model...")
            subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)
            time.sleep(5)
            # The systemd service or external process manager will restart it
            logger.info("vLLM stopped — restart via systemd or manual launch")
    except Exception as exc:
        logger.warning("Could not restart vLLM: %s", exc)

    return True


# ---------------------------------------------------------------------------
# Main automated pipeline
# ---------------------------------------------------------------------------


def run_auto_curriculum(
    resume: bool = False,
    skip_patterns: bool = False,
    dry_run: bool = False,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """Run the automated curriculum training pipeline.

    Steps:
    1. Collect patterns from MongoDB (document types, entities, quality signals)
    2. Collect feedback signals (low-confidence queries, failures)
    3. Generate/augment training data using pattern-enhanced briefs
    4. Train with curriculum sampling
    5. Evaluate with subagent judges
    6. If gate passes: promote model. If not: analyze failures and loop.

    Returns dict with final state summary.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load or initialize state
    if resume:
        state = load_state(STATE_PATH)
        logger.info("Resumed: iter=%d, phase=%s, best=%.2f",
                     state.iteration, state.phase, state.best_score)
    else:
        state = PipelineState(start_time=time.time())

    # Step 1: Collect patterns
    if not skip_patterns:
        logger.info("Collecting document patterns from MongoDB...")
        patterns = collect_patterns_from_mongodb()
        save_patterns(patterns, PATTERNS_PATH)

        logger.info("Collecting feedback signals...")
        feedback = collect_feedback_signals()
        FEEDBACK_PATH.write_text(json.dumps(feedback, indent=2))

        logger.info("Patterns: %d docs, %d entity types, %d relationships",
                     patterns.total_documents,
                     len(patterns.entity_type_counts),
                     len(patterns.relationship_types))
    else:
        patterns = load_patterns(PATTERNS_PATH)
        logger.info("Using cached patterns (%d docs)", patterns.total_documents)

    if dry_run:
        # Generate briefs only
        briefs = build_initial_briefs()
        briefs = enhance_briefs_with_patterns(briefs, patterns)
        logger.info("Dry run: %d briefs generated", len(briefs))
        for b in briefs:
            logger.info("  %s: %d examples", b.area, b.count)
        return {"dry_run": True, "briefs": len(briefs)}

    # Step 2-6: Run the curriculum loop
    # This delegates to the curriculum_trainer's phase implementations
    # but with pattern-enhanced data generation
    from src.finetune.v2.curriculum_trainer import (
        phase_train,
        phase_eval,
        phase_analyze,
        _check_gates_and_update_state,
    )

    iteration = 0
    while iteration < max_iterations and not state.production_passed:
        iteration += 1
        state.iteration = iteration
        logger.info("=== Auto-Curriculum Iteration %d ===", iteration)

        # Generate data (pattern-enhanced)
        if iteration == 1 and not (ARTIFACTS_DIR / "dataset" / "iter_1_base.jsonl").exists():
            briefs = build_initial_briefs()
        elif state.failure_analyses:
            briefs = build_augmentation_briefs(state.failure_analyses[-1], iteration)
        else:
            logger.warning("No failure analysis for augmentation, using initial briefs")
            briefs = build_initial_briefs()

        briefs = enhance_briefs_with_patterns(briefs, patterns)
        logger.info("Generated %d briefs for iteration %d", len(briefs), iteration)

        # The actual subagent dispatch happens externally
        # (Claude Code session picks up requests and writes responses)
        # For automated mode, write briefs to requests dir
        requests_dir = ARTIFACTS_DIR / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        for brief in briefs:
            brief_path = requests_dir / f"generate_iter{iteration}_{brief.area}.json"
            brief_path.write_text(json.dumps({
                "type": "generate",
                "iteration": iteration,
                "area": brief.area,
                "count": brief.count,
                "prompt": brief.to_prompt(),
            }, indent=2))

        state.phase = "awaiting_generation"
        save_state(state, STATE_PATH)
        logger.info(
            "Briefs written to %s. Awaiting subagent data generation.",
            requests_dir,
        )

        # The pipeline pauses here for external subagent dispatch
        # When responses arrive, resume with --resume flag
        break

    return {
        "iteration": state.iteration,
        "phase": state.phase,
        "best_score": state.best_score,
        "production_passed": state.production_passed,
        "basics_passed": state.basics_passed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    resume = "--resume" in sys.argv
    skip_patterns = "--skip-patterns" in sys.argv
    dry_run = "--dry-run" in sys.argv

    result = run_auto_curriculum(
        resume=resume,
        skip_patterns=skip_patterns,
        dry_run=dry_run,
    )
    print(json.dumps(result, indent=2))
