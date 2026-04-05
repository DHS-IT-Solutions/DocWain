"""Curriculum training orchestrator for DocWain V2.

Manages a generate -> train -> eval -> analyze loop driven by
subagent dispatch via a file-based request/response protocol.

Usage::

    python -m src.finetune.v2.curriculum_trainer
    python -m src.finetune.v2.curriculum_trainer --resume
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASES = ["generate", "train", "eval", "analyze"]
MAX_ITERATIONS = 10
BASICS_ESCALATION_ITER = 5
ARTIFACTS_DIR = Path("finetune_artifacts/v2_curriculum")
JUDGE_BATCH_SIZE = 10

SUBAGENT_REQUEST_DIR = ARTIFACTS_DIR / "requests"
SUBAGENT_RESPONSE_DIR = ARTIFACTS_DIR / "responses"
SUBAGENT_POLL_INTERVAL = 30   # seconds
SUBAGENT_TIMEOUT = 3600       # 1 hour


# ---------------------------------------------------------------------------
# PipelineState
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """Serialisable snapshot of pipeline progress."""

    iteration: int = 0
    phase: str = "generate"
    basics_passed: bool = False
    production_passed: bool = False
    dataset_sizes: Dict[str, int] = field(default_factory=dict)
    eval_history: List[Dict[str, Any]] = field(default_factory=list)
    failure_analyses: List[Dict[str, Any]] = field(default_factory=list)
    best_checkpoint: Optional[str] = None
    best_score: float = 0.0
    start_time: Optional[float] = None
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "phase": self.phase,
            "basics_passed": self.basics_passed,
            "production_passed": self.production_passed,
            "dataset_sizes": self.dataset_sizes,
            "eval_history": self.eval_history,
            "failure_analyses": self.failure_analyses,
            "best_checkpoint": self.best_checkpoint,
            "best_score": self.best_score,
            "start_time": self.start_time,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineState":
        return cls(
            iteration=d.get("iteration", 0),
            phase=d.get("phase", "generate"),
            basics_passed=d.get("basics_passed", False),
            production_passed=d.get("production_passed", False),
            dataset_sizes=d.get("dataset_sizes", {}),
            eval_history=d.get("eval_history", []),
            failure_analyses=d.get("failure_analyses", []),
            best_checkpoint=d.get("best_checkpoint"),
            best_score=d.get("best_score", 0.0),
            start_time=d.get("start_time"),
            last_updated=d.get("last_updated"),
        )


def save_state(state: PipelineState, path: Path) -> None:
    """Persist pipeline state to a JSON file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.last_updated = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug("State saved to %s (iter=%d, phase=%s)", path, state.iteration, state.phase)


def load_state(path: Path) -> PipelineState:
    """Load pipeline state from a JSON file.  Returns a fresh state if the file is missing."""
    path = Path(path)
    if not path.exists():
        logger.info("No state file found at %s — starting fresh", path)
        return PipelineState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = PipelineState.from_dict(raw)
        logger.info("Resumed state: iter=%d, phase=%s", state.iteration, state.phase)
        return state
    except Exception as exc:
        logger.warning("Failed to load state from %s (%s) — starting fresh", path, exc)
        return PipelineState()


# ---------------------------------------------------------------------------
# Subagent dispatch protocol (file-based request/response)
# ---------------------------------------------------------------------------


def write_subagent_request(
    request_type: str,
    iteration: int,
    area: str,
    payload: Dict[str, Any],
) -> Path:
    """Write a JSON request file for a subagent to pick up.

    The request file is written to SUBAGENT_REQUEST_DIR with the naming
    convention ``{request_type}_iter{iteration}_{area}.json``.

    Returns the path to the written request file.
    """
    SUBAGENT_REQUEST_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{request_type}_iter{iteration}_{area}.json"
    request_path = SUBAGENT_REQUEST_DIR / filename
    request_data = {
        "type": request_type,
        "iteration": iteration,
        "area": area,
        "payload": payload,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    request_path.write_text(json.dumps(request_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote subagent request: %s", request_path)
    return request_path


def read_subagent_response(
    request_type: str,
    iteration: int,
    area: str,
) -> Optional[str]:
    """Check whether a subagent has written a response for the given request.

    Response files are expected at SUBAGENT_RESPONSE_DIR with naming
    ``{request_type}_iter{iteration}_{area}.jsonl`` or ``.json``.

    Returns the raw response text if found, or None if not yet available.
    """
    for ext in (".jsonl", ".json", ".txt"):
        response_path = SUBAGENT_RESPONSE_DIR / f"{request_type}_iter{iteration}_{area}{ext}"
        if response_path.exists():
            logger.info("Found subagent response: %s", response_path)
            return response_path.read_text(encoding="utf-8")
    return None


def wait_for_responses(
    request_type: str,
    iteration: int,
    areas: List[str],
    timeout: int = SUBAGENT_TIMEOUT,
) -> Dict[str, str]:
    """Poll until all areas have responses or timeout is reached.

    Returns a dict mapping area -> response text for all areas that responded.
    Areas that timed out are absent from the returned dict.
    """
    responses: Dict[str, str] = {}
    pending = list(areas)
    deadline = time.monotonic() + timeout
    last_log = time.monotonic()

    while pending and time.monotonic() < deadline:
        still_pending: List[str] = []
        for area in pending:
            text = read_subagent_response(request_type, iteration, area)
            if text is not None:
                responses[area] = text
                logger.info("Response received for %s/%s (iter %d)", request_type, area, iteration)
            else:
                still_pending.append(area)

        pending = still_pending

        if pending:
            # Log progress at most once per minute
            now = time.monotonic()
            if now - last_log >= 60:
                logger.info(
                    "Still waiting for %d/%d %s responses (iter %d): %s",
                    len(pending), len(areas), request_type, iteration, pending,
                )
                last_log = now
            if time.monotonic() < deadline:
                time.sleep(SUBAGENT_POLL_INTERVAL)

    if pending:
        logger.warning(
            "Timed out waiting for %s responses (iter %d). Missing: %s",
            request_type, iteration, pending,
        )

    return responses


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def phase_generate(state: PipelineState) -> None:
    """Generate training data and write it to the dataset directory.

    Iteration 1: builds initial briefs covering all areas.
    Later iterations: builds augmentation briefs from the latest failure analysis.
    """
    from src.finetune.v2.curriculum_generator import (
        build_initial_briefs,
        build_augmentation_briefs,
        parse_generated_examples,
    )

    dataset_dir = ARTIFACTS_DIR / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if state.iteration == 1:
        briefs = build_initial_briefs()
        output_key = "iter_1_base"
        output_file = dataset_dir / "iter_1_base.jsonl"
    else:
        latest_analysis = state.failure_analyses[-1] if state.failure_analyses else {}
        briefs = build_augmentation_briefs(latest_analysis, state.iteration)
        output_key = f"iter_{state.iteration}_augment"
        output_file = dataset_dir / f"iter_{state.iteration}_augment.jsonl"

    if not briefs:
        logger.warning("No generation briefs produced for iteration %d", state.iteration)
        return

    # Write subagent requests
    areas = [brief.area for brief in briefs]
    for brief in briefs:
        write_subagent_request(
            request_type="generate",
            iteration=state.iteration,
            area=brief.area,
            payload={"prompt": brief.to_prompt(), "count": brief.count},
        )

    logger.info(
        "Dispatched %d generation requests for iter %d, waiting for responses...",
        len(briefs), state.iteration,
    )

    # Wait for all responses
    responses = wait_for_responses("generate", state.iteration, areas)

    # Parse and write examples
    all_examples = []
    for area in areas:
        raw = responses.get(area, "")
        if not raw:
            logger.warning("No generation response for area %s (iter %d)", area, state.iteration)
            continue
        examples = parse_generated_examples(raw)
        logger.info("Parsed %d valid examples from %s (iter %d)", len(examples), area, state.iteration)
        all_examples.extend(examples)

    # Write merged output file for this iteration
    with open(output_file, "w", encoding="utf-8") as fh:
        for ex in all_examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    state.dataset_sizes[output_key] = len(all_examples)
    logger.info(
        "Generation complete: %d examples written to %s", len(all_examples), output_file
    )


def phase_train(state: PipelineState) -> Optional[str]:
    """Merge datasets and train the curriculum model.

    Returns the path to the merged checkpoint directory, or None on failure.
    """
    from src.finetune.v2.curriculum_generator import merge_datasets
    from src.finetune.v2.train_track import TrackTrainingConfig, train_track

    dataset_dir = ARTIFACTS_DIR / "dataset"
    merged_path = dataset_dir / f"iter_{state.iteration}_merged.jsonl"

    # Collect all dataset files up to this iteration
    source_files: List[Path] = []
    for candidate in sorted(dataset_dir.glob("iter_*.jsonl")):
        # Skip merged files from previous iterations in the source list
        if "_merged" not in candidate.name:
            source_files.append(candidate)

    if not source_files:
        logger.error("No source dataset files found in %s for training", dataset_dir)
        return None

    n_merged = merge_datasets(source_files, merged_path)
    logger.info("Merged %d examples into %s", n_merged, merged_path)

    output_dir = ARTIFACTS_DIR / "checkpoints" / f"iter_{state.iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = 3 if state.iteration == 1 else 2

    config = TrackTrainingConfig(
        track_name="curriculum",
        data_path=str(merged_path),
        output_dir=str(output_dir),
        lora_dropout=0.05,
        curriculum_sampling=True,
        skip_ollama_export=True,
        epochs=epochs,
    )

    try:
        checkpoint = train_track(config)
        logger.info("Training complete. Checkpoint: %s", checkpoint)
        return checkpoint
    except Exception as exc:
        logger.error("Training failed for iter %d: %s", state.iteration, exc, exc_info=True)
        return None


def phase_eval(state: PipelineState, checkpoint_path: str) -> List[Dict[str, Any]]:
    """Run LoRA inference and judge the outputs.

    Returns a list of scored example dicts with track/category/difficulty metadata.
    """
    from src.finetune.v2.curriculum_evaluator import (
        JudgingBrief,
        run_lora_inference,
        parse_judge_scores,
    )
    from src.finetune.v2.eval.test_bank import get_test_bank

    eval_dir = ARTIFACTS_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_bank = get_test_bank()
    prompts = [ex["prompt"] for ex in test_bank]

    logger.info(
        "Running LoRA inference on %d test prompts (iter %d)...", len(prompts), state.iteration
    )

    base_model = "unsloth/Qwen3-14B-bnb-4bit"
    responses = run_lora_inference(base_model, checkpoint_path, prompts)

    # Save raw responses
    raw_responses_path = eval_dir / f"iter_{state.iteration}_responses.json"
    with open(raw_responses_path, "w", encoding="utf-8") as fh:
        json.dump(
            [{"prompt": p, "response": r} for p, r in zip(prompts, responses)],
            fh, indent=2, ensure_ascii=False,
        )

    # Attach metadata to each scored item
    scored_examples: List[Dict[str, Any]] = []
    for ex, response in zip(test_bank, responses):
        scored_examples.append({
            "track": ex["track"],
            "category": ex["category"],
            "difficulty": ex["difficulty"],
            "prompt": ex["prompt"],
            "response": response,
            "reference": ex["reference"],
        })

    # Build judging brief batches and dispatch subagent requests
    batches: List[List[Dict[str, Any]]] = []
    for i in range(0, len(scored_examples), JUDGE_BATCH_SIZE):
        batches.append(scored_examples[i : i + JUDGE_BATCH_SIZE])

    batch_areas = [f"batch{i}" for i in range(len(batches))]
    for i, batch in enumerate(batches):
        brief = JudgingBrief(examples=batch, batch_index=i)
        write_subagent_request(
            request_type="judge",
            iteration=state.iteration,
            area=f"batch{i}",
            payload={"prompt": brief.to_prompt(), "batch_index": i},
        )

    logger.info(
        "Dispatched %d judge requests (iter %d), waiting for responses...",
        len(batches), state.iteration,
    )

    judge_responses = wait_for_responses("judge", state.iteration, batch_areas)

    # Parse all judge scores and attach metadata
    all_scores: List[Dict[str, Any]] = []
    for i, batch in enumerate(batches):
        raw_judge = judge_responses.get(f"batch{i}", "")
        if not raw_judge:
            logger.warning("No judge response for batch %d (iter %d)", i, state.iteration)
            continue
        batch_scores = parse_judge_scores(raw_judge)
        for score_entry in batch_scores:
            idx = score_entry.get("example_index", 0)
            if 0 <= idx < len(batch):
                meta = batch[idx]
                all_scores.append({
                    "track": meta["track"],
                    "category": meta["category"],
                    "difficulty": meta["difficulty"],
                    "prompt": meta["prompt"],
                    "response": meta["response"],
                    "scores": score_entry.get("scores", {}),
                })

    # Save all scores
    results_path = eval_dir / f"iter_{state.iteration}_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(all_scores, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Eval complete: %d scored examples saved to %s", len(all_scores), results_path
    )
    return all_scores


def phase_analyze(state: PipelineState, all_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build failure analysis from eval scores and append to state.

    Returns the analysis dict.
    """
    from src.finetune.v2.curriculum_evaluator import build_failure_analysis

    analysis = build_failure_analysis(all_scores)

    eval_dir = ARTIFACTS_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = eval_dir / f"iter_{state.iteration}_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2, ensure_ascii=False)

    state.failure_analyses.append(analysis)
    logger.info(
        "Failure analysis saved to %s — %d weak areas identified",
        analysis_path, len(analysis.get("weak_areas", [])),
    )
    return analysis


# ---------------------------------------------------------------------------
# Gate evaluation helper
# ---------------------------------------------------------------------------


def _check_gates_and_update_state(state: PipelineState, all_scores: List[Dict[str, Any]]) -> None:
    """Aggregate scores, check gates, and update state flags."""
    from src.finetune.v2.curriculum_evaluator import aggregate_scores, check_gates

    aggregated = aggregate_scores(all_scores)
    gate_result = check_gates(aggregated)

    overall_avg = aggregated.get("overall_avg", 0.0)
    logger.info(
        "Gate check iter %d: overall_avg=%.4f, basics=%s, production=%s",
        state.iteration, overall_avg, gate_result.basics_passed, gate_result.production_passed,
    )

    state.basics_passed = gate_result.basics_passed or state.basics_passed
    if gate_result.production_passed:
        state.production_passed = True

    if float(overall_avg) > state.best_score:
        state.best_score = float(overall_avg)

    state.eval_history.append({
        "iteration": state.iteration,
        "overall_avg": overall_avg,
        "min_dimension": aggregated.get("min_dimension", 0.0),
        "basics_passed": gate_result.basics_passed,
        "production_passed": gate_result.production_passed,
    })


# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------


def run_pipeline(resume: bool = False) -> None:
    """Run the full curriculum training loop.

    Iterates through generate -> train -> eval -> analyze phases until
    the production gate passes or MAX_ITERATIONS is reached.

    Parameters
    ----------
    resume:
        If True, load existing state from disk and continue from the last
        saved phase.  If False, always start from the beginning.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    state_path = ARTIFACTS_DIR / "pipeline_state.json"

    if resume:
        state = load_state(state_path)
    else:
        state = PipelineState()
        state.start_time = time.time()

    if state.start_time is None:
        state.start_time = time.time()

    logger.info(
        "Starting curriculum pipeline — resume=%s, iter=%d, phase=%s",
        resume, state.iteration, state.phase,
    )

    while not state.production_passed and state.iteration < MAX_ITERATIONS:
        # Determine current phase index
        try:
            phase_idx = PHASES.index(state.phase)
        except ValueError:
            logger.warning("Unknown phase '%s', resetting to 'generate'", state.phase)
            state.phase = "generate"
            phase_idx = 0

        # ---- generate ----
        if state.phase == "generate":
            state.iteration += 1
            logger.info("=== Iteration %d — generate ===", state.iteration)
            phase_generate(state)
            state.phase = "train"
            save_state(state, state_path)

        # ---- train ----
        if state.phase == "train":
            logger.info("=== Iteration %d — train ===", state.iteration)
            checkpoint_path = phase_train(state)
            if checkpoint_path:
                if state.best_checkpoint is None:
                    state.best_checkpoint = checkpoint_path
            state.phase = "eval"
            save_state(state, state_path)

        # ---- eval ----
        if state.phase == "eval":
            if state.best_checkpoint is None:
                logger.error("No checkpoint available for eval at iter %d, skipping", state.iteration)
                state.phase = "analyze"
                save_state(state, state_path)
            else:
                logger.info("=== Iteration %d — eval ===", state.iteration)
                all_scores = phase_eval(state, state.best_checkpoint)
                # Update best checkpoint if this eval is better
                if all_scores:
                    from src.finetune.v2.curriculum_evaluator import aggregate_scores
                    agg = aggregate_scores(all_scores)
                    overall = float(agg.get("overall_avg", 0.0))
                    if overall > state.best_score:
                        # Resolve current checkpoint from latest training output
                        latest_ckpt = str(
                            ARTIFACTS_DIR / "checkpoints" / f"iter_{state.iteration}" / "merged_16bit"
                        )
                        state.best_checkpoint = latest_ckpt
                    _check_gates_and_update_state(state, all_scores)
                state.phase = "analyze"
                save_state(state, state_path)
                # Store for the analyze phase
                _last_scores = all_scores

        # ---- analyze ----
        if state.phase == "analyze":
            logger.info("=== Iteration %d — analyze ===", state.iteration)
            # Retrieve last scores from eval (or reload from disk)
            try:
                last_scores = _last_scores  # type: ignore[name-defined]
            except NameError:
                last_scores = _reload_scores_from_disk(state.iteration)

            phase_analyze(state, last_scores)
            state.phase = "generate"
            save_state(state, state_path)

            # Log iteration summary
            if state.eval_history:
                last_eval = state.eval_history[-1]
                logger.info(
                    "Iteration %d complete — avg=%.4f basics=%s production=%s",
                    state.iteration,
                    last_eval.get("overall_avg", 0.0),
                    last_eval.get("basics_passed"),
                    last_eval.get("production_passed"),
                )

            # Check escalation: if basics not passed by BASICS_ESCALATION_ITER, warn
            if state.iteration >= BASICS_ESCALATION_ITER and not state.basics_passed:
                logger.warning(
                    "Basics gate not passed after %d iterations — consider reviewing "
                    "data quality or training configuration.",
                    state.iteration,
                )

            if state.production_passed:
                break

    # Final summary
    if state.production_passed:
        logger.info(
            "Production gate passed at iteration %d! Best score: %.4f. Checkpoint: %s",
            state.iteration, state.best_score, state.best_checkpoint,
        )
    else:
        logger.info(
            "Max iterations (%d) reached. Best score: %.4f. Basics passed: %s.",
            MAX_ITERATIONS, state.best_score, state.basics_passed,
        )

    save_state(state, state_path)


def _reload_scores_from_disk(iteration: int) -> List[Dict[str, Any]]:
    """Fall back to loading eval results from disk if in-memory scores are lost."""
    results_path = ARTIFACTS_DIR / "eval" / f"iter_{iteration}_results.json"
    if results_path.exists():
        try:
            return json.loads(results_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to reload scores from %s: %s", results_path, exc)
    return []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    resume = "--resume" in sys.argv
    run_pipeline(resume=resume)
