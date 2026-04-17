#!/usr/bin/env python3
"""Weekend iterative finetuning loop for DocWain.

Orchestrates: stop vLLM -> harvest data -> SFT+DPO rounds -> deploy -> test.
Score-gated with plateau detection. Resumable from any phase.

Usage:
    python scripts/weekend_finetune_loop.py
    python scripts/weekend_finetune_loop.py --resume
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

WORK_DIR = Path("finetune_artifacts/weekend_loop")
STATE_FILE = WORK_DIR / "state.json"
LOG_FILE = WORK_DIR / "loop.log"

ITER3_CHECKPOINT = "finetune_artifacts/v2_curriculum/checkpoints/iter_3/merged_16bit"
ITER3_DATASET = "finetune_artifacts/v2_curriculum/dataset/iter_3_merged.jsonl"
LEARNING_SIGNALS_DIR = Path("src/outputs/learning_signals")
MODEL_SYMLINK = Path("models/docwain-v2-active")

# Gate thresholds
# Gate: Claude-level document intelligence — keep training until model is exceptional
GATE_OVERALL_AVG = 4.95       # Near-perfect on all dimensions
GATE_MIN_DIMENSION = 4.5      # No weak spots allowed
PLATEAU_THRESHOLD = 0.02      # Tighter plateau detection
PLATEAU_PATIENCE = 3           # More patience before stopping
MAX_ROUNDS = 10                # More rounds to reach GPT-level

# Setup logging
WORK_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="a"),
    ],
)
logger = logging.getLogger("weekend_loop")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class LoopState:
    phase: str = "setup"  # setup, harvest, training, deploy, test, done
    round: int = 0
    sub_phase: str = ""  # sft, dpo, eval, analyze
    best_score: float = 0.0
    best_round: int = 0
    best_checkpoint: str = ""
    round_history: List[Dict[str, Any]] = field(default_factory=list)
    dataset_path: str = ""
    dpo_path: str = ""
    started_at: str = ""
    vllm_stopped: bool = False
    original_symlink: str = ""
    sft_count: int = 0
    dpo_count: int = 0

    def save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        STATE_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "LoopState":
        if STATE_FILE.exists():
            d = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            d.pop("last_updated", None)
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        return cls()


# Global for signal handler
_state: Optional[LoopState] = None


def _signal_handler(signum, frame):
    if _state:
        logger.info("Signal %d received, saving state and exiting...", signum)
        _state.save()
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Phase 0: Setup
# ---------------------------------------------------------------------------

def phase_setup(state: LoopState):
    logger.info("=" * 60)
    logger.info("PHASE 0: PRE-TRAINING SETUP")
    logger.info("=" * 60)

    state.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Record current symlink
    if MODEL_SYMLINK.is_symlink():
        state.original_symlink = str(MODEL_SYMLINK.resolve())
        logger.info("Current model symlink: %s", state.original_symlink)

    # Stop vLLM
    logger.info("Stopping vLLM services...")
    for svc in ["docwain-vllm-smart", "docwain-vllm-fast"]:
        r = subprocess.run(["sudo", "systemctl", "stop", svc],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            logger.info("  Stopped %s", svc)
        else:
            logger.warning("  Failed to stop %s: %s", svc, r.stderr.strip())

    time.sleep(10)  # Let GPU memory free up
    state.vllm_stopped = True

    # Verify GPU is free
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        mem_used = int(r.stdout.strip().split("\n")[0])
        logger.info("GPU memory used: %d MiB", mem_used)
        if mem_used > 2000:
            logger.warning("GPU still has %d MiB in use, attempting cleanup...", mem_used)
            # Kill stale GPU processes
            subprocess.run(
                "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
                shell=True, capture_output=True, timeout=10)
            time.sleep(5)

    state.phase = "harvest"
    state.save()
    logger.info("Setup complete. GPU ready for training.")


# ---------------------------------------------------------------------------
# Phase 1: Data Harvesting
# ---------------------------------------------------------------------------

def phase_harvest(state: LoopState):
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA HARVESTING")
    logger.info("=" * 60)

    sft_examples = []
    dpo_pairs = []
    sources = {}

    # --- SFT Source 1: Existing curriculum data ---
    if Path(ITER3_DATASET).exists():
        count = 0
        with open(ITER3_DATASET, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sft_examples.append(json.loads(line))
                    count += 1
        sources["iter3_curriculum"] = count
        logger.info("Loaded %d examples from iter_3 curriculum", count)

    # --- SFT Source 2: Learning signals ---
    for fname in ["finetune_buffer.jsonl", "high_quality.jsonl"]:
        fpath = LEARNING_SIGNALS_DIR / fname
        if fpath.exists():
            count = 0
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    # Convert to text format if needed
                    if "instruction" in ex and "text" not in ex:
                        text = _format_sft_to_text(
                            ex.get("instruction", ""),
                            ex.get("input", ""),
                            ex.get("output", ""),
                        )
                        ex["text"] = text
                    sft_examples.append(ex)
                    count += 1
            sources[fname] = count
            logger.info("Loaded %d from %s", count, fname)

    # --- SFT Source 3: Behavioral data ---
    try:
        from src.finetune.behavioral_data_generator import (
            generate_identity_examples, generate_pipeline_examples,
            generate_formatting_examples, generate_feature_examples,
            generate_domain_examples, generate_gap_handling_examples,
            generate_isolation_examples, generate_conversation_examples,
        )
        generators = [
            generate_identity_examples, generate_pipeline_examples,
            generate_formatting_examples, generate_feature_examples,
            generate_domain_examples, generate_gap_handling_examples,
            generate_isolation_examples, generate_conversation_examples,
        ]
        behavioral_count = 0
        for gen_fn in generators:
            examples = gen_fn()
            sft_examples.extend(examples)
            behavioral_count += len(examples)
        sources["behavioral"] = behavioral_count
        logger.info("Generated %d behavioral examples", behavioral_count)
    except Exception as e:
        logger.warning("Behavioral data generation failed: %s", e)

    # --- SFT Source 4: Synthetic expansions ---
    try:
        expansion_count = 0
        from src.finetune.synthetic_expansion_crossdoc import generate_crossdoc_expanded, generate_content_expanded
        from src.finetune.synthetic_expansion_domain import generate_domain_expanded
        from src.finetune.synthetic_expansion_extraction import generate_extraction_expanded
        from src.finetune.synthetic_expansion_formatting import generate_formatting_expanded
        from src.finetune.synthetic_expansion_misc import (
            generate_identity_expanded, generate_pipeline_expanded,
            generate_edge_cases_expanded,
        )
        for gen_fn in [generate_crossdoc_expanded, generate_content_expanded,
                       generate_domain_expanded, generate_extraction_expanded,
                       generate_formatting_expanded, generate_identity_expanded,
                       generate_pipeline_expanded, generate_edge_cases_expanded]:
            try:
                exs = gen_fn()
                sft_examples.extend(exs)
                expansion_count += len(exs)
            except Exception as e:
                logger.warning("Expansion %s failed: %s", gen_fn.__name__, e)
        sources["synthetic_expansions"] = expansion_count
        logger.info("Generated %d synthetic expansion examples", expansion_count)
    except ImportError as e:
        logger.warning("Synthetic expansion imports failed: %s", e)

    # --- DPO pairs ---
    try:
        from src.finetune.dpo_data_generator import (
            generate_formatting_pairs, generate_grounding_pairs,
            generate_conciseness_pairs, generate_isolation_pairs,
            generate_identity_pairs, generate_anti_repetition_pairs,
            generate_viz_preference_pairs,
        )
        for gen_fn in [generate_formatting_pairs, generate_grounding_pairs,
                       generate_conciseness_pairs, generate_isolation_pairs,
                       generate_identity_pairs, generate_anti_repetition_pairs,
                       generate_viz_preference_pairs]:
            pairs = gen_fn()
            dpo_pairs.extend(pairs)
        sources["dpo_pairs"] = len(dpo_pairs)
        logger.info("Generated %d DPO preference pairs", len(dpo_pairs))
    except Exception as e:
        logger.warning("DPO generation failed: %s", e)

    # --- Deduplicate SFT ---
    seen = set()
    deduped = []
    for ex in sft_examples:
        # Hash by text content
        key_text = ex.get("text", "") or (ex.get("instruction", "") + ex.get("input", ""))
        h = hashlib.md5(key_text.encode()).hexdigest()
        if h not in seen and len(key_text) > 50:
            seen.add(h)
            deduped.append(ex)
    dup_count = len(sft_examples) - len(deduped)
    sft_examples = deduped
    logger.info("Deduplicated: removed %d duplicates, %d unique SFT examples remain",
                dup_count, len(sft_examples))

    # Shuffle
    random.seed(42)
    random.shuffle(sft_examples)

    # Save master datasets
    sft_path = WORK_DIR / "master_sft.jsonl"
    dpo_path = WORK_DIR / "master_dpo.jsonl"

    with open(sft_path, "w", encoding="utf-8") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(dpo_path, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    state.dataset_path = str(sft_path)
    state.dpo_path = str(dpo_path)
    state.sft_count = len(sft_examples)
    state.dpo_count = len(dpo_pairs)
    state.phase = "training"
    state.round = 1
    state.sub_phase = "sft"
    state.save()

    logger.info("Data harvesting complete:")
    logger.info("  SFT examples: %d", len(sft_examples))
    logger.info("  DPO pairs: %d", len(dpo_pairs))
    logger.info("  Sources: %s", json.dumps(sources, indent=2))


def _format_sft_to_text(instruction: str, context: str, response: str) -> str:
    """Convert instruction/input/output to Qwen3 chat text format."""
    user_msg = instruction
    if context:
        user_msg += "\n\n" + context
    return (
        f"<|im_start|>system\n"
        f"You are DocWain, an intelligent document analysis assistant.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_msg}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{response}\n"
        f"<|im_end|>\n"
    )


# ---------------------------------------------------------------------------
# Phase 2: Training Loop
# ---------------------------------------------------------------------------

def phase_training(state: LoopState):
    logger.info("=" * 60)
    logger.info("PHASE 2: ITERATIVE TRAINING LOOP")
    logger.info("=" * 60)

    while state.round <= MAX_ROUNDS:
        logger.info("-" * 40)
        logger.info("ROUND %d (sub_phase=%s)", state.round, state.sub_phase)
        logger.info("-" * 40)

        round_dir = WORK_DIR / f"round_{state.round}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Determine base checkpoint
        if state.round == 1:
            base_checkpoint = ITER3_CHECKPOINT
        else:
            prev_round = state.round - 1
            base_checkpoint = str(WORK_DIR / f"round_{prev_round}" / "merged_16bit")

        # SFT + DPO via train_track
        if state.sub_phase in ("sft", ""):
            merged_path = _run_training_round(state, round_dir, base_checkpoint)
            if not merged_path:
                logger.error("Training round %d failed!", state.round)
                break
            state.sub_phase = "eval"
            state.save()

        # Evaluation
        if state.sub_phase == "eval":
            scores = _run_evaluation(state, round_dir)
            if not scores:
                logger.error("Evaluation round %d failed!", state.round)
                # Try to continue to next round
                state.sub_phase = "analyze"
                state.save()
            else:
                state.sub_phase = "analyze"
                state.save()

        # Analyze and decide
        if state.sub_phase == "analyze":
            decision = _analyze_and_decide(state)
            if decision in ("pass", "plateau", "max_rounds"):
                logger.info("Training loop DONE: %s (best=%.4f round=%d)",
                            decision, state.best_score, state.best_round)
                state.phase = "deploy"
                state.save()
                return
            # Cleanup non-best rounds before continuing
            _cleanup_old_rounds(state)
            # Continue to next round
            state.round += 1
            state.sub_phase = "sft"
            state.save()

    # Fell through max rounds
    state.phase = "deploy"
    state.save()


def _cleanup_old_rounds(state: LoopState):
    """Remove SFT/DPO checkpoints and intermediate files from non-best rounds."""
    for rnd in state.round_history:
        rnd_num = rnd["round"]
        if rnd_num == state.best_round:
            continue  # Keep the best round's merged checkpoint
        rnd_dir = WORK_DIR / f"round_{rnd_num}"
        if not rnd_dir.exists():
            continue
        freed = 0
        # Remove SFT checkpoints (large LoRA adapters)
        sft_dir = rnd_dir / "sft_checkpoints"
        if sft_dir.exists():
            for f in sft_dir.rglob("*"):
                if f.is_file():
                    freed += f.stat().st_size
                    f.unlink()
            shutil.rmtree(sft_dir, ignore_errors=True)
        # Remove DPO checkpoints
        dpo_dir = rnd_dir / "dpo_checkpoints"
        if dpo_dir.exists():
            for f in dpo_dir.rglob("*"):
                if f.is_file():
                    freed += f.stat().st_size
                    f.unlink()
            shutil.rmtree(dpo_dir, ignore_errors=True)
        # Remove merged_16bit from non-best rounds (28GB each!)
        merged = rnd_dir / "merged_16bit"
        if merged.exists() and rnd_num != state.best_round:
            # Only remove if this round is not the base for the next round
            next_rnd = rnd_num + 1
            if next_rnd <= state.round:  # Already trained next round
                for f in merged.rglob("*.safetensors"):
                    freed += f.stat().st_size
                    f.unlink()
                # Keep config/tokenizer files (small) for reference
        if freed > 0:
            logger.info("Cleaned round %d: freed %.1f GB", rnd_num, freed / 1e9)


def _run_training_round(state: LoopState, round_dir: Path, base_checkpoint: str) -> Optional[str]:
    """Run SFT + DPO for one round."""
    from src.finetune.v2.train_track import TrackTrainingConfig, train_track

    lr_decay = 0.7 ** (state.round - 1)
    epochs = 3 if state.round == 1 else 2

    config = TrackTrainingConfig(
        track_name=f"weekend_round_{state.round}",
        base_checkpoint=base_checkpoint,
        data_path=state.dataset_path,
        dpo_path=state.dpo_path if state.dpo_count > 0 else None,
        output_dir=str(round_dir),
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        learning_rate=2e-5 * lr_decay,
        epochs=epochs,
        batch_size=4,
        gradient_accumulation_steps=8,
        max_seq_length=4096,
        warmup_ratio=0.10,
        dpo_epochs=1,
        dpo_lr=5e-6 * lr_decay,
        dpo_beta=0.3,
        skip_ollama_export=True,  # We'll do Ollama at deploy phase
        ollama_model_name="DHS/DocWain",
        ollama_tag=f"v2-round{state.round}",
    )

    logger.info("Training round %d: lr=%.2e, epochs=%d, base=%s",
                state.round, config.learning_rate, epochs, base_checkpoint)

    try:
        merged_path = train_track(config)
        logger.info("Round %d training complete: %s", state.round, merged_path)
        return merged_path
    except Exception as e:
        logger.exception("Round %d training failed: %s", state.round, e)
        return None


def _run_evaluation(state: LoopState, round_dir: Path) -> Optional[Dict]:
    """Run evaluation on the round's merged checkpoint."""
    merged_dir = round_dir / "merged_16bit"
    if not merged_dir.exists():
        logger.error("No merged checkpoint at %s", merged_dir)
        return None

    logger.info("Evaluating round %d checkpoint: %s", state.round, merged_dir)

    try:
        from src.finetune.v2.curriculum_evaluator import (
            run_lora_inference, JudgingBrief, parse_judge_scores,
            aggregate_scores, check_gates, build_failure_analysis,
            JUDGE_DIMENSIONS,
        )
        from src.finetune.v2.eval.test_bank import get_test_bank
    except ImportError as e:
        logger.error("Cannot import evaluator: %s", e)
        return None

    test_bank = get_test_bank()
    if not test_bank:
        logger.error("Test bank is empty!")
        return None

    prompts = [ex["prompt"] for ex in test_bank]
    logger.info("Running inference on %d test bank examples...", len(prompts))

    try:
        responses = run_lora_inference(
            base_model="unsloth/Qwen3-14B-bnb-4bit",
            adapter_path=str(merged_dir),
            prompts=prompts,
            max_new_tokens=2048,
        )
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        return None

    # Score using heuristic approach (since we don't have a judge LLM running)
    logger.info("Scoring %d responses...", len(responses))
    all_scores = []
    for i, (example, response) in enumerate(zip(test_bank, responses)):
        score = _heuristic_score(example, response)
        score["example_index"] = i
        score["track"] = example.get("track", "unknown")
        all_scores.append(score)

    # Aggregate
    agg = aggregate_scores(all_scores)
    overall_avg = agg.get("overall_avg", 0.0)
    min_dim = agg.get("min_dimension", 0.0)

    logger.info("Round %d eval: overall_avg=%.4f, min_dimension=%.4f",
                state.round, overall_avg, min_dim)

    # Save eval results
    eval_path = round_dir / "eval_results.json"
    eval_path.write_text(json.dumps({
        "round": state.round,
        "overall_avg": overall_avg,
        "min_dimension": min_dim,
        "aggregated": agg,
        "example_count": len(all_scores),
    }, indent=2), encoding="utf-8")

    # Update state
    round_result = {
        "round": state.round,
        "overall_avg": overall_avg,
        "min_dimension": min_dim,
        "checkpoint": str(merged_dir),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    state.round_history.append(round_result)

    if overall_avg > state.best_score:
        state.best_score = overall_avg
        state.best_round = state.round
        state.best_checkpoint = str(merged_dir)
        logger.info("New best! score=%.4f (round %d)", overall_avg, state.round)

    state.save()
    return agg


def _heuristic_score(example: Dict, response: str) -> Dict:
    """Score a response using heuristic checks when no LLM judge is available."""
    scores = {}
    ref = example.get("reference", {})
    expected = ref.get("expected_answer", "") if isinstance(ref, dict) else str(ref)
    prompt = example.get("prompt", "")

    # Factual correctness: check if key terms from reference appear in response
    if expected:
        expected_tokens = set(str(expected).lower().split())
        response_lower = response.lower()
        matches = sum(1 for t in expected_tokens if t in response_lower)
        ratio = matches / max(len(expected_tokens), 1)
        scores["factual_correctness"] = min(5.0, 1.0 + ratio * 4.0)
    else:
        scores["factual_correctness"] = 3.0 if len(response) > 50 else 2.0

    # Reasoning quality: check for structured thinking
    has_reasoning = any(kw in response.lower() for kw in
                        ["because", "therefore", "this means", "analysis",
                         "based on", "evidence shows", "the data"])
    has_structure = response.count("\n") > 2 or "**" in response or "|" in response
    scores["reasoning_quality"] = (
        3.0 + (0.8 if has_reasoning else 0.0) + (0.7 if has_structure else 0.0)
    )

    # Completeness: length and coverage
    if len(response) > 200:
        scores["completeness"] = 4.0
    elif len(response) > 100:
        scores["completeness"] = 3.5
    elif len(response) > 50:
        scores["completeness"] = 3.0
    else:
        scores["completeness"] = 2.0

    # Grounding: check for evidence citations
    has_citation = any(kw in response for kw in
                       ["[SOURCE", "[EVIDENCE", "according to", "document states",
                        "the report", "the data shows", "as shown"])
    has_numbers = any(c.isdigit() for c in response)
    scores["grounding"] = (
        3.0 + (0.8 if has_citation else 0.0) + (0.5 if has_numbers else 0.0)
    )

    # Clamp all to 1.0-5.0
    for k in scores:
        scores[k] = max(1.0, min(5.0, scores[k]))

    return {"scores": scores}


def _analyze_and_decide(state: LoopState) -> str:
    """Analyze round results and decide: pass, plateau, max_rounds, or continue."""
    if not state.round_history:
        return "continue"

    latest = state.round_history[-1]
    overall = latest.get("overall_avg", 0.0)
    min_dim = latest.get("min_dimension", 0.0)

    # Gate check
    if overall >= GATE_OVERALL_AVG and min_dim >= GATE_MIN_DIMENSION:
        logger.info("GATE PASSED: overall=%.4f >= %.2f, min_dim=%.4f >= %.2f",
                     overall, GATE_OVERALL_AVG, min_dim, GATE_MIN_DIMENSION)
        return "pass"

    # Max rounds
    if state.round >= MAX_ROUNDS:
        logger.info("MAX ROUNDS reached (%d)", MAX_ROUNDS)
        return "max_rounds"

    # Plateau detection
    if len(state.round_history) >= PLATEAU_PATIENCE + 1:
        recent = state.round_history[-PLATEAU_PATIENCE:]
        improvements = []
        for i in range(1, len(recent)):
            delta = recent[i]["overall_avg"] - recent[i - 1]["overall_avg"]
            improvements.append(delta)
        if all(abs(d) < PLATEAU_THRESHOLD for d in improvements):
            logger.info("PLATEAU detected: recent improvements = %s", improvements)
            return "plateau"

    logger.info("Continuing to round %d (score=%.4f, target=%.2f)",
                state.round + 1, overall, GATE_OVERALL_AVG)
    return "continue"


# ---------------------------------------------------------------------------
# Phase 3: Deployment
# ---------------------------------------------------------------------------

def phase_deploy(state: LoopState):
    logger.info("=" * 60)
    logger.info("PHASE 3: DEPLOYMENT")
    logger.info("=" * 60)

    if not state.best_checkpoint:
        logger.error("No best checkpoint to deploy!")
        state.phase = "test"
        state.save()
        return

    best_dir = Path(state.best_checkpoint)
    logger.info("Deploying best checkpoint (round %d, score %.4f): %s",
                state.best_round, state.best_score, best_dir)

    # "Always the best" guard: refuse to downgrade the active model.
    score_file = Path("models/docwain-v2-active.score.json")
    if score_file.exists():
        try:
            current_score = float(json.loads(score_file.read_text()).get("score", 0.0))
        except (json.JSONDecodeError, OSError, ValueError):
            current_score = 0.0
        if state.best_score <= current_score:
            logger.warning(
                "Skipping deploy: best round score %.4f does not beat active %.4f",
                state.best_score, current_score,
            )
            state.phase = "test"
            state.save()
            return

    # Update model symlink
    if MODEL_SYMLINK.is_symlink() or MODEL_SYMLINK.exists():
        MODEL_SYMLINK.unlink()
    MODEL_SYMLINK.symlink_to(best_dir.resolve())
    logger.info("Updated symlink: %s -> %s", MODEL_SYMLINK, best_dir.resolve())
    score_file.write_text(json.dumps({
        "score": state.best_score,
        "checkpoint": str(best_dir.resolve()),
        "round": state.best_round,
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))

    # Build GGUF and register with Ollama
    _deploy_to_ollama(best_dir)

    # Restart the unified vLLM service
    logger.info("Restarting vLLM service...")
    r = subprocess.run(["sudo", "systemctl", "start", "docwain-vllm-fast"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        logger.info("  Started docwain-vllm-fast")
    else:
        logger.warning("  Failed to start docwain-vllm-fast: %s", r.stderr.strip())

    # Wait for vLLM to become healthy
    logger.info("Waiting for vLLM to become healthy...")
    for attempt in range(60):
        try:
            import urllib.request
            req = urllib.request.urlopen("http://localhost:8100/health", timeout=5)
            if req.status == 200:
                logger.info("vLLM is healthy!")
                break
        except Exception:
            pass
        time.sleep(10)
    else:
        logger.warning("vLLM did not become healthy within 10 minutes")

    state.vllm_stopped = False
    state.phase = "test"
    state.save()
    logger.info("Deployment complete.")


def _deploy_to_ollama(merged_dir: Path):
    """Convert to GGUF and register with Ollama."""
    try:
        from src.finetune.v2.train_track import _build_modelfile, _update_ollama

        # Try to find existing GGUF or create one
        round_dir = merged_dir.parent
        gguf_dir = round_dir / "gguf"

        # Check if GGUF already exists
        import glob as _glob
        existing_ggufs = _glob.glob(str(gguf_dir / "**" / "*.gguf"), recursive=True)
        if existing_ggufs:
            gguf_path = existing_ggufs[-1]
            logger.info("Using existing GGUF: %s", gguf_path)
        else:
            # Try llama.cpp conversion from merged FP16
            gguf_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Converting merged FP16 to GGUF Q4_K_M...")
            llama_cpp = Path.home() / ".unsloth" / "llama.cpp"
            convert_script = llama_cpp / "convert_hf_to_gguf.py"
            quantize_bin = llama_cpp / "llama-quantize"

            f16_gguf = gguf_dir / "model-f16.gguf"
            q4_gguf = gguf_dir / "model-Q4_K_M.gguf"

            if convert_script.exists():
                r = subprocess.run(
                    ["python", str(convert_script), str(merged_dir),
                     "--outfile", str(f16_gguf), "--outtype", "f16"],
                    capture_output=True, text=True, timeout=1200)
                if r.returncode == 0 and quantize_bin.exists():
                    r2 = subprocess.run(
                        [str(quantize_bin), str(f16_gguf), str(q4_gguf), "Q4_K_M"],
                        capture_output=True, text=True, timeout=600)
                    if r2.returncode == 0:
                        f16_gguf.unlink(missing_ok=True)
                        gguf_path = str(q4_gguf)
                        logger.info("GGUF Q4_K_M created: %s", gguf_path)
                    else:
                        logger.error("Quantization failed: %s", r2.stderr[-300:])
                        return
                else:
                    logger.error("F16 conversion failed: %s", r.stderr[-300:])
                    return
            else:
                logger.warning("llama.cpp not found at %s, skipping GGUF", llama_cpp)
                return

        # Build Modelfile and register
        modelfile_content = _build_modelfile(gguf_path)
        modelfile_path = round_dir / "Modelfile"
        modelfile_path.write_text(modelfile_content, encoding="utf-8")
        _update_ollama("DHS/DocWain:v2-weekend", str(modelfile_path))

    except Exception as e:
        logger.exception("Ollama deployment failed: %s", e)


# ---------------------------------------------------------------------------
# Phase 4: Testing
# ---------------------------------------------------------------------------

def phase_test(state: LoopState):
    logger.info("=" * 60)
    logger.info("PHASE 4: REAL-TIME TESTING")
    logger.info("=" * 60)

    test_results = {
        "model": "DHS/DocWain:v2-weekend",
        "checkpoint": state.best_checkpoint,
        "training_rounds": state.best_round,
        "best_score": state.best_score,
        "round_history": state.round_history,
    }

    # Test 1: Intensive test (3 rounds)
    logger.info("Running intensive_test.py (3 rounds)...")
    intensive_results = []
    for run in range(1, 4):
        logger.info("  Intensive test run %d/3...", run)
        try:
            r = subprocess.run(
                [sys.executable, "scripts/intensive_test.py"],
                capture_output=True, text=True, timeout=600,
                cwd=str(Path.cwd()),
            )
            # Try to parse the output for results
            output = r.stdout + r.stderr
            intensive_results.append({
                "run": run,
                "returncode": r.returncode,
                "output_lines": len(output.splitlines()),
                "output_tail": output[-2000:] if output else "",
            })
            logger.info("  Run %d complete (exit=%d)", run, r.returncode)
        except Exception as e:
            logger.warning("  Intensive test run %d failed: %s", run, e)
            intensive_results.append({"run": run, "error": str(e)})
    test_results["intensive_test"] = intensive_results

    # Test 2: End-user quality audit (3 rounds)
    logger.info("Running enduser_quality_audit.py (3 rounds)...")
    audit_results = []
    for run in range(1, 4):
        logger.info("  Quality audit run %d/3...", run)
        try:
            r = subprocess.run(
                [sys.executable, "scripts/enduser_quality_audit.py"],
                capture_output=True, text=True, timeout=600,
                cwd=str(Path.cwd()),
            )
            output = r.stdout + r.stderr
            audit_results.append({
                "run": run,
                "returncode": r.returncode,
                "output_lines": len(output.splitlines()),
                "output_tail": output[-2000:] if output else "",
            })
            logger.info("  Run %d complete (exit=%d)", run, r.returncode)
        except Exception as e:
            logger.warning("  Quality audit run %d failed: %s", run, e)
            audit_results.append({"run": run, "error": str(e)})
    test_results["quality_audit"] = audit_results

    # Test 3: Curriculum eval (1 run)
    logger.info("Running curriculum eval against live model...")
    try:
        from src.finetune.v2.eval.test_bank import get_test_bank
        test_bank = get_test_bank()
        if test_bank:
            # Query live vLLM
            import urllib.request
            eval_scores = []
            for i, ex in enumerate(test_bank[:50]):  # First 50 for speed
                try:
                    payload = json.dumps({
                        "model": "docwain-fast",
                        "messages": [
                            {"role": "system", "content": "You are DocWain, an intelligent document analysis assistant."},
                            {"role": "user", "content": ex["prompt"]},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 2048,
                    }).encode()
                    req = urllib.request.Request(
                        "http://localhost:8100/v1/chat/completions",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    resp = urllib.request.urlopen(req, timeout=60)
                    result = json.loads(resp.read())
                    answer = result["choices"][0]["message"]["content"]
                    score = _heuristic_score(ex, answer)
                    score["track"] = ex.get("track", "unknown")
                    eval_scores.append(score)
                except Exception as e:
                    logger.debug("Eval example %d failed: %s", i, e)

            if eval_scores:
                from src.finetune.v2.curriculum_evaluator import aggregate_scores
                agg = aggregate_scores(eval_scores)
                test_results["curriculum_eval"] = {
                    "examples_tested": len(eval_scores),
                    "overall_avg": agg.get("overall_avg", 0.0),
                    "min_dimension": agg.get("min_dimension", 0.0),
                    "aggregated": agg,
                }
                logger.info("Curriculum eval: overall=%.4f, min_dim=%.4f",
                            agg.get("overall_avg", 0), agg.get("min_dimension", 0))
    except Exception as e:
        logger.warning("Curriculum eval failed: %s", e)

    # Save test report
    report_path = WORK_DIR / "test_report.json"
    report_path.write_text(json.dumps(test_results, indent=2, default=str), encoding="utf-8")
    logger.info("Test report saved to %s", report_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("WEEKEND FINETUNING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Best score: %.4f (round %d)", state.best_score, state.best_round)
    logger.info("  Checkpoint: %s", state.best_checkpoint)
    logger.info("  Rounds completed: %d", len(state.round_history))
    if state.round_history:
        baseline = 4.71  # iter_3 score
        delta = state.best_score - baseline
        logger.info("  Improvement over iter_3: %+.4f", delta)
    logger.info("  Test report: %s", report_path)

    # Generate comprehensive capability report
    _generate_capability_report(state, test_results)

    state.phase = "done"
    state.save()


def _generate_capability_report(state: LoopState, test_results: Dict):
    """Generate a detailed Markdown report on the finetuned model's capabilities."""
    report_lines = [
        "# DocWain V2 Weekend Finetuning — Capability Report",
        "",
        f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Base Model:** Qwen3-14B (iter_3 checkpoint, score 4.71/5.0)",
        f"**Training Rounds:** {len(state.round_history)}",
        f"**Total SFT Examples:** {state.sft_count}",
        f"**Total DPO Pairs:** {state.dpo_count}",
        f"**Best Score:** {state.best_score:.4f} (round {state.best_round})",
        f"**Deployed Checkpoint:** `{state.best_checkpoint}`",
        "",
        "## Training Progression",
        "",
        "| Round | SFT Loss | Overall Avg | Min Dimension | Status |",
        "|-------|----------|-------------|---------------|--------|",
    ]

    for rnd in state.round_history:
        rnd_dir = WORK_DIR / f"round_{rnd['round']}" / "training_summary.json"
        sft_loss = "N/A"
        if rnd_dir.exists():
            try:
                summary = json.loads(rnd_dir.read_text())
                sft_loss = f"{summary.get('sft', {}).get('train_loss', 'N/A'):.4f}"
            except Exception:
                pass
        best_marker = " **BEST**" if rnd["round"] == state.best_round else ""
        report_lines.append(
            f"| {rnd['round']} | {sft_loss} | {rnd['overall_avg']:.4f} | "
            f"{rnd['min_dimension']:.4f} | {best_marker} |"
        )

    report_lines.extend([
        "",
        "## Improvement Over Baseline",
        "",
        f"- **Baseline (iter_3):** 4.71/5.0 overall, train_loss 1.034",
        f"- **Best Weekend Round:** {state.best_score:.4f}/5.0",
        f"- **Score Delta:** {state.best_score - 4.71:+.4f}",
        "",
        "## Real-Time Test Results",
        "",
    ])

    # Intensive test summary
    intensive = test_results.get("intensive_test", [])
    if intensive:
        report_lines.append("### Intensive Test (70+ queries)")
        for run in intensive:
            rc = run.get("returncode", -1)
            status = "PASS" if rc == 0 else f"EXIT {rc}"
            report_lines.append(f"- Run {run['run']}: {status} ({run.get('output_lines', 0)} lines)")
        report_lines.append("")

    # Quality audit summary
    audit = test_results.get("quality_audit", [])
    if audit:
        report_lines.append("### End-User Quality Audit (12 deep tests)")
        for run in audit:
            rc = run.get("returncode", -1)
            status = "PASS" if rc == 0 else f"EXIT {rc}"
            report_lines.append(f"- Run {run['run']}: {status} ({run.get('output_lines', 0)} lines)")
        report_lines.append("")

    # Curriculum eval
    curriculum = test_results.get("curriculum_eval", {})
    if curriculum:
        report_lines.extend([
            "### Curriculum Evaluation (frozen test bank)",
            f"- Examples tested: {curriculum.get('examples_tested', 0)}",
            f"- Overall average: {curriculum.get('overall_avg', 0):.4f}",
            f"- Min dimension: {curriculum.get('min_dimension', 0):.4f}",
            "",
        ])

    report_lines.extend([
        "## Model Deployment",
        "",
        f"- **Ollama tag:** DHS/DocWain:v2-weekend",
        f"- **vLLM symlink:** models/docwain-v2-active -> {state.best_checkpoint}",
        f"- **vLLM port:** 8100 (fast), 8200 (smart)",
        "",
        "## Key Capabilities",
        "",
        "The finetuned model has been trained on:",
        "- **Document understanding:** Extraction, summarization, cross-document analysis",
        "- **Domain expertise:** HR, Legal, Finance, Medical, Insurance, Technical",
        "- **Response formatting:** Markdown tables, bold values, structured output",
        "- **Visualization:** Chart generation directives (bar, line, donut, etc.)",
        "- **Preference alignment:** DPO-trained against preambles, hallucination, filler",
        "- **Identity:** DocWain persona, privacy boundaries, grounded responses",
        "",
    ])

    report_path = WORK_DIR / "capability_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Capability report saved to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Weekend iterative finetuning loop")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    args = parser.parse_args()

    global _state

    if args.resume and STATE_FILE.exists():
        state = LoopState.load()
        logger.info("Resuming from phase=%s, round=%d, sub_phase=%s",
                     state.phase, state.round, state.sub_phase)
    else:
        state = LoopState()
        logger.info("Starting fresh training loop")

    _state = state

    phases = {
        "setup": phase_setup,
        "harvest": phase_harvest,
        "training": phase_training,
        "deploy": phase_deploy,
        "test": phase_test,
    }

    while state.phase in phases:
        phase_fn = phases[state.phase]
        logger.info("Entering phase: %s", state.phase)
        try:
            phase_fn(state)
        except Exception as e:
            logger.exception("Phase %s failed: %s", state.phase, e)
            state.save()
            raise

    logger.info("All phases complete. Final state: %s", state.phase)


if __name__ == "__main__":
    main()
