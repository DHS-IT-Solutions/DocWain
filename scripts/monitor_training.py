#!/usr/bin/env python3
"""Active training monitor — checks gate scores, stops stalled training,
diagnoses gaps, and restarts.

Run hourly via cron or the loop skill:
    python scripts/monitor_training.py

Exit codes:
    0 — healthy or fixed and restarted
    1 — intervention needed (couldn't auto-fix)
    2 — training complete (all tracks passed)
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_DIR / "finetune_artifacts" / "v2_upgrade" / "state.json"
LOG_FILE = PROJECT_DIR / "finetune_artifacts" / "v2_upgrade" / "training.log"
STDOUT_LOG = PROJECT_DIR / "finetune_artifacts" / "v2_upgrade" / "training_stdout.log"
MONITOR_LOG = PROJECT_DIR / "finetune_artifacts" / "v2_upgrade" / "monitor.log"

GATE_THRESHOLD = 4.0
# If score hasn't improved after this many iterations, something is wrong
STALL_ITERATIONS = 3
# Minimum acceptable score — below this after 2+ iterations means broken
MIN_ACCEPTABLE_SCORE = 1.0


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(MONITOR_LOG, "a") as f:
        f.write(line + "\n")


def get_trainer_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", "autonomous_trainer"],
        capture_output=True, text=True,
    )
    return [int(p) for p in result.stdout.strip().split("\n") if p.strip()]


def is_training_running() -> bool:
    return len(get_trainer_pids()) > 0


def stop_training() -> None:
    pids = get_trainer_pids()
    if not pids:
        log("No training process to stop")
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"Sent SIGTERM to PID {pid}")
        except ProcessLookupError:
            pass
    # Wait up to 30s for graceful shutdown
    for _ in range(30):
        if not is_training_running():
            log("Training stopped gracefully")
            return
        time.sleep(1)
    # Force kill
    for pid in get_trainer_pids():
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"Sent SIGKILL to PID {pid}")
        except ProcessLookupError:
            pass


def start_training() -> int:
    """Start autonomous trainer, return PID."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.finetune.v2.autonomous_trainer", "--resume"],
        stdout=open(STDOUT_LOG, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_DIR),
        start_new_session=True,
    )
    log(f"Started training with PID {proc.pid}")
    return proc.pid


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def diagnose_track(track_name: str, track_data: dict) -> dict:
    """Analyse a track's iteration history and return diagnosis."""
    iterations = track_data.get("iterations", [])
    if not iterations:
        return {"status": "no_data", "action": "wait"}

    scores = [it.get("avg_score", 0) for it in iterations]
    latest_score = scores[-1]
    best_score = max(scores)
    num_iters = len(iterations)
    latest_strategy = iterations[-1].get("strategy", "unknown")

    # Check if gate is passed
    if latest_score >= GATE_THRESHOLD:
        return {"status": "passed", "score": latest_score, "action": "none"}

    # Check for zero or near-zero scores (broken)
    if latest_score <= MIN_ACCEPTABLE_SCORE and num_iters >= 2:
        return {
            "status": "broken",
            "score": latest_score,
            "action": "restart_fresh",
            "reason": f"Score {latest_score:.2f} after {num_iters} iterations — model may be corrupted",
        }

    # Check for stall (no improvement over STALL_ITERATIONS)
    if num_iters >= STALL_ITERATIONS:
        recent = scores[-STALL_ITERATIONS:]
        improvement = max(recent) - min(recent)
        if improvement < 0.1:
            return {
                "status": "stalled",
                "score": latest_score,
                "action": "advance_strategy",
                "reason": f"Score stuck at ~{latest_score:.2f} for {STALL_ITERATIONS} iterations (strategy={latest_strategy})",
            }

    # Check for regression (score going down)
    if num_iters >= 2 and scores[-1] < scores[-2] - 0.3:
        return {
            "status": "regressing",
            "score": latest_score,
            "action": "revert_to_best",
            "reason": f"Score dropped from {scores[-2]:.2f} to {scores[-1]:.2f}",
        }

    # Below average but making progress
    if num_iters >= 2 and scores[-1] > scores[-2]:
        return {
            "status": "improving",
            "score": latest_score,
            "action": "continue",
            "reason": f"Score improving: {scores[-2]:.2f} → {scores[-1]:.2f}",
        }

    return {
        "status": "in_progress",
        "score": latest_score,
        "action": "continue",
        "reason": f"Iteration {num_iters}, score {latest_score:.2f}, strategy {latest_strategy}",
    }


def check_for_crash(state: dict) -> bool:
    """Check if training appears to have crashed (not running but not complete)."""
    if is_training_running():
        return False

    completed = state.get("completed_tracks", [])
    all_tracks = ["excel_csv", "layout", "ocr_vision", "reasoning", "kg", "visualization"]

    if set(completed) >= set(all_tracks):
        return False  # All done

    # Not running, not complete — likely crashed
    return True


def main() -> int:
    log("=" * 60)
    log("DocWain V2 Training Monitor — Active Check")
    log("=" * 60)

    state = load_state()
    if not state:
        log("No state file — training hasn't started yet")
        if not is_training_running():
            log("Starting training for the first time")
            start_training()
        return 0

    completed = state.get("completed_tracks", [])
    all_tracks = ["excel_csv", "layout", "ocr_vision", "reasoning", "kg", "visualization"]
    history = state.get("track_history", {})

    log(f"Completed tracks: {completed or 'none'}")
    log(f"Running: {is_training_running()}")

    # Check if all done
    if set(completed) >= set(all_tracks):
        log("ALL TRACKS PASSED — training pipeline complete!")
        return 2

    needs_restart = False
    problem_found = False

    # Diagnose each active track
    for track in all_tracks:
        if track in completed:
            continue

        track_data = history.get(track, {})
        if not track_data.get("iterations"):
            continue

        diagnosis = diagnose_track(track, track_data)
        log(f"Track {track}: {diagnosis['status']} (score={diagnosis.get('score', 'n/a')}) — {diagnosis.get('reason', diagnosis['action'])}")

        if diagnosis["action"] == "restart_fresh":
            log(f"CRITICAL: Track {track} is broken — will stop and restart")
            problem_found = True
            needs_restart = True

        elif diagnosis["action"] == "advance_strategy":
            log(f"WARNING: Track {track} is stalled — training loop should auto-advance strategy")
            # The autonomous trainer already handles this via no_improve_count
            # But if it hasn't been advancing, we may need to restart
            if not is_training_running():
                needs_restart = True

        elif diagnosis["action"] == "revert_to_best":
            log(f"WARNING: Track {track} is regressing — training loop should recover")
            if not is_training_running():
                needs_restart = True

    # Check for crash
    if check_for_crash(state):
        log("DETECTED: Training process not running but pipeline not complete")
        needs_restart = True

    if needs_restart:
        log("Stopping any remaining processes and restarting training")
        stop_training()
        time.sleep(5)
        start_training()
        log("Training restarted successfully")
        return 0

    if not problem_found:
        # Print summary
        for track in all_tracks:
            track_data = history.get(track, {})
            iters = track_data.get("iterations", [])
            if iters:
                best = track_data.get("best_score", 0)
                log(f"  {track}: {len(iters)} iters, best={best:.2f}/4.0")
            elif track not in completed:
                log(f"  {track}: not started yet")

    log("Monitor check complete")
    return 0


if __name__ == "__main__":
    os.chdir(str(PROJECT_DIR))
    sys.exit(main())
