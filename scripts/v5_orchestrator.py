#!/usr/bin/env python3
"""V5 Pipeline Autonomous Orchestrator.

Runs as a background process once data gen is launched. Walks the
pipeline through every phase to deployment, gates each transition,
applies fallbacks on failure, and writes a state JSON + log that can
be inspected at any time.

Phases (state machine):
  1. WAIT_DATAGEN     — poll until both parallel data_gen PIDs exit
  2. MERGE_CORPUS     — concatenate sft_reused + sft_generated{,_partB}
  3. QUALITY_CHECK    — per-capability counts meet minimum thresholds
  4. STOP_VLLM        — systemctl stop docwain-vllm-fast
  5. LAUNCH_SFT       — start 14B LoRA SFT from V3 (seed merge failed)
  6. WAIT_SFT         — poll checkpoint+process until SFT exits
  7. LAUNCH_DPO       — start DPO on SFT output
  8. WAIT_DPO         — poll until DPO exits
  9. EVAL_14B         — run evaluate.py on V5-14B, check gates
 10. LAUNCH_DISTILL   — 8B distillation, teacher = V5-14B if gate pass
                       else V3 (Tier-3 fallback)
 11. WAIT_DISTILL     — poll until distillation exits
 12. QUANTIZE         — int8 for 14B + GGUF Q5_K_M for both
 13. DEPLOY           — relaunch vLLM with V5-14B-int8
 14. DONE

On any fatal failure in phases 4-13, the orchestrator halts, writes a
FAILED state, and emits a recovery recommendation. V3 is never touched
destructively; the seed eval + fallback pattern is honored throughout.

Usage (run once after data gen is up):
    nohup python3 -u scripts/v5_orchestrator.py \\
        --pid-a <DATAGEN_A_PID> --pid-b <DATAGEN_B_PID> \\
        > /tmp/v5_orchestrator_stdout.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

STATE_PATH = ROOT / "finetune_artifacts/v5/orchestrator_state.json"
LOG_PATH = ROOT / "finetune_artifacts/v5/orchestrator.log"
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

SFT_REUSED = ROOT / "finetune_artifacts/v5/sft_reused.jsonl"
SFT_GENERATED = ROOT / "finetune_artifacts/v5/sft_generated.jsonl"
SFT_GENERATED_B = ROOT / "finetune_artifacts/v5/sft_generated_partB.jsonl"
SFT_GENERATED_C = ROOT / "finetune_artifacts/v5/sft_generated_partC.jsonl"
DPO_REUSED = ROOT / "finetune_artifacts/v5/dpo_reused.jsonl"
DPO_GENERATED = ROOT / "finetune_artifacts/v5/dpo_generated.jsonl"
DPO_GENERATED_B = ROOT / "finetune_artifacts/v5/dpo_generated_partB.jsonl"
DPO_GENERATED_C = ROOT / "finetune_artifacts/v5/dpo_generated_partC.jsonl"
SFT_COMBINED = ROOT / "finetune_artifacts/v5/sft_combined.jsonl"
DPO_COMBINED = ROOT / "finetune_artifacts/v5/dpo_combined.jsonl"

V3_WEIGHTS = ROOT / "models/DocWain-14B-v2"
SFT_OUT = ROOT / "models/DocWain-14B-v5-sft"
DPO_OUT = ROOT / "models/DocWain-14B-v5"
STUDENT_OUT = ROOT / "models/DocWain-8B-v5"
QUANTIZED_14B = ROOT / "models/DocWain-14B-v5-int8"
GGUF_14B = ROOT / "models/DocWain-14B-v5-q5km.gguf"
GGUF_8B = ROOT / "models/DocWain-8B-v5-q5km.gguf"

SFT_LOG = Path("/tmp/v5_sft.log")
DPO_LOG = Path("/tmp/v5_dpo.log")
DISTILL_LOG = Path("/tmp/v5_distill.log")

CAPABILITY_MINIMUMS = {
    "schema_adherence": 500,
    "grounded_refusal": 500,
    "tool_calling": 400,
    "identity_in_weights": 200,
    "domain_recognition": 300,
    "doctype_classification": 200,
    "context_dependence": 300,
    "citation_discipline": 200,
}


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"{now_iso()}  {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"phase": "INIT", "history": [], "started": now_iso()}
    return json.loads(STATE_PATH.read_text())


def save_state(state: Dict[str, Any]) -> None:
    state["updated"] = now_iso()
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def set_phase(state: Dict[str, Any], phase: str, **extra: Any) -> None:
    prev = state.get("phase")
    if prev != phase:
        log(f"phase transition: {prev} → {phase}")
        state.setdefault("history", []).append(
            {"phase": prev, "left_at": now_iso(), **{k: v for k, v in extra.items()}}
        )
    state["phase"] = phase
    state["phase_entered"] = now_iso()
    save_state(state)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    """True only if pid is running. Zombies count as exited.

    ``os.kill(pid, 0)`` succeeds on zombie processes because the pid is
    still in the process table, which would keep the orchestrator
    blocked on ``wait_pid`` forever when a child exits normally but
    hasn't been reaped. We explicitly read /proc/<pid>/status and
    return False on the zombie state letter Z so exit is detected.
    """
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("State:"):
                    # e.g. "State:\tZ (zombie)"  — treat as exited
                    if "Z" in line.split()[1]:
                        return False
                    break
    except (FileNotFoundError, PermissionError):
        return False
    return True


def run_cmd(cmd: List[str], log_path: Path, cwd: Path = ROOT) -> int:
    """Run a command to completion, redirecting output."""
    log(f"run: {' '.join(cmd)}")
    with log_path.open("a") as f:
        f.write(f"\n=== {now_iso()} START {' '.join(cmd)} ===\n")
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    log(f"exit: rc={p.returncode} for {cmd[0]}")
    return p.returncode


def launch_bg(cmd: List[str], log_path: Path, cwd: Path = ROOT) -> int:
    """Spawn a command in the background, return PID."""
    log(f"launch_bg: {' '.join(cmd)}")
    with log_path.open("a") as f:
        f.write(f"\n=== {now_iso()} LAUNCH {' '.join(cmd)} ===\n")
    p = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=log_path.open("a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return p.pid


def wait_pid(pid: int, poll_s: int = 60, max_hours: int = 48) -> int:
    """Block until pid exits. Return exit code (or -1 if we can't tell)."""
    deadline = time.monotonic() + max_hours * 3600
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return 0
        time.sleep(poll_s)
    log(f"wait_pid: {pid} exceeded {max_hours}h — bailing")
    return -1


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def wait_datagen(state: Dict[str, Any], pids: List[int]) -> bool:
    """Phase 1. Block until all data_gen processes exit.

    Accepts any number of pids so the orchestrator can be (re)started
    over additional parallel generators (e.g. Process C after a mid-run
    template fix). Waits on union — doesn't advance until everyone exits.
    """
    set_phase(state, "WAIT_DATAGEN", pids=pids)
    log(f"waiting on datagen pids={pids}")
    while any(pid_alive(p) for p in pids):
        hb = {
            f"pid_{p}_alive": pid_alive(p) for p in pids
        }
        hb["a_rows"] = count_lines(SFT_GENERATED)
        hb["b_rows"] = count_lines(SFT_GENERATED_B)
        hb["c_rows"] = count_lines(SFT_GENERATED_C)
        log(f"datagen heartbeat: {hb}")
        state["datagen_heartbeat"] = hb
        save_state(state)
        time.sleep(300)
    log(f"all {len(pids)} datagen processes exited")
    return True


def count_lines(path: Path) -> int:
    try:
        with path.open("r") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _model_output_present(directory: Path) -> bool:
    """True if ``directory`` looks like a completed merged HF model."""
    if not directory.exists():
        return False
    if (directory / "model.safetensors.index.json").exists():
        return True
    return bool(list(directory.glob("model-*.safetensors")))


def merge_corpus(state: Dict[str, Any]) -> bool:
    """Phase 2. Concatenate into combined files."""
    set_phase(state, "MERGE_CORPUS")
    sources_sft = [p for p in (SFT_REUSED, SFT_GENERATED, SFT_GENERATED_B, SFT_GENERATED_C) if p.exists()]
    sources_dpo = [p for p in (DPO_REUSED, DPO_GENERATED, DPO_GENERATED_B, DPO_GENERATED_C) if p.exists()]

    with SFT_COMBINED.open("w") as out:
        for p in sources_sft:
            with p.open("r") as f:
                for line in f:
                    out.write(line)
    with DPO_COMBINED.open("w") as out:
        for p in sources_dpo:
            with p.open("r") as f:
                for line in f:
                    out.write(line)

    sft_count = count_lines(SFT_COMBINED)
    dpo_count = count_lines(DPO_COMBINED)
    log(f"merged: sft={sft_count} dpo={dpo_count}")
    state["corpus_counts"] = {"sft": sft_count, "dpo": dpo_count}
    save_state(state)
    return True


def quality_check(state: Dict[str, Any]) -> bool:
    """Phase 3. Verify per-capability coverage."""
    set_phase(state, "QUALITY_CHECK")
    from collections import Counter
    cap_counts: Counter = Counter()
    with SFT_COMBINED.open("r") as f:
        for line in f:
            try:
                cap_counts[json.loads(line).get("capability", "?")] += 1
            except json.JSONDecodeError:
                continue
    log(f"per-capability counts: {dict(cap_counts)}")
    state["capability_counts"] = dict(cap_counts)

    under = {
        cap: cap_counts.get(cap, 0)
        for cap, minimum in CAPABILITY_MINIMUMS.items()
        if cap_counts.get(cap, 0) < minimum
    }
    if under:
        log(f"WARNING: capabilities below minimum: {under}")
        state["quality_warnings"] = under
    total = sum(cap_counts.values())
    if total < 30_000:
        log(f"FATAL: combined SFT corpus too small ({total} rows) — aborting")
        save_state(state)
        return False
    save_state(state)
    return True


def stop_vllm(state: Dict[str, Any]) -> bool:
    """Phase 4. Free GPU for training."""
    set_phase(state, "STOP_VLLM")
    run_cmd(
        ["sudo", "systemctl", "stop", "docwain-vllm-fast", "docwain-gpu-scheduler"],
        LOG_PATH,
    )
    time.sleep(10)
    # Verify nothing's using the GPU in a way that would OOM training
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        log(f"gpu mem_used after stop: {out} MiB")
        state["gpu_memory_pre_sft"] = out
    except Exception as exc:
        log(f"nvidia-smi check failed: {exc}")
    save_state(state)
    return True


def launch_sft(state: Dict[str, Any]) -> Optional[int]:
    """Phase 5. Start LoRA SFT from V3."""
    set_phase(state, "LAUNCH_SFT")
    pid = launch_bg(
        [
            sys.executable, "-u", "-m", "src.finetune.v5.sft_trainer",
            "--base", str(V3_WEIGHTS),
            "--corpus", str(SFT_COMBINED),
            "--output", str(SFT_OUT),
            "--lora-rank", "128", "--lora-alpha", "32",
            "--epochs", "2",
            "--batch-size", "1", "--grad-accum", "16",
            "--learning-rate", "2e-5",
            "--warmup-ratio", "0.03",
            "--checkpoint-interval", "6h",
        ],
        SFT_LOG,
    )
    state["sft_pid"] = pid
    save_state(state)
    log(f"sft launched pid={pid}")
    return pid


def wait_sft(state: Dict[str, Any], pid: int) -> bool:
    """Phase 6."""
    set_phase(state, "WAIT_SFT", pid=pid)
    rc = wait_pid(pid, poll_s=180, max_hours=30)
    model_file = SFT_OUT / "model.safetensors.index.json"
    success = model_file.exists() or any(SFT_OUT.glob("model-*.safetensors"))
    state["sft_exit_rc"] = rc
    state["sft_output_present"] = bool(success)
    save_state(state)
    log(f"sft done rc={rc} output_present={success}")
    return bool(success)


def launch_dpo(state: Dict[str, Any]) -> Optional[int]:
    """Phase 7."""
    set_phase(state, "LAUNCH_DPO")
    pid = launch_bg(
        [
            sys.executable, "-u", "-m", "src.finetune.v5.dpo_trainer",
            "--base", str(SFT_OUT),
            "--pairs", str(DPO_COMBINED),
            "--output", str(DPO_OUT),
            "--lora-rank", "128", "--lora-alpha", "32",
            "--beta", "0.1", "--epochs", "2",
            "--batch-size", "1", "--grad-accum", "8",
            "--learning-rate", "5e-6",
            "--warmup-ratio", "0.03",
        ],
        DPO_LOG,
    )
    state["dpo_pid"] = pid
    save_state(state)
    log(f"dpo launched pid={pid}")
    return pid


def wait_dpo(state: Dict[str, Any], pid: int) -> bool:
    set_phase(state, "WAIT_DPO", pid=pid)
    rc = wait_pid(pid, poll_s=180, max_hours=16)
    success = (DPO_OUT / "model.safetensors.index.json").exists() or any(
        DPO_OUT.glob("model-*.safetensors")
    )
    state["dpo_exit_rc"] = rc
    state["dpo_output_present"] = bool(success)
    save_state(state)
    log(f"dpo done rc={rc} output_present={success}")
    return bool(success)


def eval_14b(state: Dict[str, Any]) -> bool:
    """Phase 9. Charter-gate evaluation."""
    set_phase(state, "EVAL_14B")
    rc = run_cmd(
        [
            sys.executable, "-u", "-m", "src.finetune.v5.evaluate",
            str(DPO_OUT),
        ],
        LOG_PATH,
    )
    report_path = ROOT / f"finetune_artifacts/v5/eval/{DPO_OUT.name}.json"
    passed = False
    if report_path.exists():
        report = json.loads(report_path.read_text())
        passed = bool(report.get("overall", {}).get("all_hard_gates_passed"))
        state["eval_14b_report"] = report_path.name
        state["eval_14b_passed"] = passed
    else:
        log("eval report not found")
        state["eval_14b_passed"] = False
    save_state(state)
    log(f"14b gate: passed={passed}")
    return passed


def launch_distill(state: Dict[str, Any]) -> Optional[int]:
    """Phase 10. 8B distillation. Teacher depends on 14B gate."""
    set_phase(state, "LAUNCH_DISTILL")
    teacher = DPO_OUT if state.get("eval_14b_passed") else V3_WEIGHTS
    state["distill_teacher"] = str(teacher)
    save_state(state)
    log(f"distill teacher = {teacher}  (gate_passed={state.get('eval_14b_passed')})")
    pid = launch_bg(
        [
            sys.executable, "-u", "-m", "src.finetune.v5.distillation",
            "--teacher", str(teacher),
            "--student", "Qwen/Qwen3-8B",
            "--corpus", str(SFT_COMBINED),
            "--output", str(STUDENT_OUT),
            "--alpha", "0.5", "--temperature", "2.0",
            "--batch-size", "2", "--grad-accum", "8",
            "--learning-rate", "1e-5",
            "--checkpoint-interval", "4h",
            "--cache-topk", "256",
        ],
        DISTILL_LOG,
    )
    state["distill_pid"] = pid
    save_state(state)
    return pid


def wait_distill(state: Dict[str, Any], pid: int) -> bool:
    set_phase(state, "WAIT_DISTILL", pid=pid)
    rc = wait_pid(pid, poll_s=180, max_hours=24)
    success = (STUDENT_OUT / "model.safetensors.index.json").exists() or any(
        STUDENT_OUT.glob("model-*.safetensors")
    )
    state["distill_exit_rc"] = rc
    state["distill_output_present"] = bool(success)
    save_state(state)
    log(f"distill done rc={rc} output_present={success}")
    return bool(success)


def quantize(state: Dict[str, Any]) -> bool:
    """Phase 11. Produce int8 + GGUF variants. Done via external tooling; we
    mark this step as 'queued' if tooling paths aren't wired in-repo, so the
    operator can pick up the handoff without blocking the rest of the state."""
    set_phase(state, "QUANTIZE")
    log("quantize step: scripting path not yet auto-wired — writing manifest for manual handoff")
    manifest = {
        "14b_source": str(DPO_OUT),
        "14b_int8_target": str(QUANTIZED_14B),
        "14b_gguf_q5km_target": str(GGUF_14B),
        "8b_source": str(STUDENT_OUT),
        "8b_gguf_q5km_target": str(GGUF_8B),
        "commands_to_run_manually": [
            "# int8 via bitsandbytes",
            f"python -m src.serving.quantize --input {DPO_OUT} --output {QUANTIZED_14B} --format int8",
            "# GGUF Q5_K_M",
            f"python -m llama_cpp.convert --outtype q5_k_m {DPO_OUT} {GGUF_14B}",
            f"python -m llama_cpp.convert --outtype q5_k_m {STUDENT_OUT} {GGUF_8B}",
        ],
    }
    (ROOT / "finetune_artifacts/v5/quantize_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    state["quantize_manifest"] = "finetune_artifacts/v5/quantize_manifest.json"
    save_state(state)
    return True


def deploy(state: Dict[str, Any]) -> bool:
    """Phase 12. Relaunch vLLM if a quantized 14B exists."""
    set_phase(state, "DEPLOY")
    if QUANTIZED_14B.exists():
        log("relaunching vLLM with V5-14B-int8")
        run_cmd(["sudo", "ln", "-sfn", str(QUANTIZED_14B), str(ROOT / "models/docwain-v2-active")],
                LOG_PATH)
        run_cmd(["sudo", "systemctl", "start", "docwain-vllm-fast", "docwain-gpu-scheduler"],
                LOG_PATH)
    else:
        log("quantized 14b absent — restarting vLLM on V3 as holding pattern")
        run_cmd(["sudo", "systemctl", "start", "docwain-vllm-fast", "docwain-gpu-scheduler"],
                LOG_PATH)
    save_state(state)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def orchestrate(pids: List[int]) -> int:
    state = load_state()
    state.setdefault("datagen", {"pids": pids})
    # Allow restart to update the pid list (e.g. Process C joined after a fix)
    state["datagen"]["pids"] = pids
    save_state(state)

    # Phase 1 — wait for all data gen processes
    wait_datagen(state, pids)

    # Phase 2 — merge
    merge_corpus(state)

    # Phase 3 — quality check
    if not quality_check(state):
        set_phase(state, "FAILED", reason="quality_check_insufficient_corpus")
        return 2

    # Phase 4 — stop vLLM
    stop_vllm(state)

    # Phase 5/6 — SFT (skip if output already produced by an earlier run)
    if _model_output_present(SFT_OUT):
        log(f"SFT output already present at {SFT_OUT} — skipping re-training")
        set_phase(state, "WAIT_SFT", skipped=True)
        state["sft_output_present"] = True
        save_state(state)
    else:
        sft_pid = launch_sft(state)
        if sft_pid is None:
            set_phase(state, "FAILED", reason="sft_launch_failed")
            return 3
        if not wait_sft(state, sft_pid):
            set_phase(state, "FAILED", reason="sft_no_output")
            return 4

    # Phase 7/8 — DPO (skip if output already produced)
    if _model_output_present(DPO_OUT):
        log(f"DPO output already present at {DPO_OUT} — skipping re-training")
        set_phase(state, "WAIT_DPO", skipped=True)
        state["dpo_output_present"] = True
        save_state(state)
        dpo_pid = None
    else:
        dpo_pid = launch_dpo(state)
        if dpo_pid is None:
            set_phase(state, "FAILED", reason="dpo_launch_failed")
            return 5
    if dpo_pid is not None and not wait_dpo(state, dpo_pid):
        log("DPO produced no output — degrading: using SFT-only output as V5-14B final")
        # Fallback: SFT alone is still a valid V5
        import shutil
        if SFT_OUT.exists() and not DPO_OUT.exists():
            shutil.copytree(SFT_OUT, DPO_OUT)
        state["dpo_fallback"] = True
        save_state(state)

    # Phase 9 — 14B gate eval
    eval_14b(state)  # result recorded even if fails

    # Phase 10/11 — distillation (skip if output already produced)
    if _model_output_present(STUDENT_OUT):
        log(f"Distill output already present at {STUDENT_OUT} — skipping re-training")
        set_phase(state, "WAIT_DISTILL", skipped=True)
        state["distill_output_present"] = True
        save_state(state)
        distill_pid = None
    else:
        distill_pid = launch_distill(state)
        if distill_pid is None:
            set_phase(state, "FAILED", reason="distill_launch_failed")
            return 6
    if distill_pid is not None and not wait_distill(state, distill_pid):
        log("distillation produced no output — 8B variant not shipped")
        state["distill_failed"] = True
        save_state(state)

    # Phase 12 — quantize
    quantize(state)

    # Phase 13 — deploy
    deploy(state)

    set_phase(state, "DONE")
    log("pipeline complete")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pids", type=str, required=True,
                    help="comma-separated list of data_gen process IDs to wait on")
    args = ap.parse_args()
    pids = [int(p) for p in args.pids.split(",") if p.strip()]
    return orchestrate(pids)


if __name__ == "__main__":
    sys.exit(main())
