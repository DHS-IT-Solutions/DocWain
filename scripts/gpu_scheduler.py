"""DocWain GPU Scheduler — manages vLLM/training coexistence on a single A100.

When vLLM is idle and training is pending, stops vLLM, runs the
autonomous trainer, hot-swaps the model symlink, and restarts vLLM.
During training, queries fall back to Ollama Cloud (397b).

Usage::

    python scripts/gpu_scheduler.py --once
    python scripts/gpu_scheduler.py --interval 300 --idle-minutes 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "gpu_scheduler.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("gpu_scheduler")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "finetune_artifacts" / "v2_upgrade" / "state.json"
MODEL_SYMLINK = PROJECT_ROOT / "models" / "docwain-v2-active"

DEFAULT_GPU_MODE_FILE = "/tmp/docwain-gpu-mode.json"
DEFAULT_TRAINING_QUEUE_FILE = "/tmp/docwain-training-queue.json"


def _systemctl(action: str, service: str) -> None:
    """Run ``sudo systemctl <action> <service>``."""
    cmd = ["sudo", "systemctl", action, service]
    logger.info("systemctl: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=120)


class GPUScheduler:
    """Decide whether the GPU should serve or train, and act on it."""

    def __init__(
        self,
        gpu_mode_file: str = DEFAULT_GPU_MODE_FILE,
        training_queue_file: str = DEFAULT_TRAINING_QUEUE_FILE,
        vllm_url: str = "http://127.0.0.1:8100",
        idle_threshold_minutes: int = 30,
    ) -> None:
        self.gpu_mode_file = Path(gpu_mode_file)
        self.training_queue_file = Path(training_queue_file)
        self.vllm_url = vllm_url.rstrip("/")
        self.idle_threshold_s = idle_threshold_minutes * 60
        self._last_request_time: float = time.time()

        # Ensure mode file exists
        if not self.gpu_mode_file.exists():
            self.set_gpu_mode("serving")

    # ------------------------------------------------------------------
    # Mode file helpers
    # ------------------------------------------------------------------

    def get_gpu_mode(self) -> str:
        """Return current GPU mode (``serving`` or ``training``)."""
        try:
            data = json.loads(self.gpu_mode_file.read_text())
            return data.get("mode", "serving")
        except (FileNotFoundError, json.JSONDecodeError):
            return "serving"

    def set_gpu_mode(self, mode: str) -> None:
        """Persist *mode* to the GPU mode file."""
        payload = {
            "mode": mode,
            "since": datetime.now(timezone.utc).isoformat(),
        }
        self.gpu_mode_file.write_text(json.dumps(payload, indent=2))
        logger.info("GPU mode set to %s", mode)

    # ------------------------------------------------------------------
    # Training queue helpers
    # ------------------------------------------------------------------

    def is_training_pending(self) -> bool:
        """Return ``True`` when ``training_queue.json`` has pending work."""
        try:
            data = json.loads(self.training_queue_file.read_text())
            return bool(data.get("pending", False))
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def clear_training_queue(self) -> None:
        """Mark training queue as empty."""
        self.training_queue_file.write_text(
            json.dumps({"pending": False, "cleared_at": datetime.now(timezone.utc).isoformat()})
        )
        logger.info("Training queue cleared")

    # ------------------------------------------------------------------
    # vLLM metrics
    # ------------------------------------------------------------------

    def _check_vllm_request_rate(self) -> Optional[float]:
        """Query vLLM ``/metrics`` and return ``vllm:num_requests_running``.

        Returns ``None`` on connection error (vLLM may be down).
        """
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"{self.vllm_url}/metrics", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
            for line in body.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    match = re.search(r"[\d.]+$", line)
                    if match:
                        val = float(match.group())
                        if val > 0:
                            self._last_request_time = time.time()
                        return val
        except (urllib.error.URLError, OSError, ValueError):
            pass
        return None

    def _is_idle(self) -> bool:
        """Return ``True`` when vLLM has been idle beyond the threshold."""
        self._check_vllm_request_rate()
        return (time.time() - self._last_request_time) >= self.idle_threshold_s

    # ------------------------------------------------------------------
    # Decision engine
    # ------------------------------------------------------------------

    def decide(self) -> str:
        """Return one of: ``keep_serving``, ``enter_training``,
        ``continue_training``, ``resume_serving``."""
        mode = self.get_gpu_mode()

        if mode == "training":
            if self.is_training_pending():
                return "continue_training"
            return "resume_serving"

        # mode == "serving"
        if not self.is_training_pending():
            return "keep_serving"

        if self._is_idle():
            return "enter_training"

        return "keep_serving"

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def enter_training_mode(self) -> None:
        """Stop vLLM, run training, clear queue."""
        logger.info("=== Entering training mode ===")
        self.set_gpu_mode("training")

        # Drain in-flight requests
        logger.info("Draining in-flight requests (10 s) …")
        time.sleep(10)

        # Stop the unified vLLM service
        try:
            _systemctl("stop", "docwain-vllm-fast")
        except subprocess.CalledProcessError:
            logger.warning("Failed to stop docwain-vllm-fast (may already be stopped)")

        # Run training
        self._run_training()

        # Hot-swap model
        self._hot_swap_model()

        # Clear queue
        self.clear_training_queue()
        logger.info("=== Training mode complete ===")

    def resume_serving_mode(self) -> None:
        """Hot-swap model and restart vLLM."""
        logger.info("=== Resuming serving mode ===")
        self._hot_swap_model()

        # Start the unified vLLM service
        try:
            _systemctl("start", "docwain-vllm-fast")
        except subprocess.CalledProcessError:
            logger.error("Failed to start docwain-vllm-fast")

        self.set_gpu_mode("serving")
        self._last_request_time = time.time()
        logger.info("=== Serving mode resumed ===")

    def _run_training(self) -> None:
        """Launch the autonomous trainer as a subprocess."""
        cmd = [
            sys.executable,
            "-m", "src.finetune.v2.autonomous_trainer",
            "--resume",
        ]
        logger.info("Running training: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=14400,  # 4 h hard cap
        )
        if result.returncode != 0:
            logger.error("Training failed (rc=%d): %s", result.returncode, result.stderr[-2000:])
        else:
            logger.info("Training finished successfully")

    def _hot_swap_model(self) -> None:
        """Update the ``models/docwain-v2-active`` symlink to the latest
        checkpoint recorded in ``state.json``."""
        try:
            state = json.loads(STATE_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Cannot read state.json — skipping hot-swap")
            return

        last_ckpt = state.get("last_checkpoint")
        if not last_ckpt:
            # Fall back to best iteration merged dir from track_history
            for _track, hist in state.get("track_history", {}).items():
                iters = hist.get("iterations", [])
                if iters:
                    last_ckpt = iters[-1].get("merged_dir")
            if not last_ckpt:
                logger.warning("No checkpoint found in state.json — skipping hot-swap")
                return

        ckpt_path = Path(last_ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path

        if not ckpt_path.exists():
            logger.error("Checkpoint path %s does not exist — skipping hot-swap", ckpt_path)
            return

        MODEL_SYMLINK.parent.mkdir(parents=True, exist_ok=True)
        if MODEL_SYMLINK.is_symlink() or MODEL_SYMLINK.exists():
            MODEL_SYMLINK.unlink()
        MODEL_SYMLINK.symlink_to(ckpt_path)
        logger.info("Model symlink updated: %s → %s", MODEL_SYMLINK, ckpt_path)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_loop(self, interval_s: int = 300) -> None:
        """Poll every *interval_s* seconds and act on the decision."""
        logger.info("GPU scheduler loop started (interval=%d s)", interval_s)
        while True:
            try:
                decision = self.decide()
                logger.info("Decision: %s", decision)
                if decision == "enter_training":
                    self.enter_training_mode()
                elif decision == "resume_serving":
                    self.resume_serving_mode()
                # keep_serving / continue_training → no-op
            except Exception:
                logger.exception("Scheduler tick failed")
            time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="DocWain GPU Scheduler")
    parser.add_argument("--once", action="store_true", help="Single check then exit")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval in seconds")
    parser.add_argument("--idle-minutes", type=int, default=30, help="Idle threshold in minutes")
    parser.add_argument("--gpu-mode-file", default=DEFAULT_GPU_MODE_FILE)
    parser.add_argument("--training-queue-file", default=DEFAULT_TRAINING_QUEUE_FILE)
    args = parser.parse_args()

    sched = GPUScheduler(
        gpu_mode_file=args.gpu_mode_file,
        training_queue_file=args.training_queue_file,
        idle_threshold_minutes=args.idle_minutes,
    )

    if args.once:
        decision = sched.decide()
        logger.info("Decision: %s", decision)
        if decision == "enter_training":
            sched.enter_training_mode()
        elif decision == "resume_serving":
            sched.resume_serving_mode()
    else:
        sched.run_loop(interval_s=args.interval)


if __name__ == "__main__":
    main()
